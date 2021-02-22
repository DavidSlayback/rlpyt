from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.buffer import build_samples_buffer, get_example_outputs_single
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer, buffer_from_example
from rlpyt.utils.logging import logger
from rlpyt.samplers.collections import BatchSpec, TrajInfo
from rlpyt.samplers.prevec.collections import TrajInfoVec
import numpy as np

class BatchPOMDPSampler:
    """ Specially made serial sampler variant for batched pomdps"""
    alternating = False  # Can't alternate, just one cuda env
    mid_batch_reset = False
    def __init__(self,
                 env,  # Instantiated BatchPOMDPEnv
                 batch_T,  # Number of timesteps per sampled batch
                 max_decorrelation_steps=0,  # Number of random actions to take at start,
                 **_  # throw away the rest
                 ):
        self.env = env
        self.batch_spec = BatchSpec(batch_T, env.num_envs)
        self.decorrelation_steps = max_decorrelation_steps
        self.TrajInfoCls = TrajInfoVec
        self.ReturnTrajInfoCls = TrajInfo

    def initialize(self,
                   agent,
                   affinity=None,
                   seed=None,
                   bootstrap_value=False,
                   traj_info_kwargs=None,
                   rank=0,
                   world_size=1, ):
        """Should instantiate all components, including setup of parallel
        process if applicable."""
        B = self.batch_spec.B
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        agent.initialize(self.env.spaces, share_memory=False, global_B=global_B, env_ranks=env_ranks)
        self.env.seed(seed)
        examples = dict()
        get_example_outputs_single(agent, self.env, examples, subprocess=False)
        samples_pyt, samples_np, examples = build_samples_buffer(agent, self.env,
            self.batch_spec, bootstrap_value, agent_shared=False,
            env_shared=False, subprocess=False, examples=examples)
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
                setattr(self.ReturnTrajInfoCls, "_" + k, v)
        self.agent_inputs, self.traj_infos = self._decorrelate_envs()
        # Collector calls start_agent here, but doesn't apply
        self.agent = agent
        logger.log("Pomdp Sampler initialized.")
        return examples

    def _decorrelate_envs(self):
        """Return agent_inputs and traj_info at the end of decorrelation using random actions (collector.start_envs)"""
        o = self.env.reset()
        prev_observation = buffer_from_example(o[0], self.batch_spec.B)
        prev_reward = np.zeros(self.batch_spec.B, dtype="float32")
        prev_action = np.zeros(self.batch_spec.B, dtype=int)
        traj_infos = self.TrajInfoCls(B=self.batch_spec.B)
        for _ in range(self.decorrelation_steps):
            prev_action[:] = self.env.action_space.sample()  # Sample random actions for each
            prev_observation[:], prev_reward[:], d, info = self.env.step(prev_action[:])  # Take step
            traj_infos.step(prev_observation, prev_action, prev_reward, d, None, info, reset_dones=True)  # Update traj_info

        return AgentInputs(prev_observation, prev_action, prev_reward), traj_infos

    def obtain_samples(self, itr):
        """Execute agent-environment interactions and return data batch."""
        self.agent_inputs, self.traj_infos, completed_infos = self._collect_batch(itr)
        return self.samples_pyt, completed_infos

    def _collect_batch(self, itr):
        """Collect batch of experience from environment (collector.collect_batch)"""
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        o, a, r = self.agent_inputs  # Previous last inputs
        o_p, a_p, r_p = torchify_buffer(self.agent_inputs)
        self.agent.sample_mode(itr)
        agent_buf.prev_action[0] = a  # Store previous action
        env_buf.prev_reward[0] = r  # Store previous reward
        for t in range(self.batch_spec.T):
            env_buf.observation[t] = o  # Store observation
            # Agent inputs and outputs are torch tensors.
            a_p, agent_info = self.agent.step(o_p, a_p, r_p)
            a = numpify_buffer(a_p)
            o[:], r[:], d, info = self.env.step(a)
            self.traj_infos.step(o, a, r, d, agent_info, info, reset_dones=False)
            # Get completed infos (non-tensor). Environment auto-resets
            completed_infos += self.traj_infos.terminate(d)
            if np.sum(d): self.agent.reset_multiple(indexes=d)
            env_buf.done[t] = d
            env_buf.reward[t] = r
            agent_buf.action[t] = a
            if info:
                env_buf.env_info[t] = info
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(o_p, a_p, r_p)

        return AgentInputs(o, a, r), self.traj_infos, completed_infos

    def evaluate_agent(self, itr):
        """Run offline agent evaluation, if applicable."""
        raise NotImplementedError

    def transfer(self, arg=0.):
        """Transfer task in gym training and evaluation environments, if applicable"""
        self.env.transfer(arg)

    def shutdown(self):
        pass

    @property
    def batch_size(self):
        return self.batch_spec.size  # For logging at least.