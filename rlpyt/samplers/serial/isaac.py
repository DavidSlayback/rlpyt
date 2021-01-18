from rlpyt.utils.logging import logger
from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.buffer import build_samples_buffer_torch_env
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.samplers.collections import BatchSpec, TrajInfo
from rlpyt.samplers.prevec.collections import TrajInfoVecGPU
import numpy as np
import torch

class IsaacSampler:
    """ Specially made serial sampler variant for isaacgym"""
    alternating = False  # Can't alternate, just one cuda env
    mid_batch_reset = False
    def __init__(self,
                 isaac_env,  # Instantiated isaac environment
                 batch_T,  # Number of timesteps per sampled batch
                 batch_B=None,  # Ignored, comes from env
                 max_decorrelation_steps=100  # Number of random actions to take at start,
                 ):
        self.batch_spec = BatchSpec(batch_T, isaac_env.num_envs)
        self.env = isaac_env
        self.decorrelation_steps = max_decorrelation_steps
        self.TrajInfoCls = TrajInfoVecGPU
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
        samples_pyt, samples_np, examples = build_samples_buffer_torch_env(agent, self.env,
            self.batch_spec, bootstrap_value, agent_shared=False,
            env_shared=False, subprocess=False)
        self.samples_pyt = buffer_to(samples_pyt, self.env.device)  # Pytorch buffer
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
                setattr(self.ReturnTrajInfoCls, "_" + k, v)
        if self.decorrelation_steps != 0:
            self.agent_inputs, self.traj_infos = self._decorrelate_envs()
        # Collector calls start_agent here, but doesn't apply
        self.agent = agent
        logger.log("Isaac Sampler initialized.")
        return examples

    def _decorrelate_envs(self):
        """Return agent_inputs and traj_info at the end of decorrelation using random actions (collector.start_envs)"""
        prev_action = torch.tensor(np.tile(self.env.action_space.null_value(), (self.batch_spec.B, 1)), device=self.env.device)
        prev_reward = torch.zeros((self.batch_spec.B,), device=self.env.device)
        observation = torch.zeros((self.batch_spec.B, self.env.observation_space.shape[0]), device=self.env.device)
        traj_infos = self.TrajInfoCls(B=self.batch_spec.B)

        self.env.reset()  # Reset all environments
        for _ in range(self.decorrelation_steps):
            a = self.env.action_space.sample()  # Sample random actions
            o, r, d, info = self.env.step(a)  # Take step
            traj_infos.step(o, a, r, d, None, info, reset_dones=True)  # Update traj_info

        observation[:], prev_action[:], prev_reward[:] = o, a, r
        return AgentInputs(o, a, r), traj_infos

    def obtain_samples(self, itr):
        """Execute agent-environment interactions and return data batch."""
        self.agent_inputs, self.traj_infos, completed_infos = self._collect_batch(itr)
        return self.samples_pyt, completed_infos

    def _collect_batch(self, itr):
        """Collect batch of experience from environment (collector.collect_batch)"""
        agent_buf, env_buf = self.samples_pyt.agent, self.samples_pyt.env
        completed_infos = list()
        o, a, r = self.agent_inputs  # Previous last inputs. Already torchified
        self.agent.sample_mode(itr)
        for t in range(self.batch_spec.T):
            agent_buf.prev_action[t] = a  # Store previous action
            env_buf.prev_reward[t] = r  # Store previous reward
            env_buf.observation[t] = o  # Store observation
            # Agent inputs and outputs are torch tensors.
            a, agent_info = self.agent.step(o, a, r)
            o, r, d, info = self.env.step(a)
            self.traj_infos.step(o, a, r, d, agent_info, info, reset_dones=False)
            # Get completed infos (non-tensor). Environment auto-resets
            completed_infos += self.traj_infos.terminate(d)
            if torch.sum(d): self.agent.reset_multiple(indexes=d.cpu().bool().numpy())
            env_buf.done[t] = d
            env_buf.reward[t] = r
            agent_buf.action[t] = a
            if info:
                env_buf.env_info[t] = info
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(o, a, r)

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