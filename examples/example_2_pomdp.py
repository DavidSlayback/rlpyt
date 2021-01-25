
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

# Samplers
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler

# Env
from rlpyt.envs.gym_pomdps.gym_pomdp_env import POMDPEnv, FOMDPEnv

# Algos
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppoc import PPOC
from rlpyt.algos.pg.a2oc import A2OC

# Agents
from rlpyt.agents.pg.pomdp import PomdpFfAgent, PomdpLstmAgent, PomdpFfOcAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context


def build_and_train(env_id="POMDP-hallway-episodic-v0", run_ID=0, cuda_idx=None, n_parallel=6):
    EnvCls = POMDPEnv
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)), alternating=True)
    env_args = dict(id=env_id)
    # sampler = AlternatingSampler(
    #     EnvCls=EnvCls,
    #     env_kwargs=env_args,
    #     eval_env_kwargs=env_args,
    #     batch_T=256,  # One time-step per sampler iteration.
    #     batch_B=8,  # One environment (i.e. sampler Batch dimension).
    #     max_decorrelation_steps=100,
    #     eval_n_envs=5,
    #     eval_max_steps=int(25e3),
    #     eval_max_trajectories=30
    # )
    #
    sampler = SerialSampler(
        EnvCls=EnvCls,
        env_kwargs=env_args,
        eval_env_kwargs=env_args,
        batch_T=256,  # One time-step per sampler iteration.
        batch_B=8,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        # eval_n_envs=2,
        # eval_max_steps=int(51e2),
        # eval_max_trajectories=5,
    )

    algo = A2C()
    agent = PomdpFfAgent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=affinity,
    )
    config = dict(env_id=env_id)
    name = "a2c_" + env_id
    log_dir = "example_2a"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='POMDP-hallway-episodic-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
