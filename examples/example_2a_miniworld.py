
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.envs.gym import make as gym_make, RLPYT_WRAPPER_KEY
from rlpyt.envs.wrappers import TransposeImageWrapper
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.miniworld import MiniWorldFfAgent as Agent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.experiments.configs.mujoco.pg.mujoco_ppo import configs


def build_and_train(env_id="MiniWorld-OneRoomTransfer-v0", run_ID=0, cuda_idx=None, n_parallel=1):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)), alternating=True)
    env_args = dict(id=env_id)
    env_args[RLPYT_WRAPPER_KEY] = [TransposeImageWrapper]
    # sampler = AlternatingSampler(
    #     EnvCls=gym_make,
    #     env_kwargs=env_args,
    #     eval_env_kwargs=env_args,
    #     batch_T=256,  # One time-step per sampler iteration.
    #     batch_B=12,  # One environment (i.e. sampler Batch dimension).
    #     max_decorrelation_steps=100,
    #     # eval_n_envs=5,
    #     # eval_max_steps=int(25e3),
    #     # eval_max_trajectories=30
    # )
    # sampler = GpuSampler(
    #     EnvCls=gym_make,
    #     env_kwargs=env_args,
    #     eval_env_kwargs=env_args,
    #     batch_T=256,  # One time-step per sampler iteration.
    #     batch_B=12,  # One environment (i.e. sampler Batch dimension).
    #     max_decorrelation_steps=100,
    #     # eval_n_envs=5,
    #     # eval_max_steps=int(25e3),
    #     # eval_max_trajectories=30
    # )
    #
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=env_args,
        eval_env_kwargs=env_args,
        batch_T=256,  # One time-step per sampler iteration.
        batch_B=8,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        # eval_n_envs=2,
        # eval_max_steps=int(51e2),
        # eval_max_trajectories=5,
    )

    algo = PPO(clip_vf_loss=False, normalize_rewards=None)  # Run with defaults.
    agent = Agent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=affinity,
        transfer=True,
        transfer_iter=150,
        log_traj_window=10
    )
    config = dict(env_id=env_id)
    name = "ppo_" + env_id
    log_dir = "example_2a_miniworld"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='MiniWorld-OneRoomTransfer-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
