
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
from rlpyt.envs.gym_procgen.procgen_env import ProcgenEnv
from rlpyt.envs.wrappers import TransposeImageWrapper, RLPYT_WRAPPER_KEY
from rlpyt.algos.pg.ppoc import PPOC
from rlpyt.agents.pg.procgen import ProcgenOcAgent as Agent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.experiments.configs.mujoco.pg.mujoco_ppo import configs


def build_and_train(game="fruitbot", run_ID=0, cuda_idx=None, n_parallel=6):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)), alternating=True)
    env_args = dict(game=game, start_level=0, num_levels=1)
    # sampler = AlternatingSampler(
    #     EnvCls=ProcgenEnv,
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
    #     EnvCls=ProcgenEnv,
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
        EnvCls=ProcgenEnv,
        env_kwargs=env_args,
        eval_env_kwargs=env_args,
        batch_T=256,  # One time-step per sampler iteration.
        batch_B=8,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        # eval_n_envs=2,
        # eval_max_steps=int(51e2),
        # eval_max_trajectories=5,
    )

    algo = PPOC(clip_vf_loss=False, normalize_rewards=None)  # Run with defaults.
    agent = Agent(model_kwargs={'option_size': 2})
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=affinity,
        # transfer=True,
        # transfer_iter=150,
        # log_traj_window=10
    )
    config = dict(game=game)
    name = "ppo_" + game
    log_dir = "example_2a_fruitbot"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    # import multiprocessing as mp
    # mp.set_start_method('spawn')
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Procgen game', default='fruitbot')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
