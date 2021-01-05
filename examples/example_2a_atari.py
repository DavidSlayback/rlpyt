
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.envs.wrappers import ClipActionsWrapper, RLPYT_WRAPPER_KEY
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.pg.a2oc import A2OC
from rlpyt.agents.pg.atari import AtariOcAgent, AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.experiments.configs.mujoco.pg.mujoco_ppo import configs
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo


def build_and_train(game="montezuma_revenge", run_ID=0, cuda_idx=None, n_parallel=6):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)), alternating=True)
    env_args = dict(id=game)
    # env_args[RLPYT_WRAPPER_KEY] = [ClipActionsWrapper]
    sampler = AlternatingSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,
        env_kwargs=dict(game=game),
        batch_T=64,  # One time-step per sampler iteration.
        batch_B=36,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=1000,
        # eval_n_envs=5,
        # eval_max_steps=int(25e3),
        # eval_max_trajectories=30
    )
    #
    # sampler = SerialSampler(
    #     EnvCls=AtariEnv,
    #     TrajInfoCls=AtariTrajInfo,
    #     env_kwargs=dict(game=game),
    #     batch_T=256,  # One time-step per sampler iteration.
    #     batch_B=8,  # One environment (i.e. sampler Batch dimension).
    #     max_decorrelation_steps=1000,
    #     # eval_n_envs=2,
    #     # eval_max_steps=int(51e2),
    #     # eval_max_trajectories=5,
    # )

    # algo = PPO(clip_vf_loss=False, normalize_rewards=None)  # Run with defaults.
    algo = A2OC(normalize_rewards=None)
    agent = AtariOcAgent(model_kwargs={'option_size': 4})
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
    log_dir = "example_2a_atari"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='environment ID', default='montezuma_revenge')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
