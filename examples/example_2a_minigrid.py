
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
from rlpyt.envs.wrappers import RLPYT_WRAPPER_KEY
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.minigrid import MinigridFfAgent, MinigridGruAgent, AlternatingMinigridGruAgent, AlternatingMinigridBabyAIAgent, AlternatingMinigridAllenActAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.experiments.configs.minigrid.pg.minigrid_ff_ppo import configs

basecfg = configs['BaseMiniGrid']
def build_and_train(env_id="MiniGrid-FourRooms-v0", run_ID=0, cuda_idx=None, n_parallel=6):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)), alternating=True)
    env_args = dict(id=env_id)
    env_args[RLPYT_WRAPPER_KEY] = basecfg["env"][RLPYT_WRAPPER_KEY]
    sampler = AlternatingSampler(
        EnvCls=gym_make,
        env_kwargs=env_args,
        eval_env_kwargs=env_args,
        batch_T=64,  # One time-step per sampler iteration.
        batch_B=18,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=100,
        # eval_n_envs=5,
        # eval_max_steps=int(25e3),
        # eval_max_trajectories=30
    )
    #
    # sampler = SerialSampler(
    #     EnvCls=gym_make,
    #     env_kwargs=env_args,
    #     eval_env_kwargs=env_args,
    #     batch_T=256,  # One time-step per sampler iteration.
    #     batch_B=8,  # One environment (i.e. sampler Batch dimension).
    #     max_decorrelation_steps=100,
    #     # eval_n_envs=2,
    #     # eval_max_steps=int(51e2),
    #     # eval_max_trajectories=5,
    # )

    algo = PPO(clip_vf_loss=False, normalize_rewards=None)  # Run with defaults.
    agent = AlternatingMinigridAllenActAgent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=5e6,
        log_interval_steps=1e3,
        affinity=affinity,
        transfer=False,
        log_traj_window=100
    )
    config = dict(env_id=env_id)
    name = "ppo_" + env_id
    log_dir = "example_2a"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='MiniGrid-FourRooms-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
