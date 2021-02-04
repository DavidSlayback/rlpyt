
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
from rlpyt.envs.bsuiteenv.bsuite_env import BSuiteEnv

# Algos
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppoc import PPOC
from rlpyt.algos.pg.a2oc import A2OC

# Agents
from rlpyt.agents.pg.bsuite import (BsuiteFfAgent, BsuiteRnnAgent, BsuiteOcFfAgent, AlternatingBsuiteRnnAgent,
                                   AlternatingBsuiteOcRnnAgent, AlternatingBsuiteOcFfAgent, BsuiteOcRnnAgent)
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context


def build_and_train(env_id="catch/0", run_ID=0, cuda_idx=None, n_parallel=6, fomdp=False):
    EnvCls = BSuiteEnv
    n_episodes = 1e4
    env_args = dict(id=env_id)
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)), alternating=True)
    lr = 1e-3

    # Model kwargs
    # model_kwargs = dict()
    model_kwargs = dict(hidden_sizes=[64, 64])
    # model_kwargs = dict(hidden_sizes=[64, 64], rnn_type='gru', rnn_size=128)
    # model_kwargs = dict(hidden_sizes=[64, 64], option_size=4, use_interest=False, use_diversity=False, use_attention=False)
    # model_kwargs = dict(hidden_sizes=[64, 64], option_size=4, use_interest=False, use_diversity=False,
    #                     use_attention=False, rnn_type='gru', rnn_size=128)

    # Samplers
    # sampler = AlternatingSampler(
    #     EnvCls=EnvCls,
    #     env_kwargs=env_args,
    #     eval_env_kwargs=env_args,
    #     batch_T=20,  # One time-step per sampler iteration.
    #     batch_B=30,  # One environment (i.e. sampler Batch dimension).
    #     max_decorrelation_steps=0,
    #     eval_n_envs=5,
    #     eval_max_steps=int(25e3),
    #     eval_max_trajectories=30
    # )
    #
    sampler = SerialSampler(
        EnvCls=EnvCls,
        env_kwargs=env_args,
        eval_env_kwargs=env_args,
        batch_T=32,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        # eval_n_envs=2,
        # eval_max_steps=int(51e2),
        # eval_max_trajectories=5,
    )

    # Algos (swapping out discount)
    algo = A2C(learning_rate=lr, clip_grad_norm=2.)
    # algo = A2OC(discount=gamma, learning_rate=lr, clip_grad_norm=2.)

    # Agents
    agent = BsuiteFfAgent(model_kwargs=model_kwargs)
    # agent = BsuiteRnnAgent(model_kwargs=model_kwargs)
    # agent = BsuiteOcFfAgent(model_kwargs=model_kwargs)
    # agent = BsuiteOcRnnAgent(model_kwargs=model_kwargs)
    # agent = AlternatingBsuiteRnnAgent(model_kwargs=model_kwargs)
    # agent = AlternatingBsuiteRnnAgent(model_kwargs=model_kwargs)
    # agent = AlternatingBsuiteOcRnnAgent(model_kwargs=model_kwargs)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=affinity,
    )
    config = dict(env_id=env_id, fomdp=fomdp, algo_name=algo.__class__.__name__, learning_rate=lr)
    name = algo.NAME + '_' + env_id
    log_dir = "Bsuites"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='catch/0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--fomdp', help='Set true if fully observable ', type=bool, default=True)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        fomdp=args.fomdp
    )
