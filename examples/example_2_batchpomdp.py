
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

import numpy as np

# Samplers
from rlpyt.samplers.serial.batchpomdp import BatchPOMDPSampler

# Env
from rlpyt.envs.gym_pomdps.gym_pomdp_env import BatchPOMDPEnv

# Algos
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppoc import PPOC
from rlpyt.algos.pg.a2oc import A2OC

# Agents
from rlpyt.agents.pg.pomdp import (PomdpFfAgent, PomdpRnnAgent, PomdpOcFfAgent, AlternatingPomdpRnnAgent,
                                   AlternatingPomdpOcRnnAgent, AlternatingPomdpOcFfAgent, PomdpOcRnnAgent)
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context


def build_and_train(env_id="POMDP-hallway-episodic-v0", run_ID=0, cuda_idx=None, n_parallel=6, fomdp=False):
    EnvCls = BatchPOMDPEnv
    SamplerCls = BatchPOMDPSampler
    batch_B = 30
    batch_T = 100
    env_args = dict(fomdp=fomdp, id=env_id, time_limit=100, batch_B=batch_B)
    env = EnvCls(**env_args)
    gamma = env.discount
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)), alternating=False)
    lr = 1e-3
    po = np.array([1,0,0,1,0], dtype=bool)
    # Model kwargs
    # model_kwargs = dict()
    # model_kwargs = dict(hidden_sizes=[64, 64], shared_processor=False)
    model_kwargs = dict(hidden_sizes=[64, 64], rnn_type='gru', rnn_size=256, rnn_placement=1, shared_processor=False, layer_norm=True, prev_action='All', prev_reward='All')
    # model_kwargs = dict(hidden_sizes=[64, 64], option_size=4, shared_processor=False, use_interest=False, use_diversity=False, use_attention=False)
    # model_kwargs = dict(hidden_sizes=[64, 64], option_size=4, use_interest=True, use_diversity=False,
    #                     use_attention=False, rnn_type='gru', rnn_size=256, rnn_placement=1, shared_processor=False, layer_norm=True, prev_option=po)
    sampler = SamplerCls(env, batch_T, max_decorrelation_steps=0)

    # Samplers

    # Algos (swapping out discount)
    algo = A2C(discount=gamma, learning_rate=lr, clip_grad_norm=2.)
    # algo = A2OC(discount=gamma, learning_rate=lr, clip_grad_norm=2.)
    # algo = PPO(discount=gamma, learning_rate=lr, clip_grad_norm=2.)
    # algo = PPOC(discount=gamma, learning_rate=lr, clip_grad_norm=2.)

    # Agents
    # agent = PomdpFfAgent(model_kwargs=model_kwargs)
    agent = PomdpRnnAgent(model_kwargs=model_kwargs)
    # agent = PomdpOcFfAgent(model_kwargs=model_kwargs)
    # agent = PomdpOcRnnAgent(model_kwargs=model_kwargs)
    # agent = AlternatingPomdpRnnAgent(model_kwargs=model_kwargs)
    # agent = AlternatingPomdpOcRnnAgent(model_kwargs=model_kwargs)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=affinity,
    )
    config = dict(env_id=env_id, fomdp=fomdp, algo_name=algo.__class__.__name__, learning_rate=lr,
                  sampler=sampler.__class__.__name__, model=model_kwargs)
    name = algo.NAME + '_' + env_id
    log_dir = "pomdps"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='POMDP-tiger-continuing-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--fomdp', help='Set true if fully observable ', type=bool, default=False)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        fomdp=args.fomdp
    )
