import sys

from rlpyt.utils.launching.affinity import affinity_from_code

# Samplers
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler

# Env
from rlpyt.envs.gym_pomdps.gym_pomdp_env import POMDPEnv, FOMDPEnv, pomdp_interface

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
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.experiments.configs.pomdp.pg.pomdp_a2c import configs

def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)
    config["algo_name"] = 'A2OC_D' + config["algo"]["delib_cost"]

    sampler = GpuSampler(
        EnvCls=pomdp_interface,
        env_kwargs=config["env"],
        **config["sampler"]
    )
    algo = A2OC(optim_kwargs=config["optim"], **config["algo"])
    agent = PomdpOcFfAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = config["env"]["id"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])