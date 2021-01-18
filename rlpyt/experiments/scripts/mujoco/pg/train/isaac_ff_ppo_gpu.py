from rlpyt.envs.gym_isaac.isaacgym_env import IsaacGymEnv
import sys

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.serial.isaac import IsaacSampler
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.experiments.configs.mujoco.pg.nv_ppo import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)
    env = IsaacGymEnv(config['env']['task'])  # Make env
    import torch.nn as nn
    config["model"]["hidden_nonlinearity"] = getattr(nn, config["model"]["hidden_nonlinearity"])  # Replace string with proper activation
    sampler = IsaacSampler(env, **config["sampler"])
    algo = PPO(optim_kwargs=config["optim"], **config["algo"])
    agent = MujocoFfAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = "ppo_nv_" + config["env"]["task"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
