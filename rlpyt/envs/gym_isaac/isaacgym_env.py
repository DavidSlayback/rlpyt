import numpy as np
import os
import yaml
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.envs.base import Env, EnvStep
from rlgpu.utils.config import (
    get_args,  # default args
    set_np_formatting,  # Pretty printing format options
    set_seed,  # Seeds torch, numpy, random
    load_cfg,  # Call on args to get config
    parse_sim_params,  # Get params from config for sim
    retrieve_cfg
)
from rlgpu.utils.parse_task import parse_task
import importlib
import pathlib

VALID_TASKS = ['Ant', 'BallBalance', 'Cartpole', 'CartpoleYUp', 'Humanoid', 'FrankaCabinet', 'ShadowHand']
from rlpyt.samplers.collections import TrajInfo, namedtuple
EnvInfo = namedtuple("EnvInfo", ["timeout"])
import torch
class IsaacSpaceWrapper(GymSpaceWrapper):
    def __init__(self, num_envs, device='cuda', **kwargs):
        super().__init__(**kwargs)
        self.B = num_envs
        self.device = device

    def sample(self):
        samples = [self.space.sample() for _ in range(self.B)]
        return torch.tensor(samples, device=self.device)

class IsaacGymEnv(Env):
    """Interface for Nvidia's Isaacgym environments

    Isaacgym environments are like Mujoco environments, but special in two ways
    1) Mandatory vectorization: These are already vectorized/multiprocessed
    2) CUDA: By default, these are run purely on the GPU, meaning, obs and actions should also stay there

    Continuous action and observation spaces. Isaacgym clips obs (-5, 5) and actions (-1,1)

    Args:
        task: Name of isaacgym task (from above). There must be configuration files defined in isaacgym assets
        seed: Probably does nothing here
        headless: True for no rendering
        num_envs: 0 for default value from config file, anything else for equivalent to BatchSpec.B
        episode_length: 0 for default value from config file, otherwise equivalent to time limit
        randomize: If true, permute the physics parameters of the task each time it is reset

    """
    def __init__(self, task,
                 seed=0,
                 headless=True,
                 num_envs=0,
                 episode_length=1000,
                 randomize=False):
        assert task in VALID_TASKS
        base_args = get_args()
        base_args.headless = headless
        base_args.task = task
        base_args.seed = seed
        base_args.episode_length = episode_length
        base_args.num_envs = num_envs
        base_args.randomize = randomize
        base_args.logdir, base_args.cfg_train, base_args.cfg_env = retrieve_cfg(base_args, False)  # Update configs properly
        cfg, cfg_train, logdir = load_cfg(base_args)
        sim_params = parse_sim_params(base_args, cfg, cfg_train)
        self.task, self.env = parse_task(base_args, cfg, cfg_train, sim_params)  # Create environment
        self.num_envs = self.env.num_envs  # Number of environments
        self.device = self.env.rl_device  # cuda or cpu
        self._observation_space = IsaacSpaceWrapper(
            num_envs=self.num_envs,
            device=self.env.rl_device,
            space=self.env.observation_space,
            name="obs",
            force_float32=True,
        )
        self._action_space = IsaacSpaceWrapper(
            num_envs=self.num_envs,
            device=self.env.rl_device,
            space=self.env.action_space,
            name="act",
            force_float32=True,
        )

    def reset(self):
        return self.env.reset()

    def step(self, action):
        o, r, d, _ = self.env.step(action)
        return EnvStep(o, r, d, EnvInfo(timeout=False))

    def render(self):
        pass

    def seed(self):
        pass

    def close(self):
        pass

    def transfer(self, transfer_arg=None):
        self.env.transfer(transfer_arg)




#import inspect
# Get base path of isaacgym config files
basepath = pathlib.Path(
    importlib.machinery.PathFinder().find_module('isaacgym').get_filename()).parent.parent / 'rlgpu'
os.chdir(basepath)
"""
Important args from get_args
    headless: True to force display off
    task: Chooses task (e.g., "Ant", "Cartpole")
    device: "CPU" or "GPU" for sim
    ppo_device: As above, for network
    num_envs: 0 default, overrides yaml
    episode_length,
    randomize: Physics parameter randomization
"""
# Use the ant yaml file as the base
base_cfg_path = (basepath / 'cfg' / 'ant.yaml').as_posix()
base_cfg_train_path = (basepath / 'cfg' / 'train' / 'pytorch_ppo_ant.yaml').as_posix()
with open(os.path.join(base_cfg_path), 'r') as f:
    base_cfg = yaml.load(f, Loader=yaml.SafeLoader)

from rlgpu.utils.process_ppo import process_ppo
if __name__ == "__main__":
    test = IsaacGymEnv('Ant')
    set_np_formatting()
    args = get_args()
    args.max_iterations=2000
    args.headless = True
    # args.task, args.
    cfg, cfg_train, logdir = load_cfg(args)
    sim = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train['seed'])
    task, env = parse_task(args, cfg, cfg_train, sim)
    ppo = process_ppo(args, env, cfg_train, logdir)
    ppo_iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        ppo_iterations = args.max_iterations
    ppo.run(num_learning_iterations=ppo_iterations, log_interval=cfg_train["learn"]["save_interval"])
