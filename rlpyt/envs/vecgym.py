"""
Gym wrappers for vectorized environments
"""

import numpy as np
import gym
from gym import Wrapper
import gym3
from gym.wrappers.time_limit import TimeLimit
from collections import namedtuple
import paper_gym
import gym_minigrid
import gym_miniworld

from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.envs.wrappers import RLPYT_WRAPPER_KEY
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import is_namedtuple_class

class Gym3Wrapper:
    def __init__(self, env, act_null_value=0, obs_null_value=0, force_float32=True):
        self.env = gym3.interop.ToBaselinesVecEnv(env)



class GymVecEnvWrapper(Wrapper):
    def __init__(self, env, act_null_value=0, obs_null_value=0, force_float32=True):
        super().__init__(env)
        o = self.env.reset()
        b = o.shape[0]  # Infer batch dimension from first part of shape
        o, r, d, info = self.env.step(self.env.action_space.sample())