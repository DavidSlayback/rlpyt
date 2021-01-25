import gym
import gym_pomdps
import numpy as np

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from collections import namedtuple

env_list = gym_pomdps.env_list
EnvInfo = namedtuple("EnvInfo", ["reward_cat", "state"])

class POMDPEnv(Env):
    def __init__(self, id):
        assert id in env_list
        self.episodic = id.split('-')[2] == 'episodic'
        self.env = gym.make(id)
        self._action_space = IntBox(low=0, high=self.env.action_space.n)
        nobs = self.env.observation_space.n
        self._observation_space = IntBox(low=0, high=self.env.observation_space.n+1)
        self._start_obs = nobs

    def step(self, action):
        o, r, d, info = self.env.step(action)
        return EnvStep(np.array(o), r, d, EnvInfo(**info, state=self.state))

    def reset(self):
        self.env.reset()  # Reset state
        return self._start_obs  # Meaningless observation with no action

    def render(self, mode='rgb_array'):
        pass

    @property
    def state(self):
        return self.env.state

class FOMDPEnv(Env):
    def __init__(self, id):
        assert id in env_list
        self.episodic = id.split('-')[2] == 'episodic'
        self.env = gym.make(id)
        self._action_space = IntBox(low=0, high=self.env.action_space.n)
        nobs = self.env.observation_space.n
        self._observation_space = IntBox(low=0, high=self.env.state_space.n)  # state
        self._start_obs = nobs

    def step(self, action):
        o, r, d, info = self.env.step(action)
        return EnvStep(np.array(self.state), r, d, EnvInfo(**info, state=self.state))

    def reset(self):
        self.env.reset()  # Reset state
        return self.state

    def render(self, mode='rgb_array'):
        pass

    @property
    def state(self):
        return self.env.state