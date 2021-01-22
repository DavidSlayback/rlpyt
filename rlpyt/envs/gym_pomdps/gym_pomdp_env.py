import gym
import gym_pomdps

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from collections import namedtuple

env_list = gym_pomdps.env_list
EnvInfo = namedtuple("EnvInfo", ["reward_cat", "state"])

class POMDPEnv(Env):
    def __init__(self, pomdp_env):
        assert pomdp_env in env_list
        self.episodic = pomdp_env.split('-')[2] == 'episodic'
        self.env = gym.make(pomdp_env)
        self._action_space = IntBox(low=0, high=self.env.action_space.n)
        nobs = self.env.observation_space.n
        self._observation_space = IntBox(low=0, high=self.env.observation_space.n+1)
        self._start_obs = nobs

    def step(self, action):
        o, r, d, info = self.env.step(action)
        return EnvStep(o, r, d, EnvInfo(**info, state=self.state))

    def reset(self):
        self.env.reset()  # Reset state
        return self._start_obs  # Meaningless observation with no action

    def render(self, mode='rgb_array'):
        pass

    @property
    def state(self):
        return self.env.state

class FOMDPEnv(Env):
    def __init__(self, pomdp_env):
        assert pomdp_env in env_list
        self.episodic = pomdp_env.split('-')[2] == 'episodic'
        self.env = gym.make(pomdp_env)
        self._action_space = IntBox(low=0, high=self.env.action_space.n)
        nobs = self.env.observation_space.n
        self._observation_space = IntBox(low=0, high=self.env.state_space.n)  # state
        self._start_obs = nobs

    def step(self, action):
        o, r, d, info = self.env.step(action)
        return EnvStep(self.state, r, d, EnvInfo(**info, state=self.state))

    def reset(self):
        self.env.reset()  # Reset state
        return self.state

    def render(self, mode='rgb_array'):
        pass

    @property
    def state(self):
        return self.env.state