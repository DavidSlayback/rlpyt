import gym
import gym_pomdps
import numpy as np

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from collections import namedtuple

env_list = gym_pomdps.env_list
EnvInfo = namedtuple("EnvInfo", ["reward_cat", "state"])

# Compiled optimal discounted returns for some of the POMDPs (optimal given observation).
# Primarily from Anthony Cassandra's thesis, using discount factors from .pomdp files
OPTIMAL_RETURNS = {
    'POMDP-hallway-episodic-v0': 1.008,  # No optimal, best reported (7-PWLC)
    'POMDP-hallway2-episodic-v0': 0.510,  # No optimal, best reported (7-PWLC)
    'POMDP-4x3-episodic-v0': 1.883,  # No optimal, best reported (ApproxVI)
    'POMDP-4x4-episodic-v0': 3.712,
    'POMDP-cheese-episodic-v0': 3.464,
    'POMDP-tiger-episodic-v0': 19.181,
    'POMDP-network-episodic-v0': 290.998,
    'POMDP-saci_s12_a6_z5-episodic-v0': 14.817,  # No optimal, best reported (ApproxVI)
    'POMDP-shuttle-episodic-v0': 32.7,
    'POMDP-rock_sample_7_8-episodic-v0': 21.27,
    'POMDP-mit-episodic-v0': 0.86,
    'POMDP-aloha30-episodic-v0': 852.773,  # No optimal, best reported (MLS)
    'POMDP-baseball-episodic-v0': 0.668,  # No optimal, best reported (WE)
    'POMDP-machine-episodic-v0': 59.884,  # No optimal, best reported (ApproxVI)
    'POMDP-paint-episodic-v0': 3.279
}

# Compiled omnisicent discounted returns for some of the POMDPs (optimal given state)
OMNISCIENT_RETURNS = {
    'POMDP-hallway-episodic-v0': 1.519,
    'POMDP-hallway2-episodic-v0': 1.189,
    'POMDP-4x3-episodic-v0': 2.466,
    'POMDP-4x4-episodic-v0': 4.654,
    'POMDP-cheese-episodic-v0': 3.910,
    'POMDP-tiger-episodic-v0': 198.816,
    'POMDP-network-episodic-v0': 490.997,
    'POMDP-saci_s12_a6_z5-episodic-v0': 16.904,
    'POMDP-shuttle-episodic-v0': 32.7,
    'POMDP-mit-episodic-v0': 0.894,
    'POMDP-paint-episodic-v0': 12.678
}

def pomdp_interface(fomdp=False, **kwargs):
    return FOMDPEnv(**kwargs) if fomdp else POMDPEnv(**kwargs)

class POMDPEnv(Env):
    def __init__(self, id):
        assert id in env_list
        self.episodic = id.split('-')[2] == 'episodic'
        self.env = gym.make(id)
        self.discount = self.env.discount
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
        self.discount = self.env.discount
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