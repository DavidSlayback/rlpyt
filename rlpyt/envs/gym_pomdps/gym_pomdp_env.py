import gym
import gym_pomdps
import numpy as np

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from collections import namedtuple

env_list = gym_pomdps.env_list
EnvInfo = namedtuple("EnvInfo", ["reward_cat", "state", "episode_done"])

# Compiled optimal discounted returns for some of the POMDPs (optimal given observation).
# Primarily from Anthony Cassandra's thesis, using discount factors from .pomdp files
# Note his thesis truncates episodes to 100 steps, so I need to use continuing environment variant
OPTIMAL_RETURNS = {
    'POMDP-hallway-continuing-v0': 1.008,  # No optimal, best reported (7-PWLC)
    'POMDP-hallway2-continuing-v0': 0.510,  # No optimal, best reported (7-PWLC)
    'POMDP-4x3-continuing-v0': 1.883,  # No optimal, best reported (ApproxVI)
    'POMDP-4x4-continuing-v0': 3.712,
    'POMDP-cheese-continuing-v0': 3.464,
    'POMDP-tiger-continuing-v0': 19.181,
    'POMDP-network-continuing-v0': 290.998,
    'POMDP-saci_s12_a6_z5-continuing-v0': 14.817,  # No optimal, best reported (ApproxVI)
    'POMDP-shuttle-continuing-v0': 32.7,
    'POMDP-rock_sample_7_8-continuing-v0': 21.27,
    'POMDP-mit-continuing-v0': 0.86,
    'POMDP-aloha30-continuing-v0': 852.773,  # No optimal, best reported (MLS)
    'POMDP-baseball-continuing-v0': 0.668,  # No optimal, best reported (WE)
    'POMDP-machine-continuing-v0': 59.884,  # No optimal, best reported (ApproxVI)
    'POMDP-paint-continuing-v0': 3.279
}

# Compiled omnisicent discounted returns for some of the POMDPs (optimal given state)
OMNISCIENT_RETURNS = {
    'POMDP-hallway-continuing-v0': 1.519,
    'POMDP-hallway2-continuing-v0': 1.189,
    'POMDP-4x3-continuing-v0': 2.466,
    'POMDP-4x4-continuing-v0': 4.654,
    'POMDP-cheese-continuing-v0': 3.910,
    'POMDP-tiger-continuing-v0': 198.816,
    'POMDP-network-continuing-v0': 490.997,
    'POMDP-saci_s12_a6_z5-continuing-v0': 16.904,
    'POMDP-shuttle-continuing-v0': 32.7,
    'POMDP-mit-continuing-v0': 0.894,
    'POMDP-paint-continuing-v0': 12.678
}

def pomdp_interface(fomdp=False, **kwargs):
    return FOMDPEnv(**kwargs) if fomdp else POMDPEnv(**kwargs)

class POMDPEnv(Env):
    """ Partially observable environment wrapped to fit rlpyt

    Args:
        id (str): Gym id of gym_pomdp environment (must be a valid id, which is determined by .pomdp files in local installation of gym-pomdps)
        time_limit (int): Maximum number of timesteps before episode termination. Defaults to None (no timelimit)
        report_episode_done (bool): If environment is not episodic, external agents may still want to reset state at the boundary.
            Setting this to true will provide this information in the info returned.
    """
    def __init__(self, id: str, time_limit: int = None, report_episode_done: bool = False):
        assert id in env_list
        self.episodic = id.split('-')[2] == 'episodic'
        self.report_episode_done = report_episode_done
        if not self.episodic and report_episode_done: id = id.replace('continuing', 'episodic')  # Even in continuing setting, use episodic env
        self.env = gym.make(id)
        self.discount = self.env.discount
        self._action_space = IntBox(low=0, high=self.env.action_space.n)
        nobs = self.env.observation_space.n
        self._observation_space = IntBox(low=0, high=self.env.observation_space.n+1)
        self._start_obs = nobs
        self.time_limit = time_limit
        self.time_elapsed = 0

    def step(self, action):
        episode_done = False
        o, r, d, info = self.env.step(action)
        # s = self.state  # Record state before potential reset
        if d and not self.episodic and self.report_episode_done:
            d = False
            episode_done = True  # Tell agent to reset state
            self.env.reset()  # Our underlying env is episodic, need to reset
        s = self.state
        self.time_elapsed += 1
        if self.time_limit is not None: d = self.time_elapsed >= self.time_limit or d
        return EnvStep(np.array(o), r, d, EnvInfo(**info, state=s, episode_done=episode_done))

    def reset(self):
        self.env.reset()  # Reset state
        self.time_elapsed = 0
        return self._start_obs  # Meaningless observation with no action

    def render(self, mode='rgb_array'):
        pass

    @property
    def state(self):
        return self.env.state

class FOMDPEnv(Env):
    """ Fully observable environment wrapped to fit rlpyt

    Args:
        id (str): Gym id of gym_pomdp environment (must be a valid id, which is determined by .pomdp files in local installation of gym-pomdps)
        time_limit (int): Maximum number of timesteps before episode termination. Defaults to None (no timelimit)
        report_episode_done (bool): If environment is not episodic, external agents may still want to reset state at the boundary.
            Setting this to true will provide this information in the info returned.
    """
    def __init__(self, id: str, time_limit: int = None, report_episode_done: bool = False):
        assert id in env_list
        self.episodic = id.split('-')[2] == 'episodic'
        self.report_episode_done = report_episode_done
        if not self.episodic and report_episode_done: id = id.replace('continuing', 'episodic')  # Even in continuing setting, use episodic env
        self.env = gym.make(id)
        self.discount = self.env.discount
        self._action_space = IntBox(low=0, high=self.env.action_space.n)
        nobs = self.env.observation_space.n
        self._observation_space = IntBox(low=0, high=self.env.state_space.n)  # state == -1 if episodic
        self._start_obs = nobs
        self.time_limit = time_limit
        self.time_elapsed = 0

    def step(self, action):
        episode_done = False
        o, r, d, info = self.env.step(action)
        if d and not self.episodic and self.report_episode_done:
            d = False
            episode_done = True  # Tell agent to reset state
            self.env.reset()  # Our underlying env is episodic, need to reset
        self.time_elapsed += 1
        s = self.state
        if self.time_limit is not None: d = self.time_elapsed >= self.time_limit or d
        return EnvStep(np.array(s), r, d, EnvInfo(**info, state=s, episode_done=episode_done))

    def reset(self):
        self.env.reset()  # Reset state
        self.time_elapsed = 0
        return self.state

    def render(self, mode='rgb_array'):
        pass

    @property
    def state(self):
        return self.env.state