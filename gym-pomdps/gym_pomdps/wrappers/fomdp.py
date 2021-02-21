import gym
import numpy as np

from gym_pomdps.envs import POMDP

__all__ = ['FullyObservablePOMDP', 'PartiallyObservablePOMDP', 'AutoresettingBatchPOMDP']

class FullyObservablePOMDP(gym.ObservationWrapper):
    """Returns state instead of observation"""
    def __init__(self, env):
        assert isinstance(env.unwrapped, POMDP)
        super().__init__(env)
        self.observation_space = self.env.state_space
        self._observable = True

    def observation(self, observation):
        return self.env.state

class PartiallyObservablePOMDP(gym.Wrapper):
    """Adds a dummy observation at beginning"""
    def __init__(self, env):
        assert isinstance(env.unwrapped, POMDP)
        super().__init__(env)
        nobs = self.env.observation_space.n
        self._start_obs = nobs+1
        self.observation_space = gym.spaces.Discrete(self.env.observation_space.n + 1)
        self._observable = False

    def reset(self):
        self.env.reset()
        return self._start_obs

class AutoresettingBatchPOMDP(gym.Wrapper):
    """Simulates multiple POMDP trajectories at the same time. Automatically reset completed trajectories. Must be applied to wrapped POMDP"""

    def __init__(self, env, batch_size, fomdp=False, time_limit=None):
        if not isinstance(env.unwrapped, POMDP):
            raise TypeError(f'Env is not a POMDP (got {type(env)}).')
        if batch_size <= 0:
            raise ValueError(f'Batch size is not positive (got ({batch_size}).')

        super().__init__(env)
        self.batch_size = batch_size
        self.state = np.full([batch_size], -1, dtype=int)
        if fomdp: self.observation_space = env.state_space; self._observable = True; self._start_obs = None
        else: self.observation_space = gym.spaces.Discrete(env.observation_space.n+1); self._observable = False; self._start_obs = env.observation_space.n
        self.max_time = time_limit or int(3000000)
        self.elapsed_time = np.zeros([batch_size], dtype=int)

    def reset(self):  # pylint: disable=arguments-differ
        self.state = self.reset_functional()
        self.elapsed_time[:] = 0
        return np.full([self.batch_size], self._start_obs, dtype=int) if not self._observable else self.state  # Hack to return some form of obs on reset

    def step(self, action):
        self.state, *ret = self.step_functional(self.state, action)
        if self._observable: ret[0] = self.state
        return ret

    def reset_functional(self, bsize=None):
        bsize = bsize if bsize is not None else self.batch_size
        if self.env.start is None:
            state = self.np_random.randint(
                self.state_space.n, size=bsize
            )
        else:
            state = self.np_random.multinomial(
                1, self.env.start, size=bsize
            ).argmax(1)
        return state

    def step_functional(self, state, action):
        if ((state == -1) != (action == -1)).any():
            raise ValueError(f'Invalid state-action pair ({state}, {action}).')

        shape = state.shape  # Shape of state (batch size)
        # Assume current state is all valid (because autoreset)
        s = np.array([
            self.np_random.multinomial(1,p).argmax() for p in self.env.T[state, action]
        ])
        o = np.array([
            self.np_random.multinomial(1, p).argmax() for p in self.env.O[state, action, s]
        ])
        r = self.env.R[state, action, s, o]
        self.elapsed_time += 1
        d = self.elapsed_time >= self.max_time
        if self.env.episodic:
            d = np.logical_or(self.env.D[state, action], d)  # Evaluated dones for new state
        s[d] = self.reset_functional(d.sum())  # Reset state
        self.elapsed_time[d] = 0  # Reset time
        if not self._observable: o[d] = self._start_obs  # Reset obs to start
        reward_cat = [self.rewards_dict[r_] for r_ in r]
        info = dict(reward_cat=reward_cat)
        return s, o, r, d, info