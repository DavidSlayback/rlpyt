import gym
import gym_classics
from gym_classics.envs.four_rooms import FourRooms, NoisyGridworld
from rlpyt.envs.gym import GymEnvWrapper, GymSpaceWrapper
import numpy as np

# Base FourRooms from gym-classics starts in (0,0) i.e. bottom-left, goal is (10,10) i.e., top-right. Probably want a stochastic version
class StochasticFourRooms(FourRooms):
    """ Classic FourRooms, but with parameterizable start and end

    Args:
        possible_starts: Set of possible (x,y) start locations, or None for default. Drawn from each episode
        possible_goals: Set of possible (x,y) end locations, or None for default. Drawn at initialization only
    """
    def __init__(self, possible_starts=None, possible_goals=None):
        blocks = frozenset({(5,0), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,9), (5,10),
                            (0,5), (2,5), (3,5), (4,5),
                            (6,4), (7,4), (9,4), (10,4)})
        if possible_starts is None: possible_starts = {(0,0)}
        possible_starts = possible_starts - blocks
        self._goal = (10, 10)
        super(NoisyGridworld, self).__init__(dims=(11, 11), starts=possible_starts, blocks=blocks)
        if possible_goals is not None:
            possible_goals = list(possible_goals - blocks)
            self._goal = possible_goals[self.np_random.randint(0, len(possible_goals))]

    def transfer(self, goal=None):
        """ Transfer original goal to a different one. Return goal (in case it doesn't change due to walls)"""
        if goal is not None:
            self._goal = goal if goal not in self._blocks else self._goal
        return self._goal

# Partially-observable four rooms
# Default state space: n=104 discrete state space
# Revised state space: One paper proposes being able to observe walls around you. Chris says maybe direction of goal?
# First stab: Merely the existence of 4 surrounding walls
class FourRoomsFourWallsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n_spaces_wall = (2 ** 4)  # Number of states for walls
        self.observation_space = gym.spaces.Discrete(n_spaces_wall)
        self._dir = np.array([[0,1], [1,0], [0,-1], [-1,0]])  # Directions

    def observation(self, observation):
        o = 0  # Base observation, no adjacent walls
        dims = np.array(self.env._dims)
        obs_decode = np.array(self.env._decode(observation))  # Convert to (x,y)
        obs_adj = self._dir + obs_decode  # Coordinates of adjacent squares (can assume these are all valid grid coordinates)
        blocks = self.env._blocks # Coordinates of blockers
        for i, arr in enumerate(obs_adj):
            o += (i * 2) * (np.any((arr < 0) | (arr >= dims)) or (tuple(arr) in blocks))
        return o

def make_po_fourrooms(max_episode_steps=2000):
    """ max_episode_steps is a possible kwargs"""
    TL = gym.wrappers.TimeLimit
    e = TL(FourRoomsFourWallsWrapper(gym.make('FourRooms-v0')), max_episode_steps)
    return GymEnvWrapper(e)

if __name__ == "__main__":
    e = StochasticFourRooms({(0,0), (1,1), (2,2)}, possible_goals={(10,10), (9,9)})
    e = FourRoomsFourWallsWrapper(e)
    s = e.reset()
    s_d = list(e.env._state)
    print("Initial_State_{}".format(s_d))
    for t in range(200):
        o, r, d, info = e.step(e.action_space.sample())
        print("Observation_{}".format(o))