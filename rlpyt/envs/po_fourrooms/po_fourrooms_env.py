import gym
import gym_classics
from gym_classics.envs.four_rooms import FourRooms, NoisyGridworld
from rlpyt.envs.gym import GymEnvWrapper, GymSpaceWrapper
import numpy as np

all_squares = [(i,j) for i in range(11) for j in range(11)]
oc_initial_goal = [(8,4)]  # Goal in east doorway
# Base FourRooms from gym-classics starts in (0,0) i.e. bottom-left, goal is (10,10) i.e., top-right. Probably want a stochastic version
class StochasticFourRooms(FourRooms):
    """ Classic FourRooms, but with parameterizable start and end

    Args:
        possible_starts: Set of possible (x,y) start locations, or None for default. Drawn from each episode
        possible_goals: Set of possible (x,y) end locations, or None for default. Drawn at initialization only
    """
    def __init__(self, possible_starts=None, possible_goals=None, goal_seed=None):
        blocks = frozenset({(5,0), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,9), (5,10),
                            (0,5), (2,5), (3,5), (4,5),
                            (6,4), (7,4), (9,4), (10,4)})
        if possible_starts is None: possible_starts = {(0,0)}
        possible_starts = possible_starts - blocks
        goal = (10,10) # Default
        if possible_goals is not None:
            possible_goals = list(possible_goals - blocks)
            np.random.seed(goal_seed)
            goal = possible_goals[np.random.randint(0, len(possible_goals))]
            possible_starts = possible_starts - {goal}
        self._goal = goal
        super(NoisyGridworld, self).__init__(dims=(11, 11), starts=possible_starts, blocks=blocks)

    def transfer(self, goal=None):
        """ Transfer original goal to a different one. Return goal (in case it doesn't change due to walls)"""
        if goal is not None:
            if isinstance(goal, (int, float)):
                goal = self._decode(int(goal))
            self._goal = goal if goal not in self._blocks else self._goal
        return self._encode(self._goal)

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

def make_po_fourrooms(fomdp=False, max_episode_steps=2000):
    """ max_episode_steps is a possible kwargs"""
    TL = gym.wrappers.TimeLimit
    e = StochasticFourRooms(possible_starts=set(all_squares), possible_goals=set(oc_initial_goal))
    if not fomdp: e = FourRoomsFourWallsWrapper(e)
    e = TL(e, max_episode_steps)
    return GymEnvWrapper(e)

if __name__ == "__main__":
    e = StochasticFourRooms({(0,0), (1,1), (2,2)}, possible_goals={(10,10), (9,9)})
    t = e.transfer()
    t2 = e.transfer(64)
    t3 = e.transfer((9,9))
    e = FourRoomsFourWallsWrapper(e)
    s = e.reset()
    s_d = list(e.env._state)
    print("Initial_State_{}".format(s_d))
    for t in range(200):
        o, r, d, info = e.step(e.action_space.sample())
        print("Observation_{}".format(o))