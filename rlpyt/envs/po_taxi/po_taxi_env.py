import gym
from rlpyt.envs.gym import GymEnvWrapper, GymSpaceWrapper

# Partially-observable taxi environment
# Default state space: (taxi_row, taxi_col, passenger_location (index of 5), destination (index of 4)) converted to single int (500 total states)
# Passenger can be in 5 locs (4 locs or taxi). Destination can be in 4 locs (but never the same one as passenger at start)
# Revised state space: Remove passenger after the first step of episode. State space remains the same (to account for first obs), but is aliased thereafter
class TaxiPartialObservabilityWrapper(gym.Wrapper):
    def step(self, action):
        s, r, d, info = self.env.step(action)
        o = list(self.env.decode(s))
        return self.env.encode(o[0], o[1], 4, o[3]), r, d, info  # Replace passenger component with "in taxi" index (which is never observed on reset)

def make_po_taxi(**kwargs):
    """ max_episode_steps is a possible kwargs"""
    e = TaxiPartialObservabilityWrapper(gym.make('Taxi-v3', **kwargs))
    return GymEnvWrapper(e)

if __name__ == "__main__":
    test1 = TaxiPartialObservabilityWrapper(gym.make('Taxi-v3'))
    s = test1.reset()
    s_d = list(test1.decode(s))
    print("Initial_State_{}".format(s_d))
    o, r, d, info = test1.step(test1.action_space.sample())
    o_d = list(test1.decode(o))
    for t in range(200):
        o, r, d, info = test1.step(test1.action_space.sample())
        print("Observation_{}".format(list(test1.decode(o))))
        test1.render()