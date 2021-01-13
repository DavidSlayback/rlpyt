import numpy as np
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo, namedtuple
from procgen.env import ENV_NAMES, EXPLORATION_LEVEL_SEEDS, DISTRIBUTION_MODE_DICT, ProcgenGym3Env
import cv2

VALID_EXTREME_ENVS = ["chaser", "dodgeball", "leaper", "starpilot"]
VALID_MEMORY_ENVS = ["caveflyer", "dodgeball", "heist", "jumper", "maze", "miner"]

EnvInfo = namedtuple("EnvInfo", ["timeout"])
class ProcgenEnv(Env):
    """Interface for OpenAI gym_procgen environments

    The action space is an `IntBox` for the number of actions.  The observation
    space is an `IntBox` with ``dtype=uint8`` to save memory; conversion to float
    should happen inside the agent's model's ``forward()`` method.

    Args:
        game (str): game name
        num_levels (int): Number of unique levels, 0 means infinite
        start_level (int): Lowest seed for generating levels. This + num_levels defines full set of possible levels
        paint_vel_info (bool): If true, puts velocity information in top right corner (for some games)
        use_generated_assets (bool): If true, use randomly generated assets instead of human (harder)
        center_agent (bool): If true, observation is centered on agent. If false, full level
        use_sequential_levels (bool): If false, reaching end of level ends episode and selects new level. If true,
            reaching end of level does not end episode, seed for new level is derived from current. If combined with
            start_level=<seed> and num_levels=1, results in a single linear series of levels (a la Atari/retro)
        distribution_mode (str): ["easy", "hard", ("extreme", "memory", "exploration")]. Easy and hard difficulities are
            for all environments. "extreme" is harder, "memory" uses larger world, smaller view, "exploration" makes use
            of specific seeds. See above for valid combinations of game/difficulty
        use_backgrounds (bool): If true, use human asset backgrounds, else black
        restrict_themes (bool): If true, levels with multiple theme options only use one.
        use_monochrome_assets (bool): If true, use monochromatic rectangles instead of human assets. Pair with
            restrict_themes=True
        num_threads (int): Number of threads to use. Set to 0 to make compatible with rlpyt
        num_envs (int): Number of environments in this wrapper. Set to 1 to make compatible with rlpyt
        horizon (int): If not 0, maximum time for environment
        num_img_obs (int): Number of observations to return each step (i.e., framestack). Defaults to 1
    """
    def __init__(self, game,
                 num_levels=500,
                 start_level=0,
                 paint_vel_info=False,
                 use_generated_assets=False,
                 center_agent=True,
                 use_sequential_levels=False,
                 distribution_mode="easy",
                 use_backgrounds=True,
                 restrict_themes=False,
                 use_monochrome_assets=False,
                 num_threads=0,
                 num_envs=1,
                 horizon=0,
                 num_img_obs=1,
                 ):
        # Check game and difficulty pairings are compatible
        assert game in ENV_NAMES
        if distribution_mode == "extreme": assert game in VALID_EXTREME_ENVS
        if distribution_mode == "memory": assert game in VALID_MEMORY_ENVS
        if distribution_mode == "exploration": assert game in list(EXPLORATION_LEVEL_SEEDS.keys())
        if use_monochrome_assets: assert restrict_themes  # Ensure monochrome only with restrict_themes
        # Wrap gym3 environment in ToGymEnv
        self.env = ProcgenGym3Env(num=num_envs, env_name=game, use_backgrounds=use_backgrounds,
                                  use_monochrome_assets=use_monochrome_assets, restrict_themes=restrict_themes,
                                  use_generated_assets=use_generated_assets, paint_vel_info=paint_vel_info,
                                  distribution_mode=distribution_mode, num_levels=num_levels, start_level=start_level,
                                  num_threads=num_threads, center_agent=center_agent,
                                  use_sequential_levels=use_sequential_levels)
        self._game = game
        _r, self._o, _f = self.env.observe()
        _, h, w, c = self._o['rgb'].shape
        self._c = c  # Keep track of number of channels
        self._frame_stack = num_img_obs
        obs_shape = (c*num_img_obs, h, w)
        self._observation_space = IntBox(0, 255, shape=obs_shape, dtype="uint8")  # Will be transposed
        self._action_space = IntBox(0, self.env.ac_space.eltype.n)
        self._time_limit = horizon
        self._time = 0
        self._obs = np.zeros(shape=obs_shape, dtype="uint8")
        self._obs_index = 0
        self._action = np.zeros(num_envs, dtype=int)  # Action input must be numpy array
        self._update_obs()


    def reset(self):
        """
        Environment does not reset unless done. Return current observation
        """
        return self._obs

    def _reset_obs(self):
        """
        Reset stacked obs
        """
        self._obs[:] = 0

    def _update_obs(self):
        """
        Add newest frame to stacked obs. Update start index as needed
        """
        if self._frame_stack == 1: self._update_obs_single()
        self._obs[self._obs_index:self._obs_index+self._c] = self._o['rgb'].transpose((0, 3, 1, 2))
        self._obs_index += self._c
        self._obs_index %= self._obs.shape[0]

    def _update_obs_single(self):
        """
        Optimization for no framestack case
        """
        self._obs[:] = self._o['rgb'].transpose((0, 3, 1, 2))

    def _get_obs(self):
        """
        Get the framestacked observation in proper order
        """
        post_idx = np.arange(self._obs_index, self._obs.shape[0])  # From obs_index to end
        pre_idx = np.arange(0, self._obs_index)  # From 0 to obs_index
        return self._obs[np.concatenate((post_idx, pre_idx))]


    def step(self, action):
        """
        Take step with action, then observe result
        """
        self._time += 1  # Update time step
        timeout=False
        self._action[:] = action
        if self._time_limit and self._time >= self._time_limit:
            self._action[:] = -1  # Can force a reset by taking action of -1
            self._time = 0  # Reset time
            timeout = True
        self.env.act(self._action)
        r, self._o, d = self.env.observe()
        r, d = r.squeeze(), d.squeeze()
        if d: self._reset_obs()  # If done, reset the stacked frames
        self._update_obs()  # Add newest observation to stacked frames
        o = self._get_obs()  # Get stacked observation in correct order
        return EnvStep(o, r, d, EnvInfo(timeout=timeout))

    def render(self, mode='rgb_array'):
        """
        Render Procgen environment
        """
        img = self._get_obs()  # Get current stacked obs
        img = img[-self._c:]  # Get last frame
        if mode == "human":
            cv2.imshow(self._game, img.transpose((1,2,0)))
            cv2.waitKey(10)
        return img

    def seed(self, seed=None):
        pass

    def close(self):
        pass






if __name__ == "__main__":
    test = ProcgenEnv('coinrun')
    test.step(0)
    pass