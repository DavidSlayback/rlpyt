import numpy as np
from rlpyt.envs.base import Env, EnvStep
from rlpyt.envs.gym import info_to_nt
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo, namedtuple

EnvInfo = namedtuple("EnvInfo", ["timeout"])
class MiniWorldEnv(Env):
    """Interface for Gym-MiniWorld environments

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
    def __init__(self):
        pass