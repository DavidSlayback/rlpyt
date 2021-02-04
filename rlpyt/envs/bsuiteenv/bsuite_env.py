import bsuite
from dm_env import specs
from bsuite.utils.wrappers import RewardNoise, RewardScale
from rlpyt.envs.base import Env, EnvStep, EnvInfo
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.utils.logging.context import LOG_DIR, osp  # Base log directory

VALID_ENV_SWEEP_IDS = bsuite.sweep.SWEEP
VALID_ENV_IDS = list(bsuite.bsuite.EXPERIMENT_NAME_TO_ENVIRONMENT.keys())
"""
Environments: Bandit, Cartpole, Cartpole Swingup, Catch, deep_sea, discounting_chain, memory_len, memory_size, mnist, mountain_car, umbrella. All 10k episodes
    - Bandit: 11 actions/arms, linspace of rewards 0-1 (determined by mapping_seed). Mapping seed applied before "seed" from wrappers
    - Cartpole: Basic cartpole (keep pole balanced), lots of parameters, but wouldn't touch anything except seed
    - CartpoleSwingup: As above, but pole begins downward, must be swung upward to see reward. Agent pays "move_cost" on action. Only touch seed.
    - Catch: Agent catches balls falling from in columnsxrows (10x5) grid. Obs are binary grid (1 for ball, paddle; 0 otherwise). rows, columns, seed. Episode ends on missing or catching ball
    - DeepSea: sizexsize grid, deterministic=False for 1/N chance of 'right' action failure. At each state, one action is 'left', one 'right' (randomized across states by mapping_seed). Action always results in agent dropping one row. Far right corner is reward
        obs is size*size grid, but with 1 where agent is. Rewards are move_cost and taking right action at rightmost column. seed for nondeterministic
    - DiscountingChain: Observation is two pixels: (context, time_to_live). Context will only be -1 in the first step, then equal to the action selected in
        the first step. For all future decisions the agent is in a "chain" for that action. Reward of +1 come  at one of: 1, 3, 10, 30, 100
        However, depending on the seed, one of these chains has a 10% bonus. mapping_seed
    - MemoryChain: Obs is (context, ttl). Context is 0 except at first step ( +-1 bitstring). seed. memory_length for length of episode. num_bits for size of "context"
    - MNISTBandit: fraction is set of training set to keep. seed for index of image. +1 for correct label, -1 for wrong (10 labels). Terminates episode
    - MountainCar: max_steps (1000 default) and seed (for position). -1 per timestep
    - UmbrellaChain: obs is (need_umbrella, have_umbrella, ttl, n_distractor...). Only first action (pickup or not) matters. chain_length is length of episode, n_distractor is size of distractors, seed
    
Noise (bandit, cartpole, catch, mnist, mountain_car): Wrap environment with RewardNoise wrapper. noise_scale is stddev of gaussian on reward. Also takes "seed"
Scale (bandit, cartpole, catch, mnist, mountain_car): Wrap environment with RewardScale wrapper. reward_scale is factor by which reward is multiplied

Auto reset 
"""
class BSuiteEnv(Env):
    """ DeepMind bsuite environment in rlpyt-compatible format

    Args:
        id: Id of bsuite environment (str). Must be listed in bsuite.sweep.SWEEP
        exp_kwargs: Experiment kwargs for bsuite (dict or None). If None, id must be sweep id. Else, must be valud args for specified environment
        external_logging: Type of bsuite logging to use (str). bsuite has logging methods built into its environments,
            either 'sqlite' or 'csv'. Defaults to 'none' (use rlpyt logging). Only applies to SWEEP IDs
        save_path: Path to save data too (on top of base rlpyt data directory) (str).
    """
    def __init__(self,
                 id: str,
                 exp_kwargs: dict = None,
                 external_logging: str = 'none',
                 save_path: str = '',):
        assert (id in VALID_ENV_SWEEP_IDS) or (id in VALID_ENV_IDS and exp_kwargs is not None)  # Either using one of presets or using base experiment with other settings
        aug_path = osp.join(LOG_DIR, save_path)  # LOG_DIR + save_path
        if id in VALID_ENV_SWEEP_IDS: # Pre-parameterized experiments
            if external_logging == 'none':
                env = bsuite.load_from_id(id)  # No recording
            else:
                env = bsuite.load_and_record(id, aug_path, external_logging)  # Record in sql or csv. same sql for each id
            self.num_episodes = env.bsuite_num_episodes
        else:
            noise_scale = exp_kwargs.pop('noise_scale', 0.)
            noise_scale_seed = exp_kwargs.pop('noise_scale_seed', 0.)
            reward_scale = exp_kwargs.pop('reward_scale', 0.)
            env = bsuite.load(id, **exp_kwargs)
            if noise_scale: env = RewardNoise(env, noise_scale, noise_scale_seed)
            if reward_scale: env = RewardScale(env, reward_scale)
            self.num_episodes = 1e4  # Default
        self.env = env
        self._action_space = IntBox(low=0, high=self.env.action_spec().num_values)
        o_spec = self.env.observation_spec()
        if isinstance(o_spec, specs.BoundedArray):
            self._observation_space = FloatBox(low=o_spec.minimum.item(),
                                               high=o_spec.maximum.item(),
                                               shape=o_spec.shape,
                                               dtype=o_spec.dtype)
        else:
            self._observation_space = FloatBox(low=-float('inf'),
                                               high=float('inf'),
                                               shape=o_spec.shape,
                                               dtype=o_spec.dtype)
        self._last_observation = None
        self.game_over = False,
        self.viewer = None

    def reset(self):
        self.game_over = False
        timestep = self.env.reset()
        self._last_observation = timestep.observation
        return timestep.observation

    def step(self, action):
        timestep = self.env.step(action)
        self._last_observation = timestep.observation
        reward = timestep.reward or 0.
        if timestep.last():
            self.game_over = True
        return EnvStep(timestep.observation, reward, timestep.last(), EnvInfo())

    def render(self, mode: str = 'rgb_array'):
        if self._last_observation is None:
            raise ValueError('Environment not ready to render. Call reset() first.')

        if mode == 'rgb_array':
            return self._last_observation

        if mode == 'human':
            if self.viewer is None:
                # pylint: disable=import-outside-toplevel
                # pylint: disable=g-import-not-at-top
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self._last_observation)
            return self.viewer.isopen

if __name__ == "__main__":
    for env in VALID_ENV_SWEEP_IDS:
        test = BSuiteEnv(env)
        testo = test.reset()
        print(test.observation_space.shape)
    test = BSuiteEnv('bandit/0')
    testo = test.reset()
    testa = test.step(test.action_space.sample())