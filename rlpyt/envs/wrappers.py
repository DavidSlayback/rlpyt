import gym
from gym import ObservationWrapper, ActionWrapper, Wrapper, RewardWrapper, spaces
import json
import os
from os.path import join
import numpy as np
import cv2

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main_observation_space(env):
    if hasattr(env.observation_space, 'spaces'):
        return env.observation_space.spaces['obs']
    else:
        return env.observation_space


def has_image_observations(observation_space):
    """It's a heuristic."""
    return len(observation_space.shape) >= 2



# Base
class RecordingWrapper(Wrapper):
    """
    Generalized recording wrapper for environments that render
    """
    def __init__(self, env, record_to, player_id=0):
        super().__init__(env)

        self._record_to = record_to
        self._episode_recording_dir = None
        self._record_id = 0
        self._frame_id = 0
        self._player_id = player_id
        self._recorded_episode_reward = 0
        self._recorded_episode_shaping_reward = 0

        self._recorded_actions = []

        # Experimental! Recording Doom replay. Does not work in all scenarios, e.g. when there are in-game bots.
        self.unwrapped.record_to = record_to

    def reset(self):
        if self._episode_recording_dir is not None and self._record_id > 0:
            # save actions to text file
            with open(join(self._episode_recording_dir, 'actions.json'), 'w') as actions_file:
                json.dump(self._recorded_actions, actions_file)

            # rename previous episode dir
            reward = self._recorded_episode_reward + self._recorded_episode_shaping_reward
            new_dir_name = self._episode_recording_dir + f'_r{reward:.2f}'
            os.rename(self._episode_recording_dir, new_dir_name)

        dir_name = f'ep_{self._record_id:03d}_p{self._player_id}'
        self._episode_recording_dir = join(self._record_to, dir_name)
        ensure_dir_exists(self._episode_recording_dir)

        self._record_id += 1
        self._frame_id = 0
        self._recorded_episode_reward = 0
        self._recorded_episode_shaping_reward = 0

        self._recorded_actions = []

        return self.env.reset()

    def _record(self, img):
        frame_name = f'{self._frame_id:05d}.png'
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(join(self._episode_recording_dir, frame_name), img)
        self._frame_id += 1

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if isinstance(action, np.ndarray):
            self._recorded_actions.append(action.tolist())
        elif isinstance(action, np.int64):
            self._recorded_actions.append(int(action))
        else:
            self._recorded_actions.append(action)

        self._record(observation)
        self._recorded_episode_reward += reward
        if hasattr(self.env.unwrapped, '_total_shaping_reward'):
            # noinspection PyProtectedMember
            self._recorded_episode_shaping_reward = self.env.unwrapped._total_shaping_reward

        return observation, reward, done, info

class PixelFormatChwWrapper(ObservationWrapper):
    """TODO? This can be optimized for VizDoom, can we query CHW directly from VizDoom?"""

    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            img_obs_space = env.observation_space['obs']
            self.dict_obs_space = True
        else:
            img_obs_space = env.observation_space
            self.dict_obs_space = False

        if not has_image_observations(img_obs_space):
            raise Exception('Pixel format wrapper only works with image-based envs')

        obs_shape = img_obs_space.shape
        max_num_img_channels = 4

        if len(obs_shape) <= 2:
            raise Exception('Env obs do not have channel dimension?')

        if obs_shape[0] <= max_num_img_channels:
            raise Exception('Env obs already in CHW format?')

        h, w, c = obs_shape
        low, high = img_obs_space.low.flat[0], img_obs_space.high.flat[0]
        new_shape = [c, h, w]

        if self.dict_obs_space:
            dtype = env.observation_space.spaces['obs'].dtype if env.observation_space.spaces['obs'].dtype is not None else np.float32
        else:
            dtype = env.observation_space.dtype if env.observation_space.dtype is not None else np.float32

        new_img_obs_space = spaces.Box(low, high, shape=new_shape, dtype=dtype)

        if self.dict_obs_space:
            self.observation_space = env.observation_space
            self.observation_space.spaces['obs'] = new_img_obs_space
        else:
            self.observation_space = new_img_obs_space

        self.action_space = env.action_space

    @staticmethod
    def _transpose(obs):
        return np.transpose(obs, (2, 0, 1))  # HWC to CHW for PyTorch

    def observation(self, observation):
        if observation is None:
            return observation

        if self.dict_obs_space:
            observation['obs'] = self._transpose(observation['obs'])
        else:
            observation = self._transpose(observation)
        return observation

# Continuous Control
class ClipActionsWrapper(ActionWrapper):
    """
    Wrapper to clip actions within the range of the environment
    """
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

RLPYT_CONTROL_WRAPPERS = [ClipActionsWrapper]
RLPYT_RECORDING_CONTROL_WRAPPERS = [RecordingWrapper] + RLPYT_CONTROL_WRAPPERS
# Gym-Minigrid (from sample-factory)
class MinigridRecordingWrapper(RecordingWrapper):
    """
    Specific recording wrapper for minigrid
    """
    def __init__(self, env, record_to):
        super().__init__(env, record_to)

    # noinspection PyMethodOverriding
    def render(self, mode, **kwargs):
        self.env.render()
        frame = self.env.render('rgb_array')
        self._record(frame)
        return frame

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._recorded_episode_reward += reward
        return observation, reward, done, info

class RenameImageObsWrapper(gym.ObservationWrapper):
    """We call the main observation just 'obs' in all algorithms."""

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = env.observation_space
        self.observation_space.spaces['obs'] = self.observation_space.spaces['image']
        self.observation_space.spaces.pop('image')

    def observation(self, observation):
        observation['obs'] = observation.pop('image')
        return observation

from gym_minigrid.wrappers import ImgObsWrapper
RLPYT_MINIGRID_WRAPPERS = [ImgObsWrapper]
RLPYT_RECORDING_MINIGRID_WRAPPERS = [MinigridRecordingWrapper] + RLPYT_MINIGRID_WRAPPERS

# Miniworld
class TransposeImageWrapper(gym.ObservationWrapper):
    """ Transpose image from [H,W,C] to [C,H,W]"""
    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 1, 0)

RLPYT_MINIWORLD_WRAPPERS = [TransposeImageWrapper]

RLPYT_WRAPPER_KEY = "RLPYT_Extra_Wrappers"