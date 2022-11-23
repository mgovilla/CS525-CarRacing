"""original code:
https://github.com/ray-project/ray/blob/master/rllib/env/wrappers/atari_wrappers.py
The code is modified to make it work on gym version 0.25.0 (WPI Fall2022: DS595 Reinforcement Learning)
"""
import gym
import numpy as np
from gym import spaces
from collections import deque
import cv2


def is_atari(env):
    if (
        hasattr(env.observation_space, "shape")
        and env.observation_space.shape is not None
        and len(env.observation_space.shape) <= 2
    ):
        return False
    return hasattr(env, "unwrapped") and hasattr(env.unwrapped, "ale")


def get_wrapper_by_cls(env, cls):
    """Returns the gym env wrapper of the given class, or None."""
    currentenv = env
    while True:
        if isinstance(currentenv, cls):
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            return None


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class EarlyStopEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shp, dtype=env.observation_space.dtype
        )
        self.noop_max = noop_max
        self.noop_frames = 0

    def reset(self):
        ob = self.env.reset()
        self.noop_frames = 0

        return ob

    def step(self, action):
        ob, reward, done, truncated, info = self.env.step(action)
        if self.noop_frames >= self.noop_max:
            self.noop_frames = 0
            ob = env.reset()
        else:
            self.noop_frames += 1

        return ob, reward, done, truncated, info

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


def wrap_deepmind(env, dim=84, clip_rewards=True, framestack=True, scale=False):
    """Configure environment for DeepMind-style Atari.
    Note that we assume reward clipping is done outside the wrapper.
    Args:
        env: The env object to wrap.
        dim: Dimension to resize observations to (dim x dim).
        framestack: Whether to framestack observations.
    """
    env = EarlyStopEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    if scale is True:
        env = ScaledFloatFrame(env)  # TODO: use for dqn?
    if clip_rewards is True:
        env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    # 4x image framestacking.
    if framestack is True:
        env = FrameStack(env, 4)
    return env


def make_wrap_box2d(env_id='CarRacing-v2', clip_rewards=True, continuous=False):
    env = gym.make(env_id, new_step_api=True, continuous=continuous)
    return wrap_deepmind(env, dim=96, clip_rewards=clip_rewards, framestack=True, scale=False)