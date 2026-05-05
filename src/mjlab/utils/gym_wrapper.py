"""Gymnasium-compliant wrapper for ManagerBasedRlEnv."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.spaces import Box as MjBox, Dict as MjDict, Space as MjSpace


def _to_gym_space(space: MjSpace) -> gym.Space:
  """Convert mjlab's lightweight Space definition to a Gymnasium space."""

  dtype_map = {
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
  }

  if isinstance(space, MjDict):
    return gym.spaces.Dict({k: _to_gym_space(v) for k, v in space.spaces.items()})
  if isinstance(space, MjBox):
    dtype = dtype_map[space.dtype]
    low = np.array(space.low, dtype=dtype)
    high = np.array(space.high, dtype=dtype)
    # Broadcast scalars to the full shape
    low = np.broadcast_to(low, space.shape)
    high = np.broadcast_to(high, space.shape)
    return gym.spaces.Box(low=low, high=high, dtype=dtype)
  raise TypeError(f"Unsupported space type: {type(space)}")


class MjlabGymVecEnv(gym.Env):
  """Wrapper that exposes a ManagerBasedRlEnv through Gymnasium semantics.

  This wrapper adapts mjlab's lightweight spaces to Gymnasium spaces so that
  third-party libraries (Gymnasium, Stable-Baselines3, CleanRL, etc.) can
  consume the environment as if it were a native Gymnasium vector env.
  """

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.num_envs = env.num_envs
    self.device = env.device
    
    # Metadata
    self.metadata = env.metadata
    self.render_mode = env.render_mode
    
    # Spaces
    self.single_action_space = _to_gym_space(env.single_action_space)
    self.single_observation_space = _to_gym_space(env.single_observation_space)
    self.action_space = _to_gym_space(env.action_space)
    self.observation_space = _to_gym_space(env.observation_space)
    print(self.action_space, self.observation_space ,self.single_action_space, self.single_observation_space)
  def reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Reset the environment."""
    obs, extras = self.env.reset(seed=seed, options=options)
    return obs, extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
  ]:
    """Execute a step in the environment.
    
    Returns:
        obs: Dictionary of observation tensors.
        reward: Reward tensor of shape (num_envs,).
        terminated: Boolean tensor of shape (num_envs,) indicating terminal states.
        truncated: Boolean tensor of shape (num_envs,) indicating time-limits.
        info: Dictionary containing extra information.
    """
    obs, reward, terminated, truncated, extras = self.env.step(actions)
    
    # Gymnasium expects 'info' to contain final observation info for auto-reset envs.
    # ManagerBasedRlEnv already handles auto-reset internally and returns the 
    # observation of the NEW step (post-reset).
    
    # Store individual reward terms in extras for inspection
    extras["reward_terms"] = {}
    if hasattr(self.env, "reward_manager"):
        for i, name in enumerate(self.env.reward_manager._term_names):
             # The stored _step_reward is (num_envs, num_terms)
             # We can expose it here
             extras["reward_terms"][name] = self.env.reward_manager._step_reward[:, i]
    return obs, reward, terminated, truncated, extras

  def close(self):
    return self.env.close()

  def render(self):
    return self.env.render()

  @property
  def unwrapped(self):
    return self.env.unwrapped
