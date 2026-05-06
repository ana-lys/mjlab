from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.sensor.contact_ffw import ContactSensorFFW
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _active_mask(env: "ManagerBasedRlEnv") -> torch.Tensor:
  sr = getattr(env, "ffw_state", None)
  if sr is None:
    return torch.ones(env.num_envs, device=env.device)
  return sr.env_ready.float()



def joint_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint velocities on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  mask = _active_mask(env)
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1) * mask


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  mask = _active_mask(env)
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1) * mask


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  mask = _active_mask(env)
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  ) * mask


def collision_penalty(
  env: ManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
  """Penalize collisions based on contact presence."""
  sensor: ContactSensorFFW = env.scene[sensor_name]
  mask = _active_mask(env)
  return sensor.data.collision_detected.squeeze(-1) * mask


def _compute_pos_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor | None:
    """Compute [N, S, 3] position error (target - ee) from live sim data."""
    sr = getattr(env, "ffw_state", None)
    if sr is None:
        return None
    asset: Entity = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [N, S, 3]
    return sr.target_ee_pos - ee_pos_w                          # [N, S, 3]


def ee_tracking_l2_bimanual(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize summed L2 distance of both end-effectors (sites) from their targets."""
    pos_error = _compute_pos_error(env, asset_cfg)
    mask = _active_mask(env)
    if pos_error is None:
      return torch.zeros(env.num_envs, device=env.device) * mask

    # Sum of squared distances over both dimensions and both hands
    l2_sq = torch.sum(torch.square(pos_error), dim=(1, 2))
    # Convert to strictly positive reward: 1.0 (perfect) -> 0.0 (far), exp(-d^2)
    return torch.exp(-l2_sq) * mask

def ee_tracking_tanh_bimanual(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Tanh-based dense reward for each hand being close to its target, summed."""
    pos_error = _compute_pos_error(env, asset_cfg)
    mask = _active_mask(env)
    if pos_error is None:
      return torch.zeros(env.num_envs, device=env.device) * mask
        
    # Per-hand Euclidean distance [B, 2]
    # pos_error is (target - ee), norm is distance
    dist_per_hand = torch.norm(pos_error, dim=-1)
    
    # Tanh reward per hand: 1 - tanh(dist / std)
    reward_per_hand = 1.0 - torch.tanh(dist_per_hand / std)
    return torch.sum(reward_per_hand, dim=-1) * mask
