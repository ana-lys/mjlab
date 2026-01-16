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



def joint_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint velocities on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def collision_penalty(
  env: ManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
  """Penalize collisions based on contact presence."""
  sensor: ContactSensorFFW = env.scene[sensor_name]
  # Return whether collision is detected
  return sensor.data.collision_detected.squeeze(-1)


def ee_tracking_l2_bimanual(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize summed L2 distance of both end-effectors (sites) from their targets."""
    # Try using cached error from observations (fastest)
    # Cache is [N, S, 3] vector (target - ee)
    pos_error = getattr(env, "ffw_cached_pos_error_w", None)
    
    if pos_error is None:
        asset: Entity = env.scene[asset_cfg.name]
        ee_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
        target_pos = getattr(env, "target_ee_pos")
        pos_error = target_pos - ee_pos

    # Sum of squared distances over both dimensions and both hands
    l2_sq = torch.sum(torch.square(pos_error), dim=(1, 2))
    # Convert to strictly positive reward: 1.0 (perfect) -> 0.0 (far), exp(-d^2)
    return torch.exp(-l2_sq)

def ee_tracking_tanh_bimanual(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tanh-based dense reward for each hand being close to its target, summed."""
    # Try using cached error from observations (fastest)
    pos_error = getattr(env, "ffw_cached_pos_error_w", None)
    
    if pos_error is None:
        asset: Entity = env.scene[asset_cfg.name]
        ee_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
        target_pos = getattr(env, "target_ee_pos")
        pos_error = target_pos - ee_pos
        
    # Per-hand Euclidean distance [B, 2]
    # pos_error is (target - ee), norm is distance
    dist_per_hand = torch.norm(pos_error, dim=-1)
    
    # Tanh reward per hand: 1 - tanh(dist / std)
    reward_per_hand = 1.0 - torch.tanh(dist_per_hand / std)
    
    return torch.sum(reward_per_hand, dim=-1)


def ee_tracking_quat_bimanual(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize orientation error of both end-effectors."""
    # Try using cached error from observations (fastest)
    # Cache is [N, S, 3] axis-angle vector representing rotation difference
    rot_error = getattr(env, "ffw_cached_rot_error_w", None)
    
    if rot_error is not None:
        # Magnitude of axis-angle vector is the angle (radians)
        error_rad = torch.norm(rot_error, dim=-1)
    else:
        from mjlab.utils.lab_api.math import quat_error_magnitude
        asset: Entity = env.scene[asset_cfg.name]
        ee_quat = asset.data.site_quat_w[:, asset_cfg.site_ids, :]
        target_quat = getattr(env, "target_ee_quat")
        error_rad = quat_error_magnitude(ee_quat, target_quat)
    
    return torch.sum(error_rad, dim=-1)