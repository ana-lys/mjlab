from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def ee_pos(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Observation of end-effector position using specific sites."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, :].reshape(env.num_envs, -1)


def ee_quat(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Observation of end-effector orientation (quaternion) using specific sites."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_quat_w[:, asset_cfg.site_ids, :].reshape(env.num_envs, -1)


def target_ee_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Observation of target end-effector position."""
    target = getattr(env, "target_ee_pos")
    return target.reshape(env.num_envs, -1)

def target_ee_quat(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Observation of target end-effector quat."""
    target = getattr(env, "target_ee_quat")
    return target.reshape(env.num_envs, -1)


def ee_pose_error(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Observation of error between end-effector and target (pos and orientation).

  Returns:
      [N, 6 * num_sites] tensor containing (pos_error_xyz, axis_angle_error_xyz) for each site,
      expressed in the robot's base frame.
  """
  from mjlab.utils.lab_api.math import quat_box_minus, quat_apply_inverse
  
  asset: Entity = env.scene[asset_cfg.name]
  
  # Current State (World Frame)
  ee_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]   # [N, S, 3]
  ee_quat_w = asset.data.site_quat_w[:, asset_cfg.site_ids, :] # [N, S, 4]
  
  # Target State (World Frame)
  target_pos_w = getattr(env, "target_ee_pos")   # [N, S, 3]
  target_quat_w = getattr(env, "target_ee_quat") # [N, S, 4]
  
  # Robot Base State (World Frame)
  root_quat_w = asset.data.root_link_quat_w # [N, 4]
  
  # 1. Position Error in World Frame
  pos_error_w = target_pos_w - ee_pos_w  # [N, S, 3]
  
  # 2. Orientation Error in World Frame (Axis-Angle)
  # Vector representing rotation needed to go from ee to target
  # Magnitude is angle, direction is axis (in World Frame)
  rot_error_w = quat_box_minus(target_quat_w, ee_quat_w) # [N, S, 3]

  # --- Cache for rewards ---
  # We cache the raw world-frame errors so rewards don't have to re-fetch/compute.
  # pos_error_w: vector FROM ee TO target.
  # rot_error_w: axis-angle vector representing rotation from ee to target.
  env.ffw_cached_pos_error_w = pos_error_w
  env.ffw_cached_rot_error_w = rot_error_w
  # -------------------------
  
  # 3. Transform errors into Base Frame
  # Expand root_quat to broadcast over sites: [N, 1, 4]
  root_quat_w_exp = root_quat_w.unsqueeze(1).expand(-1, pos_error_w.shape[1], -1)
  
  pos_error_b = quat_apply_inverse(root_quat_w_exp, pos_error_w)
  rot_error_b = quat_apply_inverse(root_quat_w_exp, rot_error_w)
  
  # Concatenate [N, S, 6] -> [N, S*6]
  return torch.cat([pos_error_b, rot_error_b], dim=-1).reshape(env.num_envs, -1)

