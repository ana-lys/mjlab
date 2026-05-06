from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorFFW

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.ffw.mdp.reset import StatefulReset

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
  """Observation of target end-effector position.

  Also promotes queued targets once after reset to avoid using colliding poses.
  """
  sr: StatefulReset = getattr(env, "ffw_state")
  return sr.target_ee_pos.reshape(env.num_envs, -1)

def env_ready(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Observation of environment readiness.

  Returns a boolean tensor indicating which environments are ready.
  """
  sr: StatefulReset = getattr(env, "ffw_state")
  return sr.env_ready.reshape(env.num_envs, -1)

def collision_occupancy(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Get a binary tensor indicating whether a collision is detected."""
  sensor: ContactSensorFFW = env.scene[sensor_name]
  return sensor.data.occupancy


def top_collision_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensorFFW = env.scene[sensor_name]
  return sensor.data.top_force.view(sensor.data.force.shape[0], -1)

def top_collision_pos(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensorFFW = env.scene[sensor_name]
  return sensor.data.top_pos.view(sensor.data.force.shape[0], -1)
  

def ee_pos_error(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Observation of Euclidean position error between end-effector and target.

  Returns:
      [N, num_sites] tensor containing per-site Euclidean distance in world frame.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Current and target positions (world frame)
  ee_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]   # [N, S, 3]
  sr = getattr(env, "ffw_state")
  target_pos_w = sr.target_ee_pos                                # [N, S, 3]

  # Position error vector and per-site Euclidean norm
  pos_error_w = target_pos_w - ee_pos_w                        # [N, S, 3]
  dist_per_site = torch.norm(pos_error_w, dim=-1)              # [N, S]

  # Cache for rewards to reuse without recomputing
  sr.cached_pos_error_w = pos_error_w

  return dist_per_site.reshape(env.num_envs, -1)

