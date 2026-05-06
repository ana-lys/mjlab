from __future__ import annotations
import os
import numpy as np
import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs import mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from typing import TYPE_CHECKING

from mjlab.entity import Entity
from mjlab.scene.scene import SceneCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_gaussian,
  sample_log_uniform,
  sample_uniform,
)
import math
if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_OBSTACLE_POS_RANGES: dict[str, dict] = {
    "obs_table":   {"r": (01.65, 2.2), "phi": (-math.pi, math.pi), "z": (0.5, 1.5), "center": (0.0, 0.0)},
    "obs_pole":    {"r": (01.5, 2.0), "phi": (-math.pi, math.pi), "z": (0.5, 1.0), "center": (0.0, 0.0)},
    "obs_cube":    {"r": (01.5, 2.0), "phi": (-math.pi, math.pi), "z": (0.5, 1.0), "center": (0.0, 0.0)},
    "obs_capsule": {"r": (01.5, 2.0), "phi": (-math.pi, math.pi), "z": (0.5, 1.0), "center": (0.0, 0.0)},
}

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")
_DEFAULT_SCENE_CFG_prototype =SceneCfg(num_envs=1)

def reset_joints_uniformly(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: tuple[float, float] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  collision_force_threshold: float = 1.0,
) -> None: 
  """Reset joint positions uniformly within their soft limits.
  Retries up to `max_retries` times to find a collision-free configuration.

  Args:
    env: The environment.
    env_ids: Environment IDs to reset.
    velocity_range: Optional velocity range for randomizing joint velocities.
    asset_cfg: Asset configuration.
    max_retries: Maximum number of retries to find a valid configuration.
    collision_force_threshold: Net force threshold to consider a collision.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)

  # if(len(env_ids) != 0):
  #   print("Resetting joints for env_ids:", env_ids.cpu().numpy())
  asset: Entity = env.scene[asset_cfg.name]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  # We need to resolve joint indices once for efficiency.
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  # Keep track of which envs still need a valid pose.
  envs_to_reset = env_ids.clone()
  
  if len(envs_to_reset) == 0:
    return 
  # Get limits for the specified joints [num_envs_to_reset, num_joints, 2].
  joint_limits = soft_joint_pos_limits[envs_to_reset][:, joint_ids]  
  # Sample uniformly between lower (index 0) and upper (index 1) limits.
  joint_pos = sample_uniform(
    joint_limits[..., 0],
    joint_limits[..., 1],
    joint_limits.shape[:-1],  # (num_envs_to_reset, num_joints)
    env.device,
  )  
  # Handle velocities.
  if velocity_range is not None:
    joint_vel = asset.data.default_joint_vel[envs_to_reset][:, joint_ids].clone()
    joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)
  else:
    joint_vel = torch.zeros_like(joint_pos)  
  # Write state to sim.
  asset.write_joint_state_to_sim(
    joint_pos.view(len(envs_to_reset), -1),
    joint_vel.view(len(envs_to_reset), -1),
    env_ids=envs_to_reset,
    joint_ids=joint_ids,
  )
  
  return 



class StatefulReset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.cfg = cfg
        self.debug = False
        # Initialize memory
        self.target_ee_pos = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self.target_tobe_ee_pos = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self.target_next_ee_pos = torch.tensor(
          [[0.5, 0.3, 1.0], [0.5, -0.3, 1.0]],
          device=env.device,
        ).repeat(env.num_envs, 1, 1)
        self.env_ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.robot_need_reset = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.nj = 15  # Default to 15 joints if not specified; will be overridden on first reset.
        self.target_joint_pos = torch.zeros(env.num_envs, self.nj, device=env.device)
        self.target_tobe_joint_pos = torch.zeros(env.num_envs, self.nj, device=env.device)
        self.target_next_joint_pos = torch.zeros(env.num_envs, self.nj, device=env.device)
        self.boostraping = True # Force reset, Ignore reward
         
        # Cache slot used by observations / rewards.
        self.cached_pos_error_w: torch.Tensor | None = None
        # ───────────── New: Obstacle randomization bookkeeping ─────────────
        self.obstacle_entities = []
        self.obstacle_pos_ranges = {}
        self.obstacle_pos_ranges_initialized = False
        
        # Pre-allocated buffer that auto-flushes to disk every FLUSH_SIZE rows.
        # self.records = RecordBuffer()

        # Single global reference — everything accessed via env.ffw_state.*
        env.ffw_state = self  # type: ignore[attr-defined]

    def setObstacleRanges(
      self , scene_cfg : SceneCfg , _OBSTACLE_POS_RANGES: dict[str, dict]
      ):
      if not self.obstacle_pos_ranges_initialized:
        self.obstacle_pos_ranges_initialized = True
        for entity_name in scene_cfg.entities:   # or env.scene.entity_names if you have that
              if entity_name == "robot":
                  continue
              prefix = "_".join(entity_name.split("_")[:2])
              pos_range = _OBSTACLE_POS_RANGES.get(prefix)
              if pos_range is not None:
                  self.obstacle_entities.append(entity_name)
                  self.obstacle_pos_ranges[entity_name] = pos_range
      
    
    def __call__(
        self, 
        env: ManagerBasedRlEnv, 
        env_ids: torch.Tensor, 
        velocity_range: tuple[float, float] | None = None,
        asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
        scene_cfg: SceneCfg = _DEFAULT_SCENE_CFG_prototype,
    ) -> None:  
         
        self.setObstacleRanges(scene_cfg, _OBSTACLE_POS_RANGES)
        env_ids = env_ids.to(env.device, dtype=torch.long)
        
        # Filter env_ids based on robot_need_reset
        reset_robot_ids = env_ids[self.robot_need_reset[env_ids]]
        reset_obstacle_ids = env_ids[~self.robot_need_reset[env_ids]] 
        # print(env_ids , "env_ids")
        # print(self.robot_need_reset[env_ids] , "robot_need_reset for env_ids")
        # print(reset_obstacle_ids , "reset_obstacle_ids")
        # print(reset_robot_ids , "moveout_obstacle_pose , reset_robot_ids")
        # breakpoint()
      
        asset: Entity = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        nj = len(joint_ids) if isinstance(joint_ids, (list, torch.Tensor)) else asset.data.joint_pos.shape[-1]
        if nj != self.nj:
          self.nj = nj
          self.target_joint_pos = torch.zeros(env.num_envs, nj, device=env.device)
          self.target_tobe_joint_pos = torch.zeros(env.num_envs, nj, device=env.device)
          self.target_next_joint_pos = torch.zeros(env.num_envs, nj, device=env.device)

        self._asset_cfg = asset_cfg
        reset_joints_uniformly(
          env, 
          reset_robot_ids, 
          velocity_range, 
          asset_cfg,
        )
        
        
        # 2. Randomize all relevant obstacles
        for entity_name in self.obstacle_entities:
            pos_range = self.obstacle_pos_ranges[entity_name]
            
            # You can reuse your existing function
            # randomize_obstacle_pose(
            #     env=env,
            #     env_ids=reset_obstacle_ids,
            #     asset_cfg=SceneEntityCfg(entity_name),
            #     pos_range=pos_range,
            # )
            moveout_obstacle_pose(
                env=env,
                env_ids=reset_robot_ids,
                asset_cfg=SceneEntityCfg(entity_name), 
                pos_range=pos_range,
            )  
        self.env_ready[env_ids] = False

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
             env_ids = slice(None)
        # Reset internal state if needed
        pass
      


def randomize_obstacle_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
    pos_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Randomize the pose of a mocap obstacle using polar coordinates for XY.

    XY is sampled in polar form: r ∈ [r1, r2], phi ∈ [phi1, phi2] (radians),
    then converted to x = r·cos(phi), y = r·sin(phi).  Z is sampled linearly.

    Args:
        env: The environment.
        env_ids: Environment IDs to randomize.
        asset_cfg: Scene entity config pointing to the obstacle entity.
        pos_range: Dict with keys:
            "r":   (r_min, r_max)       — radial distance in metres
            "phi": (phi_min, phi_max)   — azimuthal angle in radians
            "z":   (z_min, z_max)       — height in metres
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)

    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]

    if pos_range is None:
        pos_range = {"r": (0.5, 0.8), "phi": (-torch.pi, torch.pi), "z": (0.4, 1.2)}
    # print("randomize")
    # Center of the polar disk (default: world origin).
    cx, cy = pos_range.get("center", (0.0, 0.0))

    n = len(env_ids)

    r   = sample_uniform(*pos_range["r"],   (n,), env.device)
    phi = sample_uniform(*pos_range["phi"], (n,), env.device)

    pos = torch.zeros(n, 3, device=env.device)
    pos[:, 0] = cx + r * torch.cos(phi)
    pos[:, 1] = cy + r * torch.sin(phi)
    pos[:, 2] = sample_uniform(*pos_range["z"], (n,), env.device)
    # Identity quaternion (w, x, y, z) — no rotation randomization for now.
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0

    mocap_pose = torch.cat([pos, quat], dim=-1)  # (n, 7)
    asset.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)

def moveout_obstacle_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
    pos_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Move out the obstacle to a fixed far-away location (e.g., x=5.0, y=5.0, z=5.0) for "no obstacle" condition.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)

    if len(env_ids) == 0:
        return
    asset: Entity = env.scene[asset_cfg.name]


    pos = torch.repeat_interleave(torch.tensor([[5.0, 5.0, 5.0]], device=env.device), len(env_ids), dim=0)

    # Identity quaternion (w, x, y, z) — no rotation randomization for now.
    quat = torch.zeros(len(env_ids), 4, device=env.device)
    quat[:, 0] = 1.0

    mocap_pose = torch.cat([pos, quat], dim=-1)  # (n, 7)
    asset.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)


class RecordBuffer:
  """Pre-allocated CPU buffer that flushes to a single .npz file on disk.

  Each flush *appends* — existing data in the file is preserved and the new
  chunk is concatenated.  The file is memory-mapped on read so loading stays
  cheap even when it grows large.
  """

  FLUSH_SIZE = 1_000_000  # rows before auto-flush

  def __init__(self, path: str = "/puffertank/mjlab/record/ffw_records.npz"):
    self.path = path
    self._bufs: dict[str, list[torch.Tensor]] = {}  # key -> list of cpu tensors
    self._count = 0        # rows buffered in RAM
    self._total_flushed = 0  # rows already written to disk

  # -- public API used by terminations ----------------------------------------

  def append(self, record: dict[str, torch.Tensor]) -> None:
    """Append one batch of rows (variable size along dim-0)."""
    n = next(iter(record.values())).shape[0]
    for k, v in record.items():
      self._bufs.setdefault(k, []).append(v)
    self._count += n
    if self._count >= self.FLUSH_SIZE:
      self.flush()

  def flush(self) -> None:
    """Concatenate buffered tensors, append to the .npz file, free RAM."""
    if self._count == 0:
      return
    # Concat this chunk.
    chunk: dict[str, np.ndarray] = {
      k: torch.cat(vs, dim=0).numpy() for k, vs in self._bufs.items()
    }
    # If file already exists, load + concat.
    if os.path.exists(self.path):
      existing = dict(np.load(self.path))
      for k in chunk:
        if k in existing:
          chunk[k] = np.concatenate([existing[k], chunk[k]], axis=0)
    np.savez(self.path, **chunk)
    self._total_flushed += self._count
    self._bufs.clear()
    self._count = 0

  @property
  def total(self) -> int:
    return self._total_flushed + self._count

  def __len__(self) -> int:
    return self._count