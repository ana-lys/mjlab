from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensorFFW

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.ffw.mdp.reset import StatefulReset

Record_enabled = False

def scene_shuffle(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Terminate if collision is detected."""
  sensor: ContactSensorFFW = env.scene[sensor_name]
  # FFW sensor returns [B, 1] for collision_detected

  collision = sensor.data.collision_detected.squeeze(-1).bool()

  # Promote targets once per env when no collision occurred and env not yet ready.
  sr: StatefulReset | None = getattr(env, "ffw_state", None)
  if sr is not None:
    if sr.boostraping:
      sr.boostraping = False
      return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    not_ready = ~sr.env_ready
    robot_not_ready = sr.robot_need_reset.clone()

    # breakpoint()
    robot_done_reset = robot_not_ready & (~collision)
    if torch.any(robot_done_reset):
    
      cfg = sr._asset_cfg
      asset = env.scene[cfg.name]
      sr.target_joint_pos[robot_done_reset] = sr.target_next_joint_pos[robot_done_reset]
      sr.target_tobe_joint_pos[robot_done_reset] = asset.data.joint_pos[robot_done_reset][:, cfg.joint_ids]
      
      # print("target_ee_pos" , sr.target_ee_pos[robot_done_reset], "target_next_ee_pos" , sr.target_next_ee_pos[robot_done_reset], "target_tobe_ee_pos" , sr.target_tobe_ee_pos[robot_done_reset])
      sr.target_ee_pos[robot_done_reset] = sr.target_next_ee_pos[robot_done_reset]
      # print("target_tobe_ee_pos" , sr.target_tobe_ee_pos[robot_done_reset] , "robot_ee_pos" , asset.data.site_pos_w[robot_done_reset, cfg.site_ids, :])
      sr.target_tobe_ee_pos[robot_done_reset] = asset.data.site_pos_w[robot_done_reset, cfg.site_ids, :]

      
      sr.target_next_ee_pos[robot_done_reset] = sr.target_tobe_ee_pos[robot_done_reset]
      sr.target_next_joint_pos[robot_done_reset] = sr.target_tobe_joint_pos[robot_done_reset]
      
      sr.robot_need_reset[robot_done_reset] = False
      
     
    env_ready = not_ready & (~collision) & (~robot_not_ready) 
    if torch.any(env_ready):
      sr.env_ready[env_ready] = True

  return ~sr.env_ready

def stateful_time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum."""
  sr: StatefulReset | None = getattr(env, "ffw_state", None)
  timed_out = env.episode_length_buf >= env.max_episode_length
  if sr is not None:
      if not sr.boostraping:
            sr.robot_need_reset[timed_out] = True
  return timed_out

      # Record the valid pair *before* overwriting next targets.
      
      # if Record_enabled:
      #   idxs = promote_mask.nonzero(as_tuple=False).squeeze(-1)
      #   sr.records.append({
      #     "target_joint_pos":      sr.target_joint_pos[idxs].cpu(),
      #     "target_ee_pos":         sr.target_ee_pos[idxs].cpu(),
      #     "target_tobe_joint_pos": sr.target_tobe_joint_pos[idxs].cpu(),
      #     "target_tobe_ee_pos":    sr.target_tobe_ee_pos[idxs].cpu(),
      #   })