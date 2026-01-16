from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass
class TargetPoseCommandCfg(CommandTermCfg):
    """Configuration for the target pose command."""
    class_type: type = None  # Will be set to TargetPoseCommand
    resampling_time_range: tuple[float, float] = (2.0, 5.0)
    
    # Ranges for the target position relative to the robot base (x, y, z)
    range_rel_x: tuple[float, float] = (0.3, 0.6)
    
    # Separate Y ranges to keep hands on their respective sides
    range_rel_y_left: tuple[float, float] = (-0.2, 0.4)   # Positive Y (Left)
    range_rel_y_right: tuple[float, float] = (-0.4, 0.2) # Negative Y (Right)
    range_rel_z: tuple[float, float] = (-0.3, 0.3)

    range_roll: tuple[float, float] = (-0.6, 0.6)
    range_pitch: tuple[float, float] = (-0.6, 0.6)
    range_yaw: tuple[float, float] = (-0.6, 0.6)
    
    num_targets: int = 2  # Number of targets to generate
    
    debug_vis: bool = True

    def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
        return self.class_type(self, env) # type: ignore


class TargetPoseCommand(CommandTerm):
    """Command that generates a target pose for the end-effector.
    
    Generates targets in world frame by sampling offsets in the robot's base frame.
    """
    
    cfg: TargetPoseCommandCfg

    def __init__(self, cfg: TargetPoseCommandCfg, env: ManagerBasedRlEnv):

        super().__init__(cfg, env)
        self.target_pos = torch.zeros(self.num_envs, self.cfg.num_targets, 3, device=self.device)
        self.target_quat = torch.zeros(self.num_envs, self.cfg.num_targets, 4, device=self.device)
        self.deltas_local = torch.tensor([
        [ 0.00,  0.00,  0.00],   # root / center
        [-0.20,  0.00,  0.00],   # back ≈ -x_local
        [ 0.00,  0.00, +0.20],   # up   ≈ +z_local
        ], device=self.device)  # [3, 3]
        self.target_quat[..., 0] = 1.0 # Identity (w=1)
        
        # Resample immediately to set valid non-zero targets and safe attribute
        self._resample_command(torch.arange(self.num_envs, device=self.device))
        
        setattr(self._env, "target_ee_pos", self.target_pos)
        setattr(self._env, "target_ee_quat", self.target_quat)
        setattr(self._env, "target_pose", self.command) # [N, T, 7]
        
        # Metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The current target pose (pos + quat)."""
        return torch.cat([self.target_pos, self.target_quat], dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Resample the target pose."""
        from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

        # Indices into env_ids that still need valid targets
        remaining_indices = torch.arange(len(env_ids), device=self.device)
        curr_env_ids = env_ids
        
        # Pre-fetch robot state (assuming static during resampling calculation)
        robot = self._env.scene["robot"]
        all_root_pos = robot.data.root_link_pos_w[env_ids]
        all_root_quat = robot.data.root_link_quat_w[env_ids]

        # Attempt resampling until valid
        for _ in range(5):
            if len(remaining_indices) == 0:
                break
            
            num_resample = len(remaining_indices)
            global_idxs = curr_env_ids[remaining_indices]

            # Use zeros to ensure cleaner initialization
            offsets_b = torch.zeros(num_resample, self.cfg.num_targets, 3, device=self.device)
            # Sample relative offsets
            # X and Z ranges are shared
            offsets_b[..., 0].uniform_(*self.cfg.range_rel_x)
            offsets_b[..., 2].uniform_(*self.cfg.range_rel_z)
            # Y ranges are separate for left/right targets
            offsets_b[:, 0, 1].uniform_(*self.cfg.range_rel_y_left)
            offsets_b[:, 1, 1].uniform_(*self.cfg.range_rel_y_right)
                
            # Sample relative orientation (roll, pitch, yaw)
            r = torch.zeros(num_resample, self.cfg.num_targets, device=self.device).uniform_(*self.cfg.range_roll)
            p = torch.zeros(num_resample, self.cfg.num_targets, device=self.device).uniform_(*self.cfg.range_pitch)
            y = torch.zeros(num_resample, self.cfg.num_targets, device=self.device).uniform_(*self.cfg.range_yaw)
            quat_b = quat_from_euler_xyz(r, p, y) # [N, T, 4]
            
            # Transform offsets to world frame using base orientation
            # Use cached root poses
            subset_root_pos = all_root_pos[remaining_indices]
            subset_root_quat = all_root_quat[remaining_indices]
            
            root_quat_exp = subset_root_quat.unsqueeze(1).expand(-1, self.cfg.num_targets, -1)
            offsets_w = quat_apply(root_quat_exp, offsets_b)
            
            virtual_root = subset_root_pos.clone()
            virtual_root[:, 2] = 1.0
            
            self.target_pos[global_idxs] = virtual_root.unsqueeze(1) + offsets_w
            self.target_quat[global_idxs] = quat_mul(root_quat_exp, quat_b)

            # ─── Check Validity (Collision/Reachability) ───
            valid_mask = torch.ones(num_resample, dtype=torch.bool, device=self.device)
            
            # Expand deltas: [B, T, 3, 3]
            deltas = self.deltas_local.expand(num_resample, self.cfg.num_targets, -1, -1)
            
            # Fetch just the resampled subset from global buffer
            quat_subset = self.target_quat[global_idxs] # [B, T, 4]
            
            quat_flat = quat_subset.view(num_resample * self.cfg.num_targets, 4)
            deltas_flat = deltas.view(num_resample * self.cfg.num_targets, 3, 3)
            
            rotated_flat = torch.zeros_like(deltas_flat)
            for i in range(3):
                rotated_flat[:, i, :] = quat_apply(quat_flat, deltas_flat[:, i, :])
            
            points_w = rotated_flat.view(num_resample, self.cfg.num_targets, 3, 3)
            points_w += self.target_pos[global_idxs][:, :, None, :]
            
            left_pts  = points_w[:, 0]   # [B, 3, 3]
            right_pts = points_w[:, 1]   # [B, 3, 3]
            diff = left_pts[:, :, None, :] - right_pts[:, None, :, :]
            dist_9 = torch.norm(diff, dim=-1)  # [B, 3, 3]
            
            min_dist = dist_9.view(num_resample, -1).min(dim=-1).values
            valid_mask = min_dist >= 0.2

            # Keep only invalid indices for next iteration
            remaining_indices = remaining_indices[~valid_mask]
        
        setattr(self._env, "target_ee_pos", self.target_pos)
        setattr(self._env, "target_ee_quat", self.target_quat)
        setattr(self._env, "target_pose", self.command)
        

    def _update_command(self) -> None:
        """Update the command."""
        # TODO: If targets should move WITH the base, update them here. 
        # For now, we keep them fixed in world frame until resampled.
        setattr(self._env, "target_ee_pos", self.target_pos)
        setattr(self._env, "target_ee_quat", self.target_quat)
        setattr(self._env, "target_pose", self.command)

    def _update_metrics(self) -> None:
        """Update metrics."""
        pass

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Visualize the target pose."""
        # Only visualize for the selected environment index
        env_idx = visualizer.env_idx
        
        # Get targets for this specific environment: [NumTargets, 3] and [NumTargets, 4]
        env_pos = self.target_pos[env_idx]
        env_quat = self.target_quat[env_idx]

        from mjlab.utils.lab_api.math import matrix_from_quat
        
        for i in range(env_pos.shape[0]):
            visualizer.add_frame(
                 position=env_pos[i],
                 rotation_matrix=matrix_from_quat(env_quat[i]),
                 scale=0.3,
                 label=f"Target {i}"
            )
