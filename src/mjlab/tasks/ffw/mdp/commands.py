from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import numpy as np

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass
class TargetPoseCommandCfg(CommandTermCfg):
    """Minimal configuration for target pose command."""
    class_type: type = None
    resampling_time_range: tuple[float, float] = (2.0, 5.0)
    num_targets: int = 2
    debug_vis: bool = False

    def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
        return self.class_type(self, env)  # type: ignore


class TargetPoseCommand(CommandTerm):
    """Barebones no-op target pose command.

    Keeps command manager interfaces satisfied while target logic lives elsewhere.
    """

    cfg: TargetPoseCommandCfg

    def __init__(self, cfg: TargetPoseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.target_pos = torch.zeros(
            self.num_envs, self.cfg.num_targets, 3, device=self.device
        )
        self.target_quat = torch.zeros(
            self.num_envs, self.cfg.num_targets, 4, device=self.device
        )
        self.target_quat[..., 0] = 1.0

    @property
    def command(self) -> torch.Tensor:
        """Current no-op target pose buffer."""
        return torch.cat([self.target_pos, self.target_quat], dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # No resampling needed; targets remain whatever external logic sets.
        return

    def _update_command(self) -> None:
        # No-op: external logic manages targets.
        return

    def _update_metrics(self) -> None:
        # No metrics to update.
        return

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        sr = getattr(self._env, "ffw_state", None)
        if sr is None:
            return
        target = sr.target_ee_pos

        target_np = target.detach().cpu().numpy()
        radius = max(0.03, visualizer.meansize * 0.06)
        colors = [
            (0.9, 0.2, 0.2, 0.9),  # first target
            (0.2, 0.6, 0.9, 0.9),  # second target
        ]

        for env_idx in env_indices:
            for t in range(target_np.shape[1]):
                visualizer.add_sphere(
                    center=target_np[env_idx, t],
                    radius=radius,
                    color=colors[t % len(colors)],
                    label=f"target_{t}",
                )
