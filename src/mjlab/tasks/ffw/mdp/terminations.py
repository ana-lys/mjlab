from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensorFFW

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Terminate if collision is detected."""
  sensor: ContactSensorFFW = env.scene[sensor_name]
  # FFW sensor returns [B, 1] for collision_detected
  return sensor.data.collision_detected.squeeze(-1).bool()

