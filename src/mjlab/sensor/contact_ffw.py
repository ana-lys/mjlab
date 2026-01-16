"""Simplified contact sensor for FFW robot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg


@dataclass
class ContactMatchFFW:
  """Specifies what to match on one side of a contact."""

  mode: Literal["geom", "body"]
  pattern: str
  entity: str | None = None


@dataclass
class ContactSensorFFWCfg(SensorCfg):
  """Configuration for FFW contact sensor."""

  primary: ContactMatchFFW
  secondary: ContactMatchFFW | None = None

  def build(self) -> ContactSensorFFW:
    return ContactSensorFFW(self)


@dataclass
class ContactDataFFW:
  """Data from FFW contact sensor."""

  collision_detected: torch.Tensor
  """[B, 1] Boolean indicating if any collision occurred."""


class ContactSensorFFW(Sensor[ContactDataFFW]):
  """Simplified contact sensor."""

  def __init__(self, cfg: ContactSensorFFWCfg) -> None:
    self.cfg = cfg
    self._sensor_names: list[str] = []
    self._data_views: list[torch.Tensor] = []

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    self._sensor_names.clear()

    primary_names = self._resolve_names(entities, self.cfg.primary)

    secondary_name = None
    if self.cfg.secondary:
      secondary_names = self._resolve_names(entities, self.cfg.secondary)
      if len(secondary_names) != 1:
        raise ValueError("FFWContactSensor only supports a single secondary target.")
      secondary_name = secondary_names[0]

    for prim in primary_names:
      sensor_name = f"{self.cfg.name}_{prim}"
      # 1=found, 2=maxforce (strongest), 1=num_slots
      intprm = [1, 2, 1]

      primary_entity = self.cfg.primary.entity
      objname = f"{primary_entity}/{prim}" if primary_entity else prim
      objtype = (
        mujoco.mjtObj.mjOBJ_BODY
        if self.cfg.primary.mode == "body"
        else mujoco.mjtObj.mjOBJ_GEOM
      )

      kwargs = {
        "name": sensor_name,
        "type": mujoco.mjtSensor.mjSENS_CONTACT,
        "objtype": objtype,
        "objname": objname,
        "intprm": intprm,
      }

      if secondary_name:
        secondary_entity = self.cfg.secondary.entity
        refname = f"{secondary_entity}/{secondary_name}" if secondary_entity else secondary_name
        reftype = (
          mujoco.mjtObj.mjOBJ_BODY
          if self.cfg.secondary.mode == "body"
          else mujoco.mjtObj.mjOBJ_GEOM
        )
        kwargs.update({"reftype": reftype, "refname": refname})

      scene_spec.add_sensor(**kwargs)
      self._sensor_names.append(sensor_name)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    self._data_views.clear()
    for name in self._sensor_names:
      id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
      if id == -1:
        continue

      sensor_adr = mj_model.sensor_adr[id]
      sensor_dim = mj_model.sensor_dim[id]

      if sensor_dim != 1:
        continue

      view = data.sensordata[:, sensor_adr : sensor_adr + sensor_dim]
      self._data_views.append(view)

  @property
  def data(self) -> ContactDataFFW:
    if not self._data_views:
      return ContactDataFFW(collision_detected=torch.tensor([]))

    # Sum 'found' from all primary bodies. Each view is [B, 1] (non-negative)
    total_contacts = sum(self._data_views) if self._data_views else 0
    if isinstance(total_contacts, int):  # Should be tensor unless empty
        collision = torch.tensor([], device=self._data_views[0].device) if self._data_views else torch.tensor([])
    else:
        collision = (total_contacts > 0).float()

    return ContactDataFFW(collision_detected=collision)

  def _resolve_names(
    self, entities: dict[str, Entity], match: ContactMatchFFW
  ) -> list[str]:
    if not match.entity:
      return [match.pattern]

    if match.entity not in entities:
      raise ValueError(f"Entity '{match.entity}' not found.")

    ent = entities[match.entity]

    if match.mode == "geom":
      _, names = ent.find_geoms(match.pattern)
    elif match.mode == "body":
      _, names = ent.find_bodies(match.pattern)
    else:
      raise ValueError(f"Unsupported mode: {match.mode}")

    if not names:
      raise ValueError(
        f"Pattern '{match.pattern}' matched nothing in '{match.entity}'"
      )

    return names
