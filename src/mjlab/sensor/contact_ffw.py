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
    mode: Literal["geom", "body"]
    pattern: str
    entity: str | None = None


@dataclass
class ContactSensorFFWCfg(SensorCfg):
    primary: ContactMatchFFW
    secondary: ContactMatchFFW | None = None
    num_slots: int = 2          # ← new: how many top contacts to keep (1–8 recommended)

    def build(self) -> 'ContactSensorFFW':
        return ContactSensorFFW(self)


@dataclass
class ContactDataFFW:
    collision_detected: torch.Tensor     # [B, 1] bool — any contact at all
    valid_mask: torch.Tensor            # [B, N_prim * K] bool — which slots are valid (found > 0)
    force: torch.Tensor                  # [B, N_prim * K, 3] — top-K contact forces (global)
    pos: torch.Tensor                    # [B, N_prim * K, 3] — contact points (global)
    occupancy: torch.Tensor             # [B, R³] — voxelized occupancy 
    top_force : torch.Tensor             # [B, T, 3] — top-T forces only
    top_pos : torch.Tensor               # [B, T, 3] — top-T contact points only
    # normal: torch.Tensor               # [B, N_prim * K, 3] — contact normals


class ContactSensorFFW(Sensor[ContactDataFFW]):
    def __init__(self, cfg: ContactSensorFFWCfg) -> None:
        self.cfg = cfg
        self._sensor_names: list[str] = []
        self._data_views: list[torch.Tensor] = []
        self._device: str | None = None
        self._num_slots = cfg.num_slots
        self._slot_dim = 7  # found(1) + force(3) + pos(3) + normal(3)
        self.voxel_resolution = 8  # → 512 voxels for occupancy grid

    def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
        self._sensor_names.clear()

        primary_names = self._resolve_names(entities, self.cfg.primary)
        
        secondary_name = None
        
        if self.cfg.secondary:
            secondary_names = self._resolve_names(entities, self.cfg.secondary)
            if len(secondary_names) != 1:
                raise ValueError("FFWContactSensor only supports a single secondary target.")
            secondary_name = secondary_names[0]
        
        # Bitmask: found (0) + force (1) + pos (4) + normal (5) → 1 | 2 | 16 | 32 = 51
        data_bits = (1 << 0) | (1 << 1) | (1 << 4) #| (1 << 5)
        reduce_mode = 2  # maxforce = strongest normal force
        intprm = [data_bits, reduce_mode, self._num_slots]
        for prim in primary_names:
            sensor_name = f"{self.cfg.name}_{prim}_rich"
            # ... (same as before: objname, objtype, kwargs, add_sensor)
            primary_entity = self.cfg.primary.entity
            objname = f"{primary_entity}/{prim}" if primary_entity else prim
            objtype = (
                mujoco.mjtObj.mjOBJ_BODY if self.cfg.primary.mode == "body"
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
                    mujoco.mjtObj.mjOBJ_BODY if self.cfg.secondary.mode == "body"
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
        self._device = device
        B = data.sensordata.shape[0]
        self.occupancy_map = torch.zeros((B, self.voxel_resolution ** 3), device=device)
        expected_dim = self._num_slots * self._slot_dim

        for name in self._sensor_names:
            sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid == -1:
                continue
            adr = mj_model.sensor_adr[sid]
            dim = mj_model.sensor_dim[sid]
            if dim != expected_dim:
                print(f"Warning: sensor {name} dim {dim} ≠ {expected_dim}")
                continue
            view = data.sensordata[:, adr : adr + dim]
            self._data_views.append(view)

    def _compute_data(self) -> ContactDataFFW:
        """Required by abstract base class Sensor."""
        return self.data
    
    @property
    def data(self) -> ContactDataFFW:
        K = self._num_slots
        n_prim = len(self._data_views)
        if not self._data_views:
            
            B = self._mjwarp_data.qpos.shape[0]

            empty = lambda shape: torch.zeros(shape, device=self._device, dtype=torch.float32)
            return ContactDataFFW(
                collision_detected=torch.zeros((B, 1), dtype=torch.bool, device=self._device),
                valid_mask=torch.zeros((B,  K * n_prim), dtype=torch.bool, device=self._device),
                force=empty((B, K * n_prim, 3)),
                pos=empty((B, K * n_prim, 3)),
                occupancy=empty((B, self.voxel_resolution ** 3)),  # e.g., 8x8x8 grid → 512 voxels
                top_force=empty((B, 3, 3)),
                top_pos=empty((B, 3, 3)),
                # normal=empty((B, K * n_prim, 3)),
            )

        # Infer batch size from the first view [B, dim]
        B = self._data_views[0].shape[0]

        # Concat → [B, N_prim * (K*10)]
        all_data = torch.cat(self._data_views, dim=1)

        # Reshape → [B, N_prim, K, 10]
        
        all_data = all_data.view(B, n_prim, K, self._slot_dim)

        # found [B, N_prim, K]
        found = all_data[..., 0]
        valid_mask  = (found > 0).reshape(B, -1)   # → [B, 52]
        # Any contact anywhere?
        collision = (found > 0).any(dim=(1, 2)).unsqueeze(-1)  # [B, 1]

        # For simplicity: concatenate all primaries' top-K → [B, N_prim*K, ...]
        # You can later sort / filter per env if needed
        all_force  = all_data[..., 1:4]   .view(B, n_prim * K, 3)
        all_pos    = all_data[..., 4:7]   .view(B, n_prim * K, 3)
        # all_normal = all_data[..., 7:10]  .view(B, n_prim * K, 3)
        
        force_mag = torch.norm(all_force, dim=-1)                   # [B, M]  M=52
        force_mag = force_mag * valid_mask.float()             # mask invalid
        top_mag, top_idx = torch.topk(force_mag, k=3, dim=1,
                                      largest=True, sorted=True)
        top_force = torch.gather(all_force, 1,
                                 top_idx.unsqueeze(-1).expand(-1, -1, 3))
        top_pos   = torch.gather(all_pos, 1,
                                 top_idx.unsqueeze(-1).expand(-1, -1, 3))
        voxel = self.points_to_occupancy(points=all_pos, mask=valid_mask,
                                voxel_size=0.25, grid_resolution=8, 
                                flatten=True)
        self.occupancy_map *= 0.95 # decay old occupancy
        self.occupancy_map += voxel # add new occupancy
        self.occupancy_map = torch.clamp(self.occupancy_map, -1.0, 10.0)
        
        
        return ContactDataFFW(
            collision_detected=collision,
            valid_mask=valid_mask,
            force=all_force,
            pos=all_pos,
            occupancy=self.occupancy_map,
            top_force=top_force,
            top_pos=top_pos,
            # normal=all_normal,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
          env_ids = slice(None)
        self.occupancy_map[env_ids] = 0.0
    def points_to_occupancy(
            self,
            points: torch.Tensor,                   # (B, P, 3) world coordinates
            mask: torch.Tensor | None = None,       # (B, P)    bool — True = include
            voxel_size: float = 0.25,              # 8 * 0.25 = 2.0 span → covers [-1, +1]
            grid_resolution: int = 8,
            offset: float | None = None,            # default = G/2 * voxel_size = 4*0.25 = 1.0
            flatten: bool = False,
        ) -> torch.Tensor:
        """
        Occupancy-only batched voxelization.

        G=8, voxel_size=0.25, offset=1.0  →  covers [-1, +1] in x, y, z

        world → voxel:   idx    = floor((world + offset) / voxel_size)
        voxel → world:   centre = (idx + 0.5) * voxel_size - offset

        Returns:
            flatten=False  →  (B, G, G, G)
            flatten=True   →  (B, G³)
        """
        device = points.device
        B, P, _ = points.shape
        G = grid_resolution
        voxels_per_env = G ** 3

        if offset is None:
            offset = (G / 2) * voxel_size      # 4 * 0.25 = 1.0

        # ── Voxel indices (B, P, 3) ───────────────────────────────────
        voxel_idx = ((points + offset) / voxel_size).floor().long().clamp(0, G - 1)

        # ── Flat global index  (z*G*G + y*G + x) ─────────────────────
        flat_idx = (
            voxel_idx[..., 2] * (G * G) +
            voxel_idx[..., 1] * G +
            voxel_idx[..., 0]
        )                                                           # (B, P)

        # ── Pack all B envs into one flat buffer ──────────────────────
        # env 0 → slots [0,       G³)
        # env 1 → slots [G³,    2*G³)
        # env k → slots [k*G³, (k+1)*G³)
        env_offsets     = torch.arange(B, device=device) * voxels_per_env  # [0, 512, 1024, ...]
        global_flat_idx = env_offsets.unsqueeze(1) + flat_idx              # (B, P)

        # ── Mask: redirect invalid points to scratch slot ─────────────
        real_total = B * voxels_per_env
        if mask is not None:
            global_flat_idx = torch.where(
                mask,
                global_flat_idx,
                torch.full_like(global_flat_idx, real_total),      # one scratch slot at the end
            )
            total_elements = real_total + 1
        else:
            total_elements = real_total

        # ── Scatter-add: count points per voxel ──────────────────────
        out_flat = torch.zeros(total_elements, device=device, dtype=torch.float32)
        out_flat.scatter_add_(0, global_flat_idx.flatten(),
                              torch.ones(B * P, device=device, dtype=torch.float32))

        out = out_flat[:real_total]                                 # trim scratch slot
        return out.view(B, voxels_per_env) if flatten else out.view(B, G, G, G)
    
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