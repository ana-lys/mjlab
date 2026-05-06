"""Obstacle EntityCfg factories for the FFW task.

Each obstacle is a single-body XML (no joints → auto-wrapped as mocap).
Poses are randomized per-episode via event functions.
"""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

_XML_DIR = Path(__file__).parent / "xmls"


def _make_spec_fn(xml_name: str):
    """Return a zero-arg callable that loads a spec from *xml_name*."""
    xml_path = _XML_DIR / xml_name
    assert xml_path.exists(), f"Obstacle XML not found: {xml_path}"

    def _load() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(xml_path))

    return _load


def get_obstacle_cfgs(
    num_tables: int = 1,
    num_poles: int = 1,
    num_cubes: int = 1,
    num_capsules: int = 1,
) -> dict[str, EntityCfg]:
    """Return a dict of obstacle EntityCfg, keyed by scene-entity name.

    Args:
        num_tables: Number of table obstacles.
        num_poles: Number of pole obstacles.
        num_cubes: Number of small cube obstacles.
        num_capsules: Number of small capsule obstacles.
    """
    cfgs: dict[str, EntityCfg] = {}
    _templates = [
        ("obs_table", "obstacle_table.xml", (0.0, 0.0, 0.0), num_tables),
        ("obs_pole", "obstacle_pole.xml", (0.0, 0.0, 0.0), num_poles),
        ("obs_cube", "obstacle_cube.xml", (0.0, 0.0, 0.0), num_cubes),
        ("obs_capsule", "obstacle_capsule.xml", (0.0, 0.0, 0.0), num_capsules),
    ]
    for prefix, xml_name, default_pos, count in _templates:
        for i in range(count):
            name = f"{prefix}_{i}" if count > 1 else prefix
            cfgs[name] = EntityCfg(
                spec_fn=_make_spec_fn(xml_name),
                init_state=EntityCfg.InitialStateCfg(pos=default_pos),
            )
    return cfgs
