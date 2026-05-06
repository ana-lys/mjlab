"""Robotis FFW minimal task environment configuration."""

import torch
import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.asset_zoo.robots.robotis_ffw.ffw_constant import get_robotis_ffw_robot_cfg
from mjlab.asset_zoo.robots.robotis_ffw.ffw_obstacles import get_obstacle_cfgs
from mjlab.envs import mdp
from mjlab.tasks.ffw import mdp as ffw_mdp
from mjlab.tasks.ffw.mdp.commands import TargetPoseCommandCfg, TargetPoseCommand

from mjlab.sensor.contact_ffw import ContactSensorFFWCfg as ContactSensorCfg
from mjlab.sensor.contact_ffw import ContactMatchFFW as ContactMatch

SCENE_CFG = SceneCfg(
  num_envs=8192,
  extent=1.0,
  entities={
    "robot": get_robotis_ffw_robot_cfg(),
    **get_obstacle_cfgs(num_tables=0, num_poles=0, num_cubes=0, num_capsules=0),
  },
  sensors=(
      ContactSensorCfg(
          name="contact_sensor",
          primary=ContactMatch(mode="body", pattern="^(?!.*(pole|drive)).*", entity="robot"),
          num_slots = 2,
      ),
  ),
)

VIEWER_CONFIG = ViewerConfig(
  entity_name="robot",
  body_name="arm_base_link",
  distance=3.0,
  elevation=15.0,
  azimuth=20.0,
)

SIM_CFG = SimulationCfg(
    njmax=200,
    nconmax=200,
    mujoco=MujocoCfg(
    # 50 Hz env step with decimation=1 -> timestep = 0.02 s.
    timestep=0.02,
    iterations=10,
    ls_iterations=20,
    ccd_iterations=25,
  ),
)


def create_ffw_actions() -> dict[str, ActionTermCfg]:
    """Policy outputs joint position targets for all actuators.

    Automatically map [-1, 1] to each actuator's full ctrlrange from XML.
    """
    import mujoco
    from mjlab.asset_zoo.robots.robotis_ffw.ffw_constant import get_spec

    spec = get_spec()
    model = spec.compile()

    scale = {}
    offset = {}
    actuator_patterns = []
    exclude_keywords = ["head", "gripper"]

    for i in range(model.nu):
      # Map actuator to its controlled joint index (first element)
      j = int(model.actuator_trnid[i][0])
      joint_name = model.joint(j).name
      actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)

      if any(k in actuator_name for k in exclude_keywords) or any(k in joint_name for k in exclude_keywords):
        continue

      # Use joint name as the pattern since FindJoints expects joint names
      actuator_patterns.append(joint_name)
      
      ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
      scale[joint_name] = float(0.5 * (ctrl_max - ctrl_min))
      offset[joint_name] = float(0.5 * (ctrl_max + ctrl_min))

    return {
        "joint_pos": JointPositionActionCfg(
          entity_name="robot",
          actuator_names=actuator_patterns,
          scale=scale,
          offset=offset,
          use_default_offset=False,
        ),
    }

def create_ffw_observations() -> dict[str, ObservationGroupCfg]:
  """Observe joint pos/vel relative to defaults; include last action."""
  joint_filter = "^(?!.*(head|gripper)).*"

  def get_terms():
    return {
      "joint_pos_rel": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(joint_filter,))},
      ),
      "joint_vel_rel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(joint_filter,))},
      ),
      "top_collision_force": ObservationTermCfg(
        func=ffw_mdp.top_collision_forces,
        params={"sensor_name": "contact_sensor"},
      ),
      "top_collision_pos": ObservationTermCfg(
        func=ffw_mdp.top_collision_pos,
        params={"sensor_name": "contact_sensor"},
      ),
      # "collision_occupancy": ObservationTermCfg(
      #   func=ffw_mdp.collision_occupancy,
      #   params={"sensor_name": "contact_sensor"},
      # ),
      "ee_pos": ObservationTermCfg(
        func=ffw_mdp.target_ee_pos,
      ),
      "last_action": ObservationTermCfg(func=mdp.last_action),
      "ee_pos_error": ObservationTermCfg(
        func=ffw_mdp.ee_pos_error,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", site_names=("left_gripper_site", "right_gripper_site")
          )
        },
      ),
      # "body_pos": ObservationTermCfg(
      #   func=mdp.body_pos_flat,
      #   params={
      #     "asset_cfg": SceneEntityCfg(
      #       "robot", body_names=("arm_[lr]_link[3-7]",)
      #     )
      #   },
      # ),
      
    }

  # Set concatenate_terms=False to keep observations as a dict of terms
  return {
    "actor": ObservationGroupCfg(terms=get_terms(), concatenate_terms=True),
    "critic": ObservationGroupCfg(terms=get_terms(), concatenate_terms=True),
  }

def create_ffw_commands() -> dict[str, CommandTermCfg]:
    """Define commands for the task."""
    return {
        "target_pose": TargetPoseCommandCfg(
            class_type=TargetPoseCommand,
            resampling_time_range=(900000, 1000000.0),
            debug_vis=True,
        )
    }

def create_ffw_rewards() -> dict[str, RewardTermCfg]:
  """FFW rewards focused on end-effector pose tracking and stability."""
  
  return {
    "ee_track_l2": RewardTermCfg(
        func=ffw_mdp.ee_tracking_l2_bimanual,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("left_gripper_site", "right_gripper_site")
            ),
        },
    ),
    "ee_track_fine": RewardTermCfg(
        func=ffw_mdp.ee_tracking_tanh_bimanual,
        weight=3.0,
        params={
            "std": 0.05, # ~5cm precision for max bonus
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("left_gripper_site", "right_gripper_site")
            ),
        },
    ),
    "joint_vel_l2": RewardTermCfg(
        func=ffw_mdp.joint_vel_l2,
        weight=-0.00035,
        params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "collision": RewardTermCfg(
        func=ffw_mdp.collision_penalty,
        weight=-50.0,
        params={"sensor_name": "contact_sensor"},
    ),
    "joint_acc_l2": RewardTermCfg(
        func=ffw_mdp.joint_acc_l2,
        weight=-2.5e-8,
    ),
  }


def create_ffw_events() -> dict[str, EventTermCfg]:
  """Reset + random end-effector goal sampling + obstacle randomization."""

  events: dict[str, EventTermCfg] = {
     "reset_robot_joints": EventTermCfg(
      func=ffw_mdp.StatefulReset,  # Pass the class, manager instantiates it
      mode="reset",
      params={
        "velocity_range": (-1.0, 1.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("^(?!.*(head|gripper)).*",)),
        "scene_cfg": SCENE_CFG,
      },
    ),
  }
  return events


def create_ffw_terminations() -> dict[str, TerminationTermCfg]:
  """Timeout + Collision."""
  return {
    "timeout": TerminationTermCfg(func=ffw_mdp.stateful_time_out, time_out=True),
    "illegal_contact": TerminationTermCfg(
        func=ffw_mdp.scene_shuffle,
        params={"sensor_name": "contact_sensor"},
    ),
  }


def create_ffw_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Robotis FFW minimal environment configuration."""
  return ManagerBasedRlEnvCfg(
    scene=SCENE_CFG,
    observations=create_ffw_observations(),
    actions=create_ffw_actions(),
    commands=create_ffw_commands(),
    rewards=create_ffw_rewards(),
    events=create_ffw_events(),
    terminations=create_ffw_terminations(),
    sim=SIM_CFG,
    viewer=VIEWER_CONFIG,
    # Decimation=1 so env step frequency matches simulation (50 Hz).
    decimation=1,
    episode_length_s=0.6, # 200 steps * 0.02s
  )


FFW_MINIMAL_ENV_CFG = create_ffw_env_cfg()
