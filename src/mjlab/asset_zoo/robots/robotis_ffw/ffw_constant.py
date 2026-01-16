from pathlib import Path
import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlPositionActuatorCfg

ROBOTIS_FFW_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "robotis_ffw" / "xmls" / "scene.xml"
)
assert ROBOTIS_FFW_XML.exists(), f"XML not found: {ROBOTIS_FFW_XML}"

STANDBY_FRAME= EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
      "lift_joint": 0.0,
      "arm_l_joint1": 0,
      "arm_r_joint1": 0,
      "arm_l_joint2": 0.0, "arm_l_joint3": 0.0, "arm_l_joint4": 0.0,
      "arm_l_joint5": 0.0, "arm_l_joint6": 0.0, "arm_l_joint7": 0.0,
      "arm_r_joint2": 0.0, "arm_r_joint3": 0.0, "arm_r_joint4": 0.0,
      "arm_r_joint5": 0.0, "arm_r_joint6": 0.0, "arm_r_joint7": 0.0,
      "gripper_*": 0.0,
      "head_*": 0.0,
  }, 
  joint_vel={".*": 0.0},
)

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(ROBOTIS_FFW_XML))

def get_robotis_ffw_robot_cfg() -> EntityCfg:
  """Get a fresh Robotis FFW robot configuration instance.

  Also wires XML-defined actuators so mjlab actions can control the joints.
  """
  return EntityCfg(
    spec_fn=get_spec,
    init_state=STANDBY_FRAME,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlPositionActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer
  robot = Entity(get_robotis_ffw_robot_cfg())
  viewer.launch(robot.spec.compile())