from mjlab.tasks.registry import register_mjlab_task

from .ffw_env_cfg import FFW_MINIMAL_ENV_CFG
from .ffw_rl_cfg import ffw_ppo_runner_cfg

# Register with mjlab task registry so `train` recognizes it.
register_mjlab_task(
  task_id="Mjlab-FFW-Minimal",
  env_cfg=FFW_MINIMAL_ENV_CFG,
  play_env_cfg=FFW_MINIMAL_ENV_CFG,
  rl_cfg=ffw_ppo_runner_cfg(),
  runner_cls=None,
)
