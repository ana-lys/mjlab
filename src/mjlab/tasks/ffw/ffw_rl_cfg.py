"""RL configuration for FFW minimal task."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlModelCfg,
  RslRlPpoAlgorithmCfg,
  RslRLRnnModelCfg
)

    # actor=RslRlModelCfg(
    #   hidden_dims=(512, 256, 128),
    #   activation="elu",
    #   init_noise_std=0.5,
    #   obs_normalization=True,
    #   stochastic=True,
    # ),
    # critic=RslRlModelCfg(
    #   hidden_dims=(512, 256, 128),
    #   activation="elu",
    #   obs_normalization=True,
    #   stochastic=False,
    # ),
    
    
      # actor=RslRlModelCfg(
      # class_name="RNNModel",
      # rnn_type="lstm",
      # rnn_hidden_dim=64,
      # rnn_num_layers=1,
      # hidden_dims=(512, 256, 128),
      # activation="elu",
      # init_noise_std=0.5,
      # obs_normalization=True,
      # stochastic=True,


def ffw_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for FFW task."""
  return RslRlOnPolicyRunnerCfg(
     actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      init_noise_std=0.5,
      obs_normalization=True,
      stochastic=True,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0035,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="ffw_minimal",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3_000,
    clip_actions=True,  # Ensure clip_actions is set
  )
