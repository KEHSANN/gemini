"""RL configuration for the Unitree Go2 contact-explicit locomotion task.

Optimised for faster convergence and higher tracking precision based on Omar &
Khadiv (L4DC 2026 & SLowRL 2026) and rsl-rl 5.0.1 best practices:
  * Running observation normalisation enabled on both actor and critic so
    heterogeneous inputs (metres, rad/s, binary contact indicators) are scaled
    uniformly from iteration 0.
  * Faster linear entropy decay (0.008 -> 0.001 over 2500 iterations) so the
    policy transitions from broad gait exploration to centimetre-level foot
    placement without needing 5000+ iterations.
  * Per-minibatch advantage normalisation enabled and ``desired_kl = 0.015`` so
    the adaptive LR schedule maintains fast step sizes during early training.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# Accelerated entropy-decay schedule (read by ContactOnPolicyRunner).
ENTROPY_START = 0.008
ENTROPY_END = 0.001
ENTROPY_DECAY_ITERS = 2500


@dataclass
class RslRlRnnModelCfg(RslRlModelCfg):
  """``RslRlModelCfg`` extended with rsl-rl 5.x ``RNNModel`` fields."""

  class_name: str = "RNNModel"
  rnn_type: str = "gru"  # "gru" (paper) or "lstm".
  rnn_hidden_dim: int = 256
  rnn_num_layers: int = 1


def unitree_go2_ppo_runner_cfg(improved: bool = False) -> RslRlOnPolicyRunnerCfg:
  """PPO runner cfg with running observation normalisation and fast schedule.

  ``improved=False`` keeps ``experiment_name="go2_contact"`` so ``runs/go2_contact``
  is used by default (matching all VPS scripts), while enabling observation
  normalisation and fast PPO convergence on both tasks.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlRnnModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      rnn_type="gru",
      rnn_hidden_dim=256,
      rnn_num_layers=1,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlRnnModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      rnn_type="gru",
      rnn_hidden_dim=256,
      rnn_num_layers=1,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=ENTROPY_START,  # Decayed by ContactOnPolicyRunner.
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.015,
      max_grad_norm=1.0,
      normalize_advantage_per_mini_batch=True,
    ),
    seed=42,
    experiment_name="go2_contact_improved" if improved else "go2_contact",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=5_000,
    logger="tensorboard",
    wandb_project="contact-rl",
    upload_model=False,
  )
