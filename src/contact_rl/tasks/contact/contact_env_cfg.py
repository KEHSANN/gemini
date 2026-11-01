"""Contact-explicit locomotion task configuration.

``make_contact_env_cfg`` builds the base config for the contact-explicit
multi-gait locomotion task of Omar & Khadiv (arXiv:2510.03599v2) and incorporates
the Go2 implementation details disclosed in Omar & Khadiv (SLowRL, 2026):
  * IMU observations (``base_ang_vel``, ``projected_gravity``) enabled in the
    actor by default (``actor_imu=True``) for lateral stability (pace/jump) and
    accurate yaw-rate tracking.
  * Explicit ``foot_clearance`` reward during the detach phase and ``foot_slip``
    penalty during the hold phase.
  * Sharper spatial kernel (``CONTACT_SIGMA_SQ = 0.04``) and stronger reach/hold
    weights (2.0) to drive foot placement error down to the 3-5 cm range.
  * ``joint_deviation`` restricted to thigh/calf joints with weight -0.01 so
    hip abduction joints can freely track wide/narrow stance widths.
  * Streamlined MuJoCo-Warp solver settings for flat terrain to maximize FPS.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from contact_rl.tasks.contact import mdp

FOOT_ORDER: tuple[str, ...] = ("FL", "FR", "RL", "RR")

COMMAND_NAME = "contact"
FEET_SENSOR_NAME = "feet_ground_contact"
ILLEGAL_CONTACT_SENSOR_NAME = "illegal_contact"

# Sharper Laplacian kernel exp(-d / sigma^2) with sigma^2 = 0.04 m (4 cm e-fold
# length scale) so the policy receives a strong gradient all the way down to
# 2-4 cm foot placement accuracy.
CONTACT_SIGMA_SQ = 0.04
CONTACT_KERNEL = "l2"

MIN_BASE_HEIGHT = 0.25
MIN_KNEE_HEIGHT = 0.08


def make_contact_env_cfg(actor_imu: bool = True) -> ManagerBasedRlEnvCfg:
  """Create the contact-explicit locomotion task configuration."""

  ##
  # Observations.
  ##

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "contact_goal": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": COMMAND_NAME},
    ),
    "feet_to_goal": ObservationTermCfg(
      func=mdp.feet_to_goal_distance,
      params={"command_name": COMMAND_NAME},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
  }
  if actor_imu:
    actor_terms["base_ang_vel"] = ObservationTermCfg(
      func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    )
    actor_terms["projected_gravity"] = ObservationTermCfg(
      func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    )

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact_state,
      params={"sensor_name": FEET_SENSOR_NAME},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions.
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.35,  # Go2 action scale from Omar & Khadiv (SLowRL 2026).
      use_default_offset=True,
    )
  }

  ##
  # Commands (the contact goal).
  ##

  commands: dict[str, CommandTermCfg] = {
    COMMAND_NAME: mdp.ContactGoalCommandCfg(
      entity_name="robot",
      foot_site_names=FOOT_ORDER,
      sensor_name=FEET_SENSOR_NAME,
      resampling_time_range=(0.34, 0.36),
      debug_vis=True,
    ),
  }

  ##
  # Events (domain randomisation).
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.02),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
    "check_foot_ordering": EventTermCfg(
      mode="startup",
      func=mdp.check_foot_ordering,
      params={
        "command_name": COMMAND_NAME,
        "sensor_name": FEET_SENSOR_NAME,
        "foot_order": FOOT_ORDER,
      },
    ),
  }

  ##
  # Rewards.
  ##

  contact_params = {"command_name": COMMAND_NAME, "sensor_name": FEET_SENSOR_NAME}
  rewards = {
    # Contact-explicit rewards (Section 3.2, Eq. 1-3 + SLowRL Section 5).
    "reach": RewardTermCfg(
      func=mdp.contact_reach,
      weight=2.0,
      params={**contact_params, "sigma_sq": CONTACT_SIGMA_SQ, "kernel": CONTACT_KERNEL},
    ),
    "hold": RewardTermCfg(
      func=mdp.contact_hold,
      weight=2.0,
      params={
        **contact_params,
        "hold_alpha": 1.0,
        "sigma_sq": CONTACT_SIGMA_SQ,
        "kernel": CONTACT_KERNEL,
      },
    ),
    "detach": RewardTermCfg(
      func=mdp.contact_detach,
      weight=0.5,
      params=dict(contact_params),
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.contact_foot_clearance,
      weight=0.5,
      params={**contact_params, "target_height": 0.08},
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.contact_foot_slip,
      weight=-0.1,
      params=dict(contact_params),
    ),
    "goal_discovery": RewardTermCfg(
      func=mdp.goal_discovery_bonus,
      weight=10.0,
      params={"command_name": COMMAND_NAME},
    ),
    # Auxiliary locomotion penalties.
    "base_ang_vel": RewardTermCfg(func=mdp.base_angular_velocity_l2, weight=-0.05),
    "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1.0e-3),
    "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2.0e-4),
    # Restrict joint_deviation to thigh and calf joints with lower weight so hip
    # abduction joints can freely realize wide/narrow stance widths.
    "joint_deviation": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-0.01,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_thigh_joint", ".*_calf_joint"))},
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    # Posture regularisation (anti-kneeling).
    "illegal_contact": RewardTermCfg(
      func=mdp.undesired_contact_count,
      weight=-4.0,
      params={"sensor_name": ILLEGAL_CONTACT_SENSOR_NAME},
    ),
    "knee_height": RewardTermCfg(
      func=mdp.knee_height_below,
      weight=-5.0,
      params={
        "min_height": MIN_KNEE_HEIGHT,
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*_calf",)),
      },
    ),
    "base_height_below": RewardTermCfg(
      func=mdp.base_height_below,
      weight=-10.0,
      params={"target": MIN_BASE_HEIGHT},
    ),
    "base_tilt": RewardTermCfg(func=mdp.base_tilt_l2, weight=-1.0),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(80.0)},
    ),
  }

  ##
  # Assemble.
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4096,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=32,
      njmax=120,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=6,
        ls_iterations=10,
        ccd_iterations=20,
      ),
      contact_sensor_maxmatch=32,
    ),
    decimation=4,  # 50 Hz control.
    episode_length_s=20.0,
  )
