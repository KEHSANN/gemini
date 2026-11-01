"""Unitree Go2 contact-explicit locomotion environment configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from contact_rl.robots.go2 import get_go2_robot_cfg
from contact_rl.tasks.contact.contact_env_cfg import (
  FEET_SENSOR_NAME,
  FOOT_ORDER,
  ILLEGAL_CONTACT_SENSOR_NAME,
  make_contact_env_cfg,
)

# Foot collision geoms, ordered to match ``FOOT_ORDER`` (FL, FR, RL, RR).
_FOOT_GEOMS = tuple(f"{name}_foot_collision" for name in FOOT_ORDER)

# Every non-foot collision geom of go2.xml. Ground contact on any of them
# (knees / calves, thighs, hips, body) is penalised by ``illegal_contact``.
_ILLEGAL_GEOMS = ("base1_collision", "base2_collision", "base3_collision") + tuple(
  f"{leg}_{part}_collision"
  for leg in FOOT_ORDER
  for part in ("hip", "thigh", "calf1", "calf2")
)

# Paper, Section 4 "Training": 8192 parallel environments. Override with
# ``--env.scene.num-envs`` on smaller GPUs (``contact-bench`` helps pick one).
PAPER_NUM_ENVS = 8192

# Uniform 0.35 rad action scale as reported in Omar & Khadiv (SLowRL 2026, §5).
GO2_PAPER_ACTION_SCALE = 0.35


def unitree_go2_flat_env_cfg(
  play: bool = False, improved: bool = True
) -> ManagerBasedRlEnvCfg:
  """Go2 flat-terrain contact-explicit configuration.

  Args:
    play: Evaluation / visualisation variant (no pushes, no obs noise).
    improved: Enables IMU signals in the actor (now True by default, matching
      Omar & Khadiv's R^82 observation in SLowRL 2026).
  """
  cfg = make_contact_env_cfg(actor_imu=improved)

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}
  cfg.scene.num_envs = PAPER_NUM_ENVS

  feet_ground_cfg = ContactSensorCfg(
    name=FEET_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=_FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  illegal_ground_cfg = ContactSensorCfg(
    name=ILLEGAL_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=_ILLEGAL_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, illegal_ground_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_PAPER_ACTION_SCALE

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.viewer.body_name = "base_link"
  cfg.commands["contact"].debug_vis = play

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.scene.num_envs = 64
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.width = 640
    cfg.viewer.height = 480

  return cfg
