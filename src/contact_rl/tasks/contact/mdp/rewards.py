"""Contact-explicit rewards for the locomotion task.

Implements the reach / hold / detach rewards of Omar & Khadiv (Section 3.2,
Eq. 1-3) plus foot clearance and slip regularisation (SLowRL, Omar & Khadiv
2026, Section 5), auxiliary locomotion penalties, and the goal-discovery bonus:

  r_reach     = exp(-d(p1, p_act) / sigma^2) * 1[I^con_1 = 0 and s <= delta]   (1)
  r_hold      = (1 + a_hold * exp(-d / sigma^2)) * 1[I^con_1 = I^act = 1]      (2)
  r_detach    = 1[I^con_1 = I^act = 0 and s > delta]                            (3)
  r_clearance = exp(-(max(h_target - z_foot, 0))^2 / 0.0025) * 1[detach]
  r_slip      = ||v_{xy,foot}||^2 * 1[hold]

Per-step phase masks and foot-to-goal distances are cached on the command term
keyed by ``env.common_step_counter`` so all 5 contact reward terms share a
single GPU calculation per environment step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from contact_rl.tasks.contact.mdp.contact_command import ContactGoalCommand
from contact_rl.tasks.contact.mdp.planning import phase_masks, proximity_kernel

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _term(env: ManagerBasedRlEnv, command_name: str) -> ContactGoalCommand:
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, ContactGoalCommand):
    raise TypeError(f"Command '{command_name}' is not a ContactGoalCommand.")
  return term


def _phase_inputs(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
):
  """Return (term, dist [N,4], reach, hold, detach masks [N,4]), cached per step."""
  term = _term(env, command_name)
  step_id = getattr(env, "common_step_counter", None)
  cache = getattr(term, "_phase_cache", None)
  if step_id is not None and cache is not None and cache[0] == step_id:
    return term, cache[1], cache[2], cache[3], cache[4]

  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.found is None:
    raise RuntimeError(f"Sensor '{sensor_name}' must record the 'found' field.")
  goal = term.goal_pos_w[:, :, 0, :]  # current target, [N, 4, 3]
  foot = asset.data.site_pos_w[:, term.foot_site_ids, :]  # [N, 4, 3]
  dist = torch.norm(goal - foot, dim=-1)  # [N, 4]
  i_act = (sensor.data.found > 0).float()
  reach, hold, detach = phase_masks(
    term.current_contact_goal, i_act, term.time_remaining, term.cfg.delta
  )
  if step_id is not None:
    term._phase_cache = (step_id, dist, reach, hold, detach)
  return term, dist, reach, hold, detach


def contact_reach(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  sigma_sq: float = 0.04,
  kernel: str = "l2",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Eq. 1, summed over feet: guide a swinging foot to its upcoming contact."""
  _, dist, reach, _, _ = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  return (proximity_kernel(dist, sigma_sq, kernel) * reach.float()).sum(dim=1)


def contact_hold(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  hold_alpha: float = 1.0,
  sigma_sq: float = 0.04,
  kernel: str = "l2",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Eq. 2, summed over feet: keep commanded contact, bonus at the right spot."""
  _, dist, _, hold, _ = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  k = proximity_kernel(dist, sigma_sq, kernel)
  return ((1.0 + hold_alpha * k) * hold.float()).sum(dim=1)


def contact_detach(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Eq. 3, summed over feet: no contact during the detach phase."""
  _, _, _, _, detach = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  return detach.float().sum(dim=1)


def contact_foot_clearance(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  target_height: float = 0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward lifting swinging feet to ``target_height`` during the detach phase.

  Prevents feet from skimming millimetres above the ground during swing, which
  causes premature ground strikes (high Hamming error) and poor landing accuracy.
  """
  term, _, _, _, detach = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, term.foot_site_ids, 2]  # [N, 4]
  deficit_sq = torch.square(torch.clamp(target_height - foot_z, min=0.0))
  return (torch.exp(-deficit_sq / 0.0025) * detach.float()).sum(dim=1)


def contact_foot_slip(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize horizontal foot sliding velocity while in the hold phase."""
  term, _, _, hold, _ = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_xy = asset.data.site_lin_vel_w[:, term.foot_site_ids, :2]  # [N, 4, 2]
  vel_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)  # [N, 4]
  return (vel_sq * hold.float()).sum(dim=1)


def goal_discovery_bonus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Bonus for discovering / completing goals (at most once per contact switch)."""
  return _term(env, command_name).just_discovered


##
# Auxiliary locomotion penalties named by the paper that are not already in
# ``mjlab.envs.mdp`` (joint vel / acc / torques / action rate come from there).
##


def base_angular_velocity_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base angular velocities (L2, body frame)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)


def joint_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint deviations from the default pose (L2)."""
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  jnt = asset_cfg.joint_ids
  return torch.sum(torch.square(asset.data.joint_pos[:, jnt] - default[:, jnt]), dim=1)


##
# Posture regularisation (anti-kneeling, see module docstring).
##


def undesired_contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Number of non-foot collision geoms (base, hips, thighs, calves) touching
  the ground. ``sensor_name`` must be a ContactSensor recording ``found``."""
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.found is None:
    raise RuntimeError(f"Sensor '{sensor_name}' must record the 'found' field.")
  return (sensor.data.found > 0).float().sum(dim=1)


def base_height_below(
  env: ManagerBasedRlEnv,
  target: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """One-sided, linear: ``max(target - z_base, 0)`` (flat terrain)."""
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.root_link_pos_w[:, 2]
  return torch.clamp(target - z, min=0.0)


def base_tilt_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize a constant roll / pitch of the base (xy of projected gravity)."""
  asset: Entity = env.scene[asset_cfg.name]
  g = asset.data.projected_gravity_b
  return torch.sum(torch.square(g[:, :2]), dim=1)


def knee_height_below(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_height: float = 0.08,
) -> torch.Tensor:
  """Dense anti-kneeling penalty (flat terrain), summed over legs."""
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  return (torch.clamp(min_height - z, min=0.0) / min_height).sum(dim=1)
