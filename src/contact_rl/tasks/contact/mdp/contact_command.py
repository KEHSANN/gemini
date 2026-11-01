"""Contact-goal command for the contact-explicit locomotion task.

Implements the contact goals of Omar & Khadiv, *Learning to Act Through Contact:
A Unified View of Multi-Task Robot Learning* (L4DC 2026, arXiv:2510.03599v2),
Section 3, for a quadruped end-effector set (the four feet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor

from contact_rl.tasks.contact.mdp.planning import (
  FOOT_ORDER,
  GAITS,
  GO2_HIP_X,
  NUM_FEET,
  NUM_SWITCHES,
  GaitPlanner,
  yaw_from_quat_wxyz,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = ["GAITS", "ContactGoalCommand", "ContactGoalCommandCfg"]


class ContactGoalCommand(CommandTerm):
  """Generates per-foot contact goals (locations + indicators + duration)."""

  cfg: ContactGoalCommandCfg

  def __init__(self, cfg: ContactGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    if self.robot.data.is_fixed_base:
      raise ValueError("ContactGoalCommand requires a floating-base robot.")
    if not 0.0 < cfg.delta < cfg.resampling_time_range[0]:
      raise ValueError(
        f"delta={cfg.delta} must lie in (0, min command duration "
        f"{cfg.resampling_time_range[0]})."
      )

    site_ids, site_names = self.robot.find_sites(
      list(cfg.foot_site_names), preserve_order=True
    )
    if tuple(site_names) != tuple(cfg.foot_site_names):
      raise ValueError(
        f"Resolved foot sites {tuple(site_names)} do not match the requested "
        f"order {tuple(cfg.foot_site_names)}."
      )
    self._foot_site_ids = site_ids

    self._sensor: ContactSensor | None = (
      env.scene[cfg.sensor_name] if cfg.sensor_name is not None else None
    )

    N, dev = self.num_envs, self.device
    self.planner = GaitPlanner(cfg, N, dev, hip_x=cfg.hip_x)

    # Goal buffers (world frame).
    self._goal_pos_w = torch.zeros(N, NUM_FEET, NUM_SWITCHES, 3, device=dev)
    self._goal_pos_w[..., 2] = cfg.foot_target_height
    self._goal_contact = torch.zeros(N, NUM_FEET, NUM_SWITCHES, device=dev)
    self._phase_cache = None

    # Goal-discovery bookkeeping.
    self.just_discovered = torch.zeros(N, device=dev)
    self._discovered_this_switch = torch.zeros(N, device=dev, dtype=torch.bool)
    self.goals_discovered = torch.zeros(N, device=dev)
    self._dwell_time_left_max = (1.0 - cfg.goal_dwell_frac) * cfg.resampling_time_range[1]

    # Flat command vector: per foot [p1(3), p2(3)], then [I1, I2], then s.
    self._command = torch.zeros(N, NUM_FEET * 8 + 1, device=dev)

    # Episode-averaged metrics (paper Fig. 6 protocol + legacy tracking error).
    self._m_steps = torch.zeros(N, device=dev)
    self._m_track = torch.zeros(N, device=dev)
    self._m_hamming = torch.zeros(N, device=dev)
    self._m_loc_sum = torch.zeros(N, device=dev)
    self._m_loc_cnt = torch.zeros(N, device=dev)
    self._skip_metrics = torch.ones(N, device=dev, dtype=torch.bool)
    for key in ("tracking_error", "goals_discovered"):
      self.metrics[key] = torch.zeros(N, device=dev)
    if self._sensor is not None:
      self.metrics["contact_plan_hamming"] = torch.zeros(N, device=dev)
      self.metrics["contact_location_error"] = torch.zeros(N, device=dev)

  # ------------------------------------------------------------ helpers

  def _root_pose_from_qpos(self, env_ids: torch.Tensor):
    """Root xy + yaw read from ``qpos`` (valid right after a state write)."""
    data = self.robot.data
    q = data.data.qpos[env_ids[:, None], data.indexing.free_joint_q_adr]  # [n, 7]
    return q[:, 0:2], yaw_from_quat_wxyz(q[:, 3:7])

  def _sync_goals(self, env_ids: torch.Tensor) -> None:
    pl = self.planner
    self._goal_pos_w[env_ids, :, 0, :2] = pl.foot_xy[env_ids]
    self._goal_pos_w[env_ids, :, 1, :2] = pl.next_xy[env_ids]
    self._goal_contact[env_ids, :, 0] = pl.cur_contact[env_ids]
    self._goal_contact[env_ids, :, 1] = pl.next_contact[env_ids]
    self._discovered_this_switch[env_ids] = False

  def foot_pos_w(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self._foot_site_ids, :]  # [N, 4, 3]

  def actual_contact(self) -> torch.Tensor:
    if self._sensor is None or self._sensor.data.found is None:
      raise RuntimeError("ContactGoalCommand has no contact sensor configured.")
    return (self._sensor.data.found > 0).float()  # [N, 4]

  # -------------------------------------------------------- CommandTerm

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def num_feet(self) -> int:
    return NUM_FEET

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Advance the contact plan by one switch (or re-anchor on reset)."""
    if len(env_ids) == 0:
      return
    first = self.command_counter[env_ids] == 0  # (re)initialisation
    init_ids = env_ids[first]
    cont_ids = env_ids[~first]
    if len(init_ids) > 0:
      base_xy, base_yaw = self._root_pose_from_qpos(init_ids)
      self.planner.reset(init_ids, base_xy, base_yaw)
      self.goals_discovered[init_ids] = 0.0
      self._m_steps[init_ids] = 0.0
      self._m_track[init_ids] = 0.0
      self._m_hamming[init_ids] = 0.0
      self._m_loc_sum[init_ids] = 0.0
      self._m_loc_cnt[init_ids] = 0.0
      self._skip_metrics[init_ids] = True
    if len(cont_ids) > 0:
      self.planner.advance(cont_ids)
    self._sync_goals(env_ids)

  def _update_command(self) -> None:
    """Per-step update: build the observation vector and detect goal completion."""
    N = self.num_envs
    goal_b = self._world_to_base(self._goal_pos_w)  # [N, 4, 2, 3]
    n_pos = NUM_FEET * NUM_SWITCHES * 3
    n_con = NUM_FEET * NUM_SWITCHES
    self._command[:, :n_pos] = goal_b.reshape(N, n_pos)
    self._command[:, n_pos : n_pos + n_con] = self._goal_contact.reshape(N, n_con)
    self._command[:, -1] = torch.clamp(self.time_left, min=0.0)

    # Goal discovery / completion: require both base proximity to the foothold
    # centroid AND accurate stance foot placement (< goal_proximity_threshold)
    # so the bonus is earned by genuine contact tracking rather than for free.
    base_xy = self.robot.data.root_link_pos_w[:, :2]
    centroid = self._goal_pos_w[:, :, 0, :2].mean(dim=1)
    dist_base = torch.norm(base_xy - centroid, dim=-1)

    foot_xy = self.foot_pos_w()[:, :, :2]
    foot_err_xy = torch.norm(self._goal_pos_w[:, :, 0, :2] - foot_xy, dim=-1)
    stance_mask = (self._goal_contact[:, :, 0] > 0.5).float()
    stance_cnt = torch.clamp(stance_mask.sum(dim=1), min=1.0)
    stance_foot_err = (foot_err_xy * stance_mask).sum(dim=1) / stance_cnt

    reached = (
      (dist_base < self.cfg.goal_proximity_threshold)
      & (stance_foot_err < self.cfg.goal_proximity_threshold)
      & (~self._discovered_this_switch)
      & (self.time_left < self._dwell_time_left_max)
    )
    self.just_discovered.copy_(reached)
    self._discovered_this_switch |= reached
    self.goals_discovered += reached.float()
    if self.cfg.discovery_advances_goals:  # Legacy behaviour (ablation only).
      self.time_left.masked_fill_(reached, 0.0)

  def _update_metrics(self) -> None:
    # Runs at the start of compute(), i.e. on the goals the policy just acted on.
    valid = (~self._skip_metrics).float()
    self._skip_metrics.fill_(False)
    foot = self.foot_pos_w()
    dist = torch.norm(self._goal_pos_w[:, :, 0, :] - foot, dim=-1)  # [N, 4]
    self._m_steps += valid
    self._m_track += valid * dist.mean(dim=1)
    denom = torch.clamp(self._m_steps, min=1.0)
    self.metrics["tracking_error"] = self._m_track / denom
    self.metrics["goals_discovered"] = self.goals_discovered
    if self._sensor is not None:
      i_con = self._goal_contact[:, :, 0]
      i_act = self.actual_contact()
      self._m_hamming += valid * (torch.abs(i_con - i_act)).sum(dim=1)
      in_contact = (i_con > 0.5) & (i_act > 0.5)
      self._m_loc_sum += valid * (dist * in_contact).sum(dim=1)
      self._m_loc_cnt += valid * in_contact.float().sum(dim=1)
      self.metrics["contact_plan_hamming"] = self._m_hamming / denom
      self.metrics["contact_location_error"] = self._m_loc_sum / torch.clamp(
        self._m_loc_cnt, min=1.0
      )

  # ---------------------------------------------------------- debug vis

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    goal = self._goal_pos_w
    con = self._goal_contact
    feet = self.robot.data.site_pos_w
    radius = 0.03
    for i in visualizer.get_env_indices(self.num_envs):
      for f in range(NUM_FEET):
        stance = bool(con[i, f, 0] > 0.5)
        c1 = (0.1, 0.9, 0.2, 0.85) if stance else (0.95, 0.25, 0.2, 0.85)
        visualizer.add_sphere(goal[i, f, 0], radius, c1, label=f"{FOOT_ORDER[f]}_p1")
        stance_next = bool(con[i, f, 1] > 0.5)
        c2 = (0.1, 0.9, 0.2, 0.3) if stance_next else (0.95, 0.25, 0.2, 0.3)
        visualizer.add_sphere(goal[i, f, 1], 0.6 * radius, c2, label=f"{FOOT_ORDER[f]}_p2")
        if not stance:
          visualizer.add_arrow(
            feet[i, self._foot_site_ids[f]],
            goal[i, f, 0],
            (1.0, 0.6, 0.0, 0.9),
            width=0.008,
            label=f"{FOOT_ORDER[f]}_reach",
          )

  # -------------------------------------------------------------- utils

  def _world_to_base(self, pos_w: torch.Tensor) -> torch.Tensor:
    """Yaw-only world->base transform of positions ``[N, ..., 3]``."""
    base_pos = self.robot.data.root_link_pos_w  # [N, 3]
    yaw = self.robot.data.heading_w  # [N]
    extra = pos_w.dim() - 2
    shape = (self.num_envs,) + (1,) * extra
    rel = pos_w - base_pos.view(shape + (3,))
    c, s = torch.cos(yaw).view(shape), torch.sin(yaw).view(shape)
    rx, ry = rel[..., 0], rel[..., 1]
    return torch.stack([c * rx + s * ry, -s * rx + c * ry, rel[..., 2]], dim=-1)

  # ------------------------------------------------------ goal accessors

  @property
  def goal_pos_w(self) -> torch.Tensor:
    return self._goal_pos_w

  @property
  def current_contact_goal(self) -> torch.Tensor:
    return self._goal_contact[:, :, 0]

  @property
  def next_contact_goal(self) -> torch.Tensor:
    return self._goal_contact[:, :, 1]

  @property
  def time_remaining(self) -> torch.Tensor:
    return self.time_left

  @property
  def foot_site_ids(self):
    return self._foot_site_ids


@dataclass(kw_only=True)
class ContactGoalCommandCfg(CommandTermCfg):
  """Configuration for :class:`ContactGoalCommand`."""

  entity_name: str = "robot"
  foot_site_names: tuple[str, ...] = FOOT_ORDER
  sensor_name: str | None = None
  hip_x: tuple[float, ...] = GO2_HIP_X

  # Command duration S ~ U[0.34, 0.36] s (Section 4).
  resampling_time_range: tuple[float, float] = (0.34, 0.36)

  # Sampled once per environment at construction (Section 4).
  stride_length_range: tuple[float, float] = (0.0, 0.3)
  stance_width_range: tuple[float, float] = (0.1, 0.3)
  leg_offset_range: tuple[float, float] = (-0.15, 0.15)
  heading_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
  yaw_rate_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
  rel_yaw_rate_envs: float = 0.5

  gaits: tuple[str, ...] | None = None

  delta: float = 0.18
  foot_target_height: float = 0.02

  # Tightened from 0.15 m to 0.06 m and gated on stance foot error so the bonus
  # rewards genuine contact goal completion rather than firing every switch.
  goal_proximity_threshold: float = 0.06

  goal_dwell_frac: float = 0.5
  discovery_advances_goals: bool = False
  debug_vis: bool = True

  def build(self, env: ManagerBasedRlEnv) -> ContactGoalCommand:
    return ContactGoalCommand(self, env)
