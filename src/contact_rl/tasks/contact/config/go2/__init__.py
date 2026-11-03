"""Registration of the Unitree Go2 contact-explicit tasks.

Both tasks now run the optimised setup: 87-dim actor observation with IMU
signals and running observation normalisation, fast PPO schedule
(max_iter 5000, KL 0.015). They differ only in the run folder:

* ``Mjlab-Contact-Flat-Unitree-Go2``          -- runs/go2_contact (default).
* ``Mjlab-Contact-Flat-Unitree-Go2-Improved`` -- runs/go2_contact_improved.
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_go2_flat_env_cfg
from .rl_cfg import unitree_go2_ppo_runner_cfg
from .runner import ContactOnPolicyRunner

register_mjlab_task(
  task_id="Mjlab-Contact-Flat-Unitree-Go2",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=ContactOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Contact-Flat-Unitree-Go2-Improved",
  env_cfg=unitree_go2_flat_env_cfg(improved=True),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True, improved=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(improved=True),
  runner_cls=ContactOnPolicyRunner,
)
