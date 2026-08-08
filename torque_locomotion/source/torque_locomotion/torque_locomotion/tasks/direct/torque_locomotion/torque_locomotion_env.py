# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv
from .torque_locomotion_env_cfg import TorqueLocomotionEnvCfg

class TorqueLocomotionEnv(LocomotionEnv):
    cfg: TorqueLocomotionEnvCfg

    def __init__(self, cfg: TorqueLocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)