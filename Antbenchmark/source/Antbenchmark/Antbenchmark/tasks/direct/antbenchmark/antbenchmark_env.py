# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .antbenchmark_env_cfg import AntbenchmarkEnvCfg
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

class AntbenchmarkEnv(LocomotionEnv):
    cfg: AntbenchmarkEnvCfg

    def __init__(self, cfg: AntbenchmarkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
    def _pre_physics_step(self, actions: torch.Tensor):
        """Convierte las acciones de la política en posiciones objetivo articulares.

        En la clase LocomotionEnv original, las acciones se transforman en torques:

            forces = action_scale * joint_gears * actions

        Para SpdrBot con servos, es más adecuado interpretar cada acción como
        un desplazamiento de posición respecto a la postura nominal.
        """

        # Guardar acciones normalizadas para observaciones y recompensas
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Acción -> posición objetivo:
        # q_target = q_default + action_scale * action
        self.joint_pos_target = (
            self.robot.data.default_joint_pos
            + self.action_scale * self.actions
        )

        # Limitar las posiciones objetivo a los límites articulares suaves
        self.joint_pos_target = torch.clamp(
            self.joint_pos_target,
            self.robot.data.soft_joint_pos_limits[:, :, 0],
            self.robot.data.soft_joint_pos_limits[:, :, 1],
        )

    def _apply_action(self):
        """Aplica control de posición a las articulaciones del robot."""

        self.robot.set_joint_position_target(
            self.joint_pos_target,
            joint_ids=self._joint_dof_idx,
        )
