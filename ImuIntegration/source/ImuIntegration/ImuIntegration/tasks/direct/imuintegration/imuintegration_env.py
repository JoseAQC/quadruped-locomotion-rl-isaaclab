# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply
from .imuintegration_env_cfg import ImuintegrationEnvCfg
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

class ImuintegrationEnv(LocomotionEnv):
    cfg: ImuintegrationEnvCfg

    def __init__(self, cfg: ImuintegrationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        
        self.previous_actions = self.actions.clone()
        self.velocity_marker = None
        self.heading_marker = None

        if self.cfg.velocity_arrow_vis:

            # Flecha de velocidad
            velocity_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/velocity_direction",
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=(
                            f"{ISAAC_NUCLEUS_DIR}"
                            "/Props/UIElements/arrow_x.usd"
                        ),
                        scale=(0.5, 0.15, 0.15),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0)
                        ),
                    ),
                },
            )

            self.velocity_marker = VisualizationMarkers(
                velocity_marker_cfg
            )

            # Flecha de orientación del robot
            heading_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/heading_direction",
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=(
                            f"{ISAAC_NUCLEUS_DIR}"
                            "/Props/UIElements/arrow_x.usd"
                        ),
                        scale=(0.5, 0.15, 0.15),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.0, 1.0)
                        ),
                    ),
                },
            )

            self.heading_marker = VisualizationMarkers(
                heading_marker_cfg
            )

    def _pre_physics_step(self, actions: torch.Tensor):


        # Guardar acciones normalizadas para observaciones y recompensas
        self.previous_actions = self.actions.clone()
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

    def _update_velocity_marker(self):
        """Actualiza la flecha que representa la velocidad horizontal."""

        if self.velocity_marker is None:
            return
        root_pos = self.robot.data.root_pos_w
        root_vel = self.robot.data.root_lin_vel_w

        vel_xy = root_vel.clone()
        vel_xy[:, 2] = 0.0

        # Módulo de la velocidad horizontal
        speed = torch.linalg.norm(
            vel_xy[:, :2],
            dim=1,
        )

        yaw = torch.atan2(
            vel_xy[:, 1],
            vel_xy[:, 0],
        )

        orientations = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
        )

        orientations[:, 0] = torch.cos(yaw / 2.0)
        orientations[:, 3] = torch.sin(yaw / 2.0)

        positions = root_pos.clone()


        positions[:, 2] += 0.20

        scales = torch.ones(
            (self.num_envs, 3),
            device=self.device,
        )

        scales[:, 0] = torch.clamp(
            speed,
            min=0.10,
            max=1.5,
        )

        # Grosor de la flecha.
        scales[:, 1] = 1.0
        scales[:, 2] = 1.0

        self.velocity_marker.visualize(
            translations=positions,
            orientations=orientations,
            scales=scales,
        )
    def _update_heading_marker(self):
        """Dibuja la dirección hacia la que apunta el robot."""

        if self.heading_marker is None:
            return
        
        root_pos = self.robot.data.root_pos_w

        root_quat = self.robot.data.root_quat_w

        forward_local = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
        )

        forward_local[:, 0] = 1.0

        forward_world = quat_apply(
            root_quat,
            forward_local,
        )

        forward_world[:, 2] = 0.0

        # Normalización
        norm = torch.linalg.norm(
            forward_world[:, :2],
            dim=1,
            keepdim=True,
        )

        forward_world[:, :2] = (
            forward_world[:, :2]
            / torch.clamp(norm, min=1.0e-6)
        )

        yaw = torch.atan2(
            forward_world[:, 1],
            forward_world[:, 0],
        )

        orientations = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
        )

        orientations[:, 0] = torch.cos(yaw / 2.0)
        orientations[:, 3] = torch.sin(yaw / 2.0)

        positions = root_pos.clone()

        positions[:, 2] += 0.30

        scales = torch.ones(
            (self.num_envs, 3),
            device=self.device,
        )

        scales[:, 0] = 0.7
        scales[:, 1] = 1.0
        scales[:, 2] = 1.0

        self.heading_marker.visualize(
            translations=positions,
            orientations=orientations,
            scales=scales,
        )
    def _get_observations(self):
        """Obtiene las observaciones del entorno de locomoción."""

        obs = super()._get_observations()
        if (
            self.cfg.velocity_arrow_vis
            and self.sim.has_gui()
        ):
            self._update_velocity_marker()
            self._update_heading_marker()

        return obs
    
    def _get_rewards(self) -> torch.Tensor:


        heading_reward = torch.where(
            self.heading_proj > 0.8,
            torch.ones_like(self.heading_proj) * self.cfg.heading_weight,
            self.cfg.heading_weight * self.heading_proj / 0.8,
        )

        up_reward = torch.where(
            self.up_proj > 0.93,
            torch.ones_like(self.up_proj) * self.cfg.up_weight,
            torch.zeros_like(self.up_proj),
        )

        actions_cost = torch.sum(
            (
                self.actions
                - self.previous_actions
            ) ** 2,
            dim=-1,
        )

        electricity_cost = torch.sum(
            torch.abs(
                self.robot.data.applied_torque
                * self.dof_vel
                ),
                dim=-1,
        )

        dof_at_limit_cost = torch.sum(
            self.dof_pos_scaled > 0.98,
            dim=-1,
        )

        alive_reward = (
            torch.ones_like(self.potentials)
            * self.cfg.alive_reward_scale
        )

        raw_progress_reward = (
            self.potentials
            - self.prev_potentials
        )

        progress_reward = torch.where(
            raw_progress_reward >= 0.0,
            torch.where(
                self.heading_proj > 0.0,
                raw_progress_reward,
                torch.zeros_like(raw_progress_reward),
            ),
            raw_progress_reward,
        )
        total_reward = (
            progress_reward
            + alive_reward
            + up_reward
            + heading_reward
            - self.cfg.actions_cost_scale
            * actions_cost
            - self.cfg.energy_cost_scale
            * electricity_cost
            - dof_at_limit_cost
        )

        total_reward = torch.where(
            self.reset_terminated,
            torch.ones_like(total_reward)
            * self.cfg.death_cost,
            total_reward,
        )

        self.extras["log"] = {
            "Rewards/Total": total_reward.mean(),
            "Rewards/Progress": progress_reward.mean(),
            "Rewards/Alive": alive_reward.mean(),
            "Rewards/Heading": heading_reward.mean(),
            "Rewards/Upright": up_reward.mean(),

            "Costs/Actions": actions_cost.mean(),
            "Costs/Energy": electricity_cost.mean(),
            "Costs/JointLimits":
                dof_at_limit_cost.float().mean(),

            "RewardContributions/Actions":
                (-self.cfg.actions_cost_scale * actions_cost).mean(),

            "RewardContributions/Energy":
                (-self.cfg.energy_cost_scale * electricity_cost).mean(),

            "RewardContributions/JointLimits":
                (-dof_at_limit_cost.float()).mean(),

            "Episode/TerminationRate":
                self.reset_terminated.float().mean(),

            # ----------------------------------------------------------
            # Variables físicas
            # ----------------------------------------------------------

            "Robot/MeanSpeedXY":
                torch.linalg.norm(
                    self.robot.data.root_lin_vel_w[:, :2],
                    dim=-1,
                ).mean(),

            "Robot/MeanHeight":
                self.robot.data.root_pos_w[:, 2].mean(),
        }

        return total_reward
    
    def _reset_idx(self, env_ids):
        """Reinicia los entornos y las acciones anteriores."""

        super()._reset_idx(env_ids)

        # Evita una penalización C_delta_a artificial inmediatamente
        # después de reiniciar un entorno.
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0