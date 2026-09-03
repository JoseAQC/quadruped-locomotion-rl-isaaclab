# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import Imu
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
)

from .imuintegration_env_cfg import ImuintegrationEnvCfg

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


class ImuintegrationEnv(LocomotionEnv):
    cfg: ImuintegrationEnvCfg

    def __init__(
        self,
        cfg: ImuintegrationEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # ----------------------------------------------------------
        # Acción anterior
        # ----------------------------------------------------------

        self.previous_actions = self.actions.clone()

        # ----------------------------------------------------------
        # Estimador de velocidad basado en IMU
        # ----------------------------------------------------------

        # Velocidad estimada expresada inicialmente en WORLD.
        #
        # En el robot real esta variable será obtenida mediante
        # navegación inercial / fusión sensorial.
        self.estimated_lin_vel_w = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=torch.float32,
        )

        # ----------------------------------------------------------
        # Marcadores
        # ----------------------------------------------------------

        self.velocity_marker = None
        self.heading_marker = None

        if self.cfg.velocity_arrow_vis:

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

    # ==============================================================
    # Scene
    # ==============================================================

    def _setup_scene(self):
        """
        Crea la escena mediante LocomotionEnv y añade la IMU.
        """

        super()._setup_scene()
    # ----------------------------------------------------------
    # Terrain origins
    # ----------------------------------------------------------

        if self.cfg.terrain.terrain_type == "generator":
            self.scene._terrain = self.terrain



        self.imu = Imu(
            cfg=self.cfg.imu
        )

        self.scene.sensors["imu"] = self.imu

    # ==============================================================
    # Actions
    # ==============================================================

    def _pre_physics_step(
        self,
        actions: torch.Tensor,
    ):

        self.previous_actions = self.actions.clone()

        self.actions = (
            actions.clone()
            .clamp(-1.0, 1.0)
        )

        # ----------------------------------------------------------
        # Control de posición
        #
        # q_target = q_default + action_scale * action
        # ----------------------------------------------------------

        self.joint_pos_target = (
            self.robot.data.default_joint_pos
            + self.action_scale * self.actions
        )

        self.joint_pos_target = torch.clamp(
            self.joint_pos_target,
            self.robot.data.soft_joint_pos_limits[:, :, 0],
            self.robot.data.soft_joint_pos_limits[:, :, 1],
        )

    def _apply_action(self):

        self.robot.set_joint_position_target(
            self.joint_pos_target,
            joint_ids=self._joint_dof_idx,
        )

    # ==============================================================
    # Estimador de velocidad
    # ==============================================================

    def _update_imu_velocity_estimate(
        self,
    ) -> torch.Tensor:
        """
        Estimación simple de velocidad lineal mediante integración
        de la aceleración de la IMU.
        """

        # ----------------------------------------------------------
        # Lectura de acelerómetro
        #
        # Está expresada en el sistema de la IMU.
        # ----------------------------------------------------------

        lin_acc_b = self.imu.data.lin_acc_b

        imu_quat_w = self.imu.data.quat_w

        # ----------------------------------------------------------
        # IMU BODY -> WORLD
        # ----------------------------------------------------------

        lin_acc_w = quat_apply(
            imu_quat_w,
            lin_acc_b,
        )

        # ----------------------------------------------------------
        # Compensación de gravedad
        #
        # La IMU simulada utiliza gravity_bias para reproducir el
        # comportamiento de un acelerómetro.
        # ----------------------------------------------------------

        gravity_bias_w = torch.tensor(
            self.cfg.imu.gravity_bias,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        linear_motion_acc_w = (
            lin_acc_w
            - gravity_bias_w
        )

        # ----------------------------------------------------------
        # Integración
        #
        # v(t) = v(t-1) + a(t) * dt
        # ----------------------------------------------------------

        self.estimated_lin_vel_w += (
            linear_motion_acc_w
            * self.step_dt
        )

        # ----------------------------------------------------------
        # WORLD -> IMU/BODY
        #
        # La política recibe la velocidad en coordenadas locales.
        # ----------------------------------------------------------

        estimated_lin_vel_b = quat_apply_inverse(
            imu_quat_w,
            self.estimated_lin_vel_w,
        )

        return estimated_lin_vel_b

    # ==============================================================
    # Observations
    # ==============================================================

    def _get_observations(
        self,
    ) -> dict:

 
        # ----------------------------------------------------------
        # 1. Estado articular
        # ----------------------------------------------------------

        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        # ----------------------------------------------------------
        # 2. IMU
        # ----------------------------------------------------------

        imu_quat_w = self.imu.data.quat_w

        angular_velocity = (
            self.imu.data.ang_vel_b
        )

        linear_acceleration = (
            self.imu.data.lin_acc_b
        )

        # ----------------------------------------------------------
        # 3. Orientación estimada
        # ----------------------------------------------------------

        roll, pitch, yaw = euler_xyz_from_quat(
            imu_quat_w
        )

        theta = torch.stack(
            (
                roll,
                pitch,
                yaw,
            ),
            dim=-1,
        )
        # ----------------------------------------------------------
        # 4. Velocidad lineal estimada desde IMU
        # ----------------------------------------------------------

        estimated_lin_vel_b = (
            self._update_imu_velocity_estimate()
        )

        # ----------------------------------------------------------
        # 5. Altura del cuerpo
        #
        # Se mantiene temporalmente.
        # En una iteración posterior se eliminará.
        # ----------------------------------------------------------

        body_height = (
            self.robot.data.root_pos_w[:, 2]
            .unsqueeze(-1)
        )

        # ----------------------------------------------------------
        # 6. Dirección del objetivo
        #
        # LocomotionEnv ya calcula angle_to_target.
        # ----------------------------------------------------------

        alpha_target = (
            self.angle_to_target
            .unsqueeze(-1)
        )


        # ----------------------------------------------------------
        # Vector final
        #
        # q_j                12
        # qdot_j             12
        # theta               2
        # angular_velocity    3
        # linear_acceleration 3
        # estimated_velocity  3
        # z                   1
        # alpha_target        1
        # heading             1
        #
        # TOTAL              38
        # ----------------------------------------------------------

        obs = torch.cat(
            (
                joint_pos,
                joint_vel,
                theta,
                angular_velocity,
                linear_acceleration,
                estimated_lin_vel_b,
                body_height,
                alpha_target,
            ),
            dim=-1,
        )

        if (
            self.cfg.velocity_arrow_vis
            and self.sim.has_gui()
        ):
            self._update_velocity_marker()
            self._update_heading_marker()

        return {
            "policy": obs
        }

    # ==============================================================
    # Rewards
    # ==============================================================

    def _get_rewards(
        self,
    ) -> torch.Tensor:

        heading_reward = torch.where(
            self.heading_proj > 0.8,
            torch.ones_like(
                self.heading_proj
            )
            * self.cfg.heading_weight,
            self.cfg.heading_weight
            * self.heading_proj
            / 0.8,
        )

        up_reward = torch.where(
            self.up_proj > 0.93,
            torch.ones_like(
                self.up_proj
            )
            * self.cfg.up_weight,
            torch.zeros_like(
                self.up_proj
            ),
        )

        # ----------------------------------------------------------
        # Penalización por variación de acción
        # ----------------------------------------------------------

        actions_cost = torch.sum(
            (
                self.actions
                - self.previous_actions
            )
            ** 2,
            dim=-1,
        )

        # ----------------------------------------------------------
        # Coste energético
        # ----------------------------------------------------------

        electricity_cost = torch.sum(
            torch.abs(
                self.robot.data.applied_torque
                * self.dof_vel
            ),
            dim=-1,
        )

        # ----------------------------------------------------------
        # Límites articulares
        # ----------------------------------------------------------

        dof_at_limit_cost = torch.sum(
            self.dof_pos_scaled > 0.98,
            dim=-1,
        )

        alive_reward = (
            torch.ones_like(
                self.potentials
            )
            * self.cfg.alive_reward_scale
        )

        # ----------------------------------------------------------
        # Progreso
        # ----------------------------------------------------------

        raw_progress_reward = (
            self.potentials
            - self.prev_potentials
        )

        progress_reward = torch.where(
            raw_progress_reward >= 0.0,
            torch.where(
                self.heading_proj > 0.0,
                raw_progress_reward,
                torch.zeros_like(
                    raw_progress_reward
                ),
            ),
            raw_progress_reward,
        )

        # ----------------------------------------------------------
        # Penalización por velocidad angular
        # ----------------------------------------------------------

        angular_velocity_cost = torch.sum(
            self.angvel_loc[:, :2] ** 2,
            dim=-1,
        )   

        # ----------------------------------------------------------
        # Penalización velocidad lateral
        # ----------------------------------------------------------


        lateral_velocity_cost = self.vel_loc[:, 1] ** 2

        # ----------------------------------------------------------
        # Recompensa total
        # ----------------------------------------------------------

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
            - self.cfg.angular_velocity_scale  * angular_velocity_cost
            - self.cfg.lateral_velocity_scale * lateral_velocity_cost
        )

        total_reward = torch.where(
            self.reset_terminated,
            torch.ones_like(
                total_reward
            )
            * self.cfg.death_cost,
            total_reward,
        )


        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------

        self.extras["log"] = {
            "Rewards/Total":
                total_reward.mean(),

            "Rewards/Progress":
                progress_reward.mean(),

            "Rewards/Alive":
                alive_reward.mean(),

            "Rewards/Heading":
                heading_reward.mean(),

            "Rewards/Upright":
                up_reward.mean(),

            "Costs/Actions":
                actions_cost.mean(),

            "Costs/Energy":
                electricity_cost.mean(),

            "Costs/JointLimits":
                dof_at_limit_cost.float().mean(),

            "RewardContributions/Actions":
                (
                    -self.cfg.actions_cost_scale
                    * actions_cost
                ).mean(),

            "RewardContributions/Energy":
                (
                    -self.cfg.energy_cost_scale
                    * electricity_cost
                ).mean(),

            "RewardContributions/JointLimits":
                (
                    -dof_at_limit_cost.float()
                ).mean(),

            "RewardContributions/AngularVelocity":
                (
                    -self.cfg.angular_velocity_scale
                    * angular_velocity_cost
                ).mean(),

            "RewardContributions/LateralVelocity":
                (
                    -self.cfg.lateral_velocity_scale 
                    * lateral_velocity_cost
                ).mean(),    

            "Episode/TerminationRate":
                self.reset_terminated
                .float()
                .mean(),

            "Robot/IMUVelocityEstimate":
                torch.linalg.norm(
                    self.estimated_lin_vel_w[:, :2],
                    dim=-1,
                ).mean(),

            "Robot/MeanHeight":
                self.robot.data.root_pos_w[:, 2]
                .mean(),
        }

        return total_reward

    # ==============================================================
    # Reset
    # ==============================================================

    def _reset_idx(
        self,
        env_ids,
    ):

        super()._reset_idx(env_ids)

        # ----------------------------------------------------------
        # Reset de acciones
        # ----------------------------------------------------------

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0

        # ----------------------------------------------------------
        # Reset del estimador inercial
        #
        # Suponemos que el robot comienza el episodio detenido.
        # ----------------------------------------------------------

        self.estimated_lin_vel_w[env_ids] = 0.0

    # ==============================================================
    # Visualization
    # ==============================================================

    def _update_velocity_marker(self):

        if self.velocity_marker is None:
            return

        root_pos = self.robot.data.root_pos_w

        root_vel = self.estimated_lin_vel_w

        vel_xy = root_vel.clone()
        vel_xy[:, 2] = 0.0

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

        orientations[:, 0] = torch.cos(
            yaw / 2.0
        )

        orientations[:, 3] = torch.sin(
            yaw / 2.0
        )

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

        self.velocity_marker.visualize(
            translations=positions,
            orientations=orientations,
            scales=scales,
        )

    def _update_heading_marker(self):

        if self.heading_marker is None:
            return

        root_pos = self.robot.data.root_pos_w

        imu_quat = self.imu.data.quat_w

        forward_local = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
        )

        forward_local[:, 0] = 1.0

        forward_world = quat_apply(
            imu_quat,
            forward_local,
        )

        forward_world[:, 2] = 0.0

        norm = torch.linalg.norm(
            forward_world[:, :2],
            dim=1,
            keepdim=True,
        )

        forward_world[:, :2] /= torch.clamp(
            norm,
            min=1.0e-6,
        )

        yaw = torch.atan2(
            forward_world[:, 1],
            forward_world[:, 0],
        )

        orientations = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
        )

        orientations[:, 0] = torch.cos(
            yaw / 2.0
        )

        orientations[:, 3] = torch.sin(
            yaw / 2.0
        )

        positions = root_pos.clone()
        positions[:, 2] += 0.30

        scales = torch.ones(
            (self.num_envs, 3),
            device=self.device,
        )

        scales[:, 0] = 0.7

        self.heading_marker.visualize(
            translations=positions,
            orientations=orientations,
            scales=scales,
        )