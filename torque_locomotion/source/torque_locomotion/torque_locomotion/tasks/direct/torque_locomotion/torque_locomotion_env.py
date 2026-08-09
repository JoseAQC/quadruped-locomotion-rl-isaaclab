# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

from .torque_locomotion_env_cfg import TorqueLocomotionEnvCfg


class TorqueLocomotionEnv(LocomotionEnv):
    cfg: TorqueLocomotionEnvCfg

    def __init__(
        self,
        cfg: TorqueLocomotionEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # ---------------------------------------------------------
        # Visualización opcional de la dirección real de movimiento
        # ---------------------------------------------------------

        self.velocity_marker = None

        if self.cfg.velocity_arrow_vis:

            marker_cfg = VisualizationMarkersCfg(
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

            self.velocity_marker = VisualizationMarkers(marker_cfg)

    def _update_velocity_marker(self):
        """Actualiza la flecha que representa la velocidad horizontal."""

        # Si la visualización está desactivada, no hacer nada.
        if self.velocity_marker is None:
            return

        # ---------------------------------------------------------
        # Posición y velocidad global del robot
        # ---------------------------------------------------------

        root_pos = self.robot.data.root_pos_w
        root_vel = self.robot.data.root_lin_vel_w

        # ---------------------------------------------------------
        # Velocidad proyectada sobre el plano XY
        # ---------------------------------------------------------

        vel_xy = root_vel.clone()
        vel_xy[:, 2] = 0.0

        # Módulo de la velocidad horizontal
        speed = torch.linalg.norm(
            vel_xy[:, :2],
            dim=1,
        )

        # ---------------------------------------------------------
        # Dirección de movimiento
        # ---------------------------------------------------------

        yaw = torch.atan2(
            vel_xy[:, 1],
            vel_xy[:, 0],
        )

        # Cuaternión [w, x, y, z]
        # Rotación únicamente alrededor del eje Z.
        orientations = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
        )

        orientations[:, 0] = torch.cos(yaw / 2.0)
        orientations[:, 3] = torch.sin(yaw / 2.0)

        # ---------------------------------------------------------
        # Posición visual de la flecha
        # ---------------------------------------------------------

        positions = root_pos.clone()

        # Elevar la flecha respecto al cuerpo del robot
        positions[:, 2] += 0.20

        # ---------------------------------------------------------
        # Escala dinámica
        # ---------------------------------------------------------

        scales = torch.ones(
            (self.num_envs, 3),
            device=self.device,
        )

        # La longitud de la flecha depende de la velocidad.
        scales[:, 0] = torch.clamp(
            speed,
            min=0.10,
            max=1.5,
        )

        # Grosor de la flecha.
        scales[:, 1] = 1.0
        scales[:, 2] = 1.0

        # ---------------------------------------------------------
        # Dibujar marcador
        # ---------------------------------------------------------

        self.velocity_marker.visualize(
            translations=positions,
            orientations=orientations,
            scales=scales,
        )

    def _get_observations(self):
        """Obtiene las observaciones y actualiza la visualización."""

        obs = super()._get_observations()

        if (
            self.cfg.velocity_arrow_vis
            and self.sim.has_gui()
        ):
            self._update_velocity_marker()

        return obs