# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ImuCfg

from isaaclab.terrains import (
    TerrainImporterCfg,
    TerrainGeneratorCfg,
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfWaveTerrainCfg,
)

from isaaclab.utils import configclass

from isaaclab_assets.robots.spdrbot import SPDRBOT_CFG

# ==============================================================
# TERRAIN FLAGS
# ==============================================================

USE_ROUGH_TERRAIN = True

USE_RANDOM_UNIFORM = True
USE_PYRAMID_SLOPE = True
USE_PYRAMID_STAIRS = True
USE_DISCRETE_OBSTACLES = True
USE_STEPPING_STONES = True
USE_WAVES = True

def build_spdrbot_terrains() -> dict:

    terrains = {}

    if USE_RANDOM_UNIFORM:
        terrains["random_uniform"] = HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.01, 0.03),
            noise_step=0.005,
            downsampled_scale=0.10,
            border_width=0.25,
        )

    if USE_PYRAMID_SLOPE:
        terrains["pyramid_slope"] = HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.0, 0.30),
            platform_width=2.0,
            border_width=0.25,
        )

    if USE_PYRAMID_STAIRS:
        terrains["pyramid_stairs"] = HfPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.01, 0.05),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.25,
        )

    if USE_DISCRETE_OBSTACLES:
        terrains["discrete_obstacles"] = HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            obstacle_height_range=(0.01, 0.05),
            obstacle_width_range=(0.10, 0.40),
            num_obstacles=30,
            platform_width=2.0,
            border_width=0.25,
        )

    if USE_STEPPING_STONES:
        terrains["stepping_stones"] = HfSteppingStonesTerrainCfg(
            proportion=1.0,
            stone_height_max=0.04,
            stone_width_range=(0.15, 0.40),
            stone_distance_range=(0.02, 0.10),
            holes_depth=-0.10,
            platform_width=2.0,
            border_width=0.25,
        )

    if USE_WAVES:
        terrains["waves"] = HfWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.01, 0.04),
            num_waves=3,
            border_width=0.25,
        )

    return terrains


SPDRBOT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,

    size=(8.0, 8.0),
    border_width=20.0,

    num_rows=10,
    num_cols=20,

    horizontal_scale=0.05,
    vertical_scale=0.005,

    slope_threshold=0.75,

    difficulty_range=(0.0, 1.0),

    use_cache=True,

    sub_terrains=build_spdrbot_terrains(),
)


@configclass
class ImuintegrationEnvCfg(DirectRLEnvCfg):

    episode_length_s = 15

    # 240 Hz física / 4 = 60 Hz política
    decimation = 4

    action_scale = 0.5
    action_space = 12

    # q_j        = 12
    # qdot_j     = 12
    # theta      = 3
    # theta_dot  = 3
    # C1_ddot    = 3
    # C_dot      = 3a
    # z          = 1
    # alpha      = 1
    #
    # TOTAL = 38
    observation_space = 38

    state_space = 0
    max_torque = 5.0

    velocity_arrow_vis: bool = False

    # ----------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=decimation,
    )

    # ----------------------------------------------------------
    # Terrain
    # ----------------------------------------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",

        terrain_type=(
            "generator"
            if USE_ROUGH_TERRAIN
            else "plane"
        ),

        terrain_generator=(
            SPDRBOT_ROUGH_TERRAINS_CFG
            if USE_ROUGH_TERRAIN
            else None
        ),

        collision_group=-1,

        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),

        debug_vis=False,
    )

    # ----------------------------------------------------------
    # Scene
    # ----------------------------------------------------------

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # ----------------------------------------------------------
    # Robot
    # ----------------------------------------------------------

    robot: ArticulationCfg = SPDRBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # ----------------------------------------------------------
    # IMU
    # ----------------------------------------------------------

    imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",

        # Actualizar en cada paso de física
        update_period=0.0,

        debug_vis=False,

        offset=ImuCfg.OffsetCfg(
            # De momento suponemos la IMU en el origen del base_link
            pos=(0.0, 0.0, 0.0),

            # Quaternion identidad (w, x, y, z)
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    joint_gears: list = [
        275, 275, 275,
        275, 275, 275,
        275, 275, 275,
        275, 275, 275,
    ]

    death_cost: float = -2.0

    alive_reward_scale: float = 0.5
    heading_weight: float = 0.5
    up_weight: float = 0.1
    actions_cost_scale: float = 0.002
    energy_cost_scale: float = 0.005
    termination_height: float = 0.01
    angular_velocity_scale: float = 0.001
    lateral_velocity_scale: float = 0.003