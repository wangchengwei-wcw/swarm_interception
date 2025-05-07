from __future__ import annotations

import gymnasium as gym
from loguru import logger
import math
import time
import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from envs.quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils.utils import quat_to_ang_between_z_body_and_z_world
from utils.minco import MinJerkOpt
from utils.controller import Controller


@configclass
class QuadcopterRGBCameraEnvCfg(DirectRLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # Env
    episode_length_s = 30.0
    physics_freq = 200.0
    control_freq = 100.0
    mpc_freq = 10.0
    control_decimation = physics_freq // control_freq
    decimation = math.ceil(physics_freq / mpc_freq)  # Environment (replan) decimation
    state_space = 0
    debug_vis = False

    # MINCO trajectory
    num_pieces = 6
    duration = 0.3
    a_max = 10.0
    v_max = 3.0
    p_max = num_pieces * v_max * duration
    action_space = 3 * (num_pieces + 2)  # inner_pts 3 x (num_pieces - 1) + tail_pva 3 x 3
    clip_action = 1.0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / physics_freq,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=5, replicate_physics=True)

    # Robot
    robot: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Camera
    # Hi there, Isaac Sim does not currently provide independent cameras that don’t see other environments.
    # One way to workaround it is to build walls around the environments,
    # which would just be large rectangle prims that block the views of other environments.
    # Another alternative would be to place the environments far apart, or on different height levels.
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=640,
        height=480,
    )
    observation_space = [tiled_camera.height, tiled_camera.width, 3]
    write_image_to_file = False


@configclass
class QuadcopterDepthCameraEnvCfg(QuadcopterRGBCameraEnvCfg):
    # Camera
    max_depth = 10.0
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, max_depth),
        ),
        width=640,
        height=480,
    )
    observation_space = [tiled_camera.height, tiled_camera.width, 1]


class QuadcopterCameraEnv(DirectRLEnv):
    cfg: QuadcopterRGBCameraEnvCfg | QuadcopterDepthCameraEnvCfg

    def __init__(self, cfg: QuadcopterRGBCameraEnvCfg | QuadcopterDepthCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Replan and control decimation must be greater than or equal to 1 #^#")

        if 1 / self.cfg.mpc_freq > self.cfg.num_pieces * self.cfg.duration:
            raise ValueError("Replan period must be less than or equal to the total trajectory duration #^#")

        # Get specific indices
        self.body_id = self.robot.find_bodies("body")[0]
        self.joint_id = self.robot.find_joints(".*joint")[0]

        self.robot_mass = self.robot.root_physx_view.get_masses()[0, 0]
        self.robot_inertia = self.robot.root_physx_view.get_inertias()[0, 0]
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)

        # Traj
        self.traj = None
        self.has_prev_traj = torch.tensor([False] * self.num_envs, device=self.device)

        # Controller
        self.controller = Controller(1 / self.cfg.control_freq, self.gravity, self.robot_mass.to(self.device), self.robot_inertia.to(self.device))
        self.control_counter = 0

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "Currently, the camera environment only supports one image type at a time but the following were" f" provided: {self.cfg.tiled_camera.data_types}"
            )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self.tiled_camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        prim_utils.create_prim("/World/Objects", "Xform")

        # FIXME: Bugs while using relative path of USD files
        cfg_black_oak = sim_utils.UsdFileCfg(usd_path="/home/laji/Wss/e2e_swarm/swarm_rl/assets/black_oak_fall/Black_Oak_Fall.usd", scale=(0.008, 0.008, 0.008))
        cfg_black_oak.func("/World/Objects/Black_Oak", cfg_black_oak, translation=(8.0, 0.0, 0.1))

    def _pre_physics_step(self, actions: torch.Tensor):
        # Action parametrization: waypoints in body frame
        self.waypoints = actions.clone().clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action

        for i in range(self.cfg.num_pieces - 1):
            self.waypoints[:, 3 * (i + 1) : 3 * (i + 2)] += self.waypoints[:, 3 * i : 3 * (i + 1)]

        # Head states
        p_odom = self.robot.data.root_state_w[:, :3]
        q_odom = self.robot.data.root_state_w[:, 3:7]
        v_odom = self.robot.data.root_state_w[:, 7:10]
        a_odom = torch.zeros_like(v_odom)
        if self.traj is not None:
            a_odom = torch.where(self.has_prev_traj.unsqueeze(1), self.traj.get_acc(self.execution_time), a_odom)
        head_pva = torch.stack([p_odom, v_odom, a_odom], dim=2)

        # Waypoints
        inner_pts = torch.zeros((self.num_envs, 3, self.cfg.num_pieces - 1), device=self.device)
        for i in range(self.cfg.num_pieces - 1):
            # Transform to world frame
            inner_pts[:, :, i] = quat_rotate(q_odom, self.waypoints[:, 3 * i : 3 * (i + 1)] * self.cfg.p_max) + p_odom

        # Tail states, transformed to world frame
        p_tail = quat_rotate(q_odom, self.waypoints[:, 3 * (self.cfg.num_pieces - 1) : 3 * (self.cfg.num_pieces + 0)] * self.cfg.p_max) + p_odom
        v_tail = quat_rotate(q_odom, self.waypoints[:, 3 * (self.cfg.num_pieces + 0) : 3 * (self.cfg.num_pieces + 1)] * self.cfg.v_max)
        a_tail = quat_rotate(q_odom, self.waypoints[:, 3 * (self.cfg.num_pieces + 1) : 3 * (self.cfg.num_pieces + 2)] * self.cfg.a_max)
        tail_pva = torch.stack([p_tail, v_tail, a_tail], dim=2)

        durations = torch.full((self.num_envs, self.cfg.num_pieces), self.cfg.duration, device=self.device)

        MJO = MinJerkOpt(head_pva, tail_pva, self.cfg.num_pieces)
        start = time.perf_counter()
        MJO.generate(inner_pts, durations)
        end = time.perf_counter()
        logger.debug(f"Local trajectory generation takes {end - start:.6f}s")

        self.traj = MJO.get_traj()
        self.execution_time = torch.zeros(self.num_envs, device=self.device)
        self.has_prev_traj.fill_(True)

    def _apply_action(self):
        if self.control_counter % self.cfg.control_decimation == 0:
            self.actions = torch.cat(
                (
                    self.traj.get_pos(self.execution_time) + self.terrain.env_origins,
                    self.traj.get_vel(self.execution_time),
                    self.traj.get_acc(self.execution_time),
                    self.traj.get_jer(self.execution_time),
                    torch.zeros_like(self.execution_time).unsqueeze(1),
                    torch.zeros_like(self.execution_time).unsqueeze(1),
                ),
                dim=1,
            )

            start = time.perf_counter()
            self.a_desired_total, self.thrust_desired, self.q_desired, self.w_desired, self.m_desired = self.controller.get_control(
                self.robot.data.root_state_w, self.actions
            )
            end = time.perf_counter()
            logger.debug(f"get_control takes {end - start:.6f}s")

            self._thrust_desired = torch.cat((torch.zeros(self.num_envs, 2, device=self.device), self.thrust_desired.unsqueeze(1)), dim=1)

            self.control_counter = 0
        self.control_counter += 1

        self.robot.set_external_force_and_torque(self._thrust_desired.unsqueeze(1), self.m_desired.unsqueeze(1), body_ids=self.body_id)
        self.execution_time += self.physics_dt

        # TODO: Only for visualization 0_0 Not working due to unknown reason :(
        self.robot.set_joint_velocity_target(self.robot.data.default_joint_vel, env_ids=self.robot._ALL_INDICES)

    def _get_observations(self) -> dict:
        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        if "rgb" in self.cfg.tiled_camera.data_types:
            camera_data = self.tiled_camera.data.output[data_type] / 255.0

            # Normalize the camera data for better training results
            # mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            # camera_data -= mean_tensor

        elif "depth" in self.cfg.tiled_camera.data_types:
            camera_data = self.tiled_camera.data.output[data_type]
            camera_data[camera_data == float("inf")] = 0
            camera_data /= self.cfg.max_depth

        observations = {"image": camera_data.clone(), "odom": self.robot.data.root_state_w.clone()}

        if self.cfg.write_image_to_file:
            save_images_to_file(observations["image"], f"quadcopter_{data_type}.png")

        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_exceed_bounds = torch.logical_or(self.robot.data.root_pos_w[:, 2] < 0.5, self.robot.data.root_pos_w[:, 2] > 10.0)
        ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robot.data.root_quat_w))
        died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 60.0)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.num_envs > 13 and len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.has_prev_traj[env_ids].fill_(False)

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # Create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # Set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Update the markers
        self.goal_pos_visualizer.visualize(self.desired_position)


gym.register(
    id="FAST-Quadcopter-RGB-Camera-Direct-v0",
    entry_point=QuadcopterCameraEnv,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": QuadcopterRGBCameraEnvCfg},
)

gym.register(
    id="FAST-Quadcopter-Depth-Camera-Direct-v0",
    entry_point=QuadcopterCameraEnv,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": QuadcopterDepthCameraEnvCfg},
)
