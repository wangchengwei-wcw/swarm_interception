from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils import quat_to_ang_between_z_body_and_z_world


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment"""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window

        Args:
            env: The environment object
            window_name: The name of the window. Defaults to "IsaacLab"
        """
        # Initialize base window
        super().__init__(env, window_name)
        # Add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # Add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(-5.0, -5.0, 1.3))

    # Env
    episode_length_s = 13.0
    decimation = 2
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = False

    ui_window_class_type = QuadcopterEnvWindow

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3, replicate_physics=True
    )

    # Robot
    robot: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 5.2

    # Reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._robot_inertia = self._robot.root_physx_view.get_inertias()[0]
        self._gravity_magnitude = torch.tensor(
            self.sim.cfg.gravity, device=self.device
        ).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        _actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight
            * self._robot_weight
            * (_actions[:, 0] + 1.0)
            / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * _actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3],
            self._robot.data.root_link_state_w[:, 3:7],
            self._desired_pos_w,
        )
        obs = torch.cat(
            [
                self._robot.data.root_com_lin_vel_b,
                self._robot.data.root_com_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_com_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1
        )
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped
            * self.cfg.distance_to_goal_reward_scale
            * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_exceed_bounds = torch.logical_or(
            self._robot.data.root_link_pos_w[:, 2] < -0.1,
            self._robot.data.root_link_pos_w[:, 2] > 5.2,
        )
        ang_between_z_body_and_z_world = torch.rad2deg(
            quat_to_ang_between_z_body_and_z_world(self._robot.data.root_link_quat_w)
        )
        died = torch.logical_or(
            z_exceed_bounds,
            ang_between_z_body_and_z_world > 60.0,
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_link_pos_w[env_ids],
            dim=1,
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(
            self._desired_pos_w[env_ids, :2]
        ).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(
            self._desired_pos_w[env_ids, 2]
        ).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

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
        self.goal_pos_visualizer.visualize(self._desired_pos_w)


gym.register(
    id="FAST-Quadcopter-Direct-v0",
    entry_point=QuadcopterEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvCfg,
    },
)
