from __future__ import annotations

import math
import torch
import gymnasium as gym

from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import CUBOID_MARKER_CFG
try:
    from isaaclab.markers import FRAME_MARKER_CFG as AXIS_MARKER_CFG
    HAS_AXIS_MARKER = True
except Exception:
    AXIS_MARKER_CFG = None
    HAS_AXIS_MARKER = False


def y_up_to_z_up(vec_m: torch.Tensor) -> torch.Tensor:
    """导弹(y-up) -> Isaac(z-up): (x_m, y_m, z_m) -> (x_w, y_w, z_w) = (x_m, z_m, y_m)"""
    xw = vec_m[..., 0]
    yw = vec_m[..., 2]
    zw = vec_m[..., 1]
    return torch.stack([xw, yw, zw], dim=-1)

def z_up_to_y_up(vec_w: torch.Tensor) -> torch.Tensor:
    """Isaac(z-up) -> 导弹(y-up): (x_w, y_w, z_w) -> (x_m, y_m, z_m) = (x_w, z_w, y_w)"""
    xm = vec_w[..., 0]
    ym = vec_w[..., 2]
    zm = vec_w[..., 1]
    return torch.stack([xm, ym, zm], dim=-1)


@configclass
class FastInterceptionSingleEnvCfg(DirectRLEnvCfg):
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # 敌方：从随机点出发，持续朝“各 env 原点上方固定高度”飞
    debug_vis_enemy = True
    enemy_spawn_radius = 12.0
    enemy_height_min = 1.0
    enemy_height_max = 3.0
    enemy_speed = 1.5
    enemy_seek_origin = True
    enemy_target_alt = 5.0
    enemy_goal_radius = 0.5

    # 友方控制/速度范围
    Vm_min = 1.0
    Vm_max = 3.0
    ny_max_g = 3.0
    nz_max_g = 3.0

    # 观测（9 维）：fr_pos(3) + fr_vel_w(3) + e_w(3)
    observation_space = 9
    state_space = 0
    action_space = 3                   # [ny, nz, Vm_cmd] ∈ [-1,1]
    clip_action = 1.0
    flight_altitude = 0.2

    # 奖励相关
    capture_radius = 3.0
    success_distance_threshold = 1.0   # 进入1m即给“成功奖励”（不终止）
    hit_radius = 1.0                   # 真正命中并终止（此处禁用命中终止）
    success_reward_weight = 300.0
    approach_reward_weight = 1.0       # 靠近奖励系数

    # 频率
    episode_length_s = 30.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

    # 仿真与地面
    sim: SimulationCfg = SimulationCfg(
        dt=1 / physics_freq,
        render_interval=render_decimation,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1000, env_spacing=5, replicate_physics=True)

    debug_vis = True
    debug_vis_goal = False


class FastInterceptionSingleEnv(DirectRLEnv):
    cfg: FastInterceptionSingleEnvCfg
    _is_closed = True

    def __init__(self, cfg: FastInterceptionSingleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._is_closed = False

        # 世界位置/速度（z-up）
        self.fr_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.fr_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.enemy_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.enemy_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # 动力学（y-up）
        self.g0 = 9.81
        self.theta = torch.zeros(self.num_envs, device=self.device)  # pitch（y-up 下绕 +Z）
        self.psi_v = torch.zeros(self.num_envs, device=self.device)  # yaw  （y-up 下绕 +Y）
        self.Vm = torch.zeros(self.num_envs, device=self.device)
        self._ny = torch.zeros(self.num_envs, device=self.device)
        self._nz = torch.zeros(self.num_envs, device=self.device)

        # 统计/动作
        self.episode_sums = {}
        self.prev_dist = None
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # 可视化器
        self.friendly_visualizer = None
        self.enemy_visualizer = None
        self.set_debug_vis(self.cfg.debug_vis)

    # ----------------- 四元数与姿态（w,x,y,z）工具 -----------------
    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton 乘法, q = q1 ⊗ q2, 均为(...,4)且(w,x,y,z)"""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    @staticmethod
    def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
        return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-9)

    def _qy(self, psi: torch.Tensor) -> torch.Tensor:
        """y-up: 绕 +Y 轴的偏航(w,x,y,z)"""
        half = 0.5 * psi
        return torch.stack([torch.cos(half), torch.zeros_like(psi), torch.sin(half), torch.zeros_like(psi)], dim=-1)

    def _qz(self, theta: torch.Tensor) -> torch.Tensor:
        """y-up: 绕 +Z 轴的俯仰(w,x,y,z)"""
        half = 0.5 * theta
        return torch.stack([torch.cos(half), torch.zeros_like(theta), torch.zeros_like(theta), torch.sin(half)], dim=-1)

    def _qx_plus_90(self, N: int) -> torch.Tensor:
        """固定旋转：+90° 绕 +X,把 y-up 姿态变到 z-up 世界系"""
        cx = math.sqrt(0.5)
        sx = cx
        q = torch.tensor([cx, sx, 0.0, 0.0], device=self.device, dtype=self.fr_pos.dtype)
        return q.repeat(N, 1)

    def _friendly_world_quats(self) -> torch.Tensor:
        """
        roll 固定为 0,y-up 下姿态 q_m = R_y(psi) * R_z(theta)
        映射到 z-up: q_w = R_x(+90°) * q_m
        """
        N = self.theta.shape[0]
        q_m = self._quat_mul(self._qy(self.psi_v), self._qz(self.theta))   # (N,4)  这里可以理解为先绕 Z 轴转 θ，再绕 Y 轴转 ψ，如果有roll则还需要再乘以一个q3，这里没有roll就相当于乘以了（1，0，0，0）
        q_w = self._quat_mul(self._qx_plus_90(N), q_m)                     # (N,4)  这里就是将y-up系转换为z-up系
        return self._quat_normalize(q_w)

    def close(self):
        if getattr(self, "_is_closed", True):
            return
        super().close()
        self._is_closed = True

    # 场景
    def _setup_scene(self):
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # 控制输入：ny, nz, Vm_cmd
    def _pre_physics_step(self, actions: torch.Tensor):
        if actions is None:
            return
        # 分别约束：ny,nz ∈ [-1,1]，throttle ∈ [0,1]
        ny = actions[:, 0].clamp(-1.0, 1.0)
        nz = actions[:, 1].clamp(-1.0, 1.0)
        throttle = actions[:, 2].clamp(0.0, 1.0)

        self._ny = ny * self.cfg.ny_max_g
        self._nz = nz * self.cfg.nz_max_g
        self.Vm = self.cfg.Vm_min + throttle * (self.cfg.Vm_max - self.cfg.Vm_min)
        print(f"Vm: {self.Vm.mean().item():.2f} m/s", end="\r")

    # 动力学 + 位置更新（可视化含姿态）
    def _apply_action(self):
        dt = float(self.physics_dt)

        # ---- 1) 角速度计算并积分 ----
        cos_th_now = torch.cos(self.theta).clamp_min(1e-6)
        theta_rate = self.g0 * (self._ny - cos_th_now) / self.Vm.clamp_min(1e-6)
        psi_rate   = - self.g0 * self._nz / (self.Vm.clamp_min(1e-6) * cos_th_now)

        # ----角加速度限幅，让其在合理范围内----
        THETA_RATE_LIMIT = 1.0   # rad/s
        PSI_RATE_LIMIT   = 1.0   # rad/s
        theta_rate = torch.clamp(theta_rate, -THETA_RATE_LIMIT, THETA_RATE_LIMIT)
        psi_rate   = torch.clamp(psi_rate,   -PSI_RATE_LIMIT,   PSI_RATE_LIMIT)

        self.theta = self.theta + theta_rate * dt
        self.psi_v = (self.psi_v + psi_rate * dt + math.pi) % (2.0 * math.pi) - math.pi

        # ---- 2) 速度向量（y-up）-> 世界 z-up ----
        sin_th, cos_th = torch.sin(self.theta), torch.cos(self.theta)
        sin_ps, cos_ps = torch.sin(self.psi_v), torch.cos(self.psi_v)

        Vxm = self.Vm * cos_th * cos_ps
        Vym = self.Vm * sin_th
        Vzm = -self.Vm * cos_th * sin_ps
        V_m = torch.stack([Vxm, Vym, Vzm], dim=-1)

        self.fr_vel_w = y_up_to_z_up(V_m)   # (xw, yw, zw) = (xm, zm, ym)
        self.fr_pos += self.fr_vel_w * dt

        # ---- 3) 敌机更新 ----
        if self.cfg.enemy_seek_origin:
            origins = self.terrain.env_origins
            goal = torch.stack([origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)], dim=-1)
            to_goal = goal - self.enemy_pos
            dist_to_goal = torch.linalg.norm(to_goal, dim=1, keepdim=True).clamp_min(1e-6)
            en_dir = to_goal / dist_to_goal
            self.enemy_vel = en_dir * float(self.cfg.enemy_speed)
        self.enemy_pos += self.enemy_vel * dt

        # ---- 4) 可视化（位置 + 姿态）----
        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(translations=self.fr_pos, orientations=fr_quats)
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self.enemy_pos)

    # ---------- 奖励（仅两项：靠近 + 1m内距离形状） ----------
    def _get_rewards(self) -> torch.Tensor:
        dist = torch.linalg.norm(self.enemy_pos - self.fr_pos, dim=1)  # [N]

        if self.prev_dist is None:
            approach = torch.zeros_like(dist)
        else:
            approach = self.prev_dist - dist

        near_1m = torch.clamp(1.0 - dist, min=0.0)
        reward = self.cfg.approach_reward_weight * approach + near_1m

        self.episode_sums.setdefault("approach", torch.zeros_like(dist))
        self.episode_sums.setdefault("near_1m", torch.zeros_like(dist))
        self.episode_sums["approach"] += approach
        self.episode_sums["near_1m"] += near_1m

        self.prev_dist = dist
        return reward

    # 终止
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.linalg.norm(self.enemy_pos - self.fr_pos, dim=1)
        success_hit = dist < self.cfg.hit_radius

        z = self.fr_pos[:, 2]
        out_z = (z < 0.0) | (z > 200.0)
        xy_rel = self.fr_pos[:, :2] - self.terrain.env_origins[:, :2]
        out_xy = torch.linalg.norm(xy_rel, dim=1) > 300.0
        nan_inf = torch.isnan(self.fr_pos).any(dim=1) | torch.isinf(self.fr_pos).any(dim=1)

        origins = self.terrain.env_origins
        goal = torch.stack([origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)], dim=-1)
        enemy_goal_dist = torch.linalg.norm(goal - self.enemy_pos, dim=1)
        enemy_goal_reached = enemy_goal_dist < float(self.cfg.enemy_goal_radius)

        died = out_z | out_xy | nan_inf | success_hit | enemy_goal_reached
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.extras.setdefault("termination", {})
        self.extras["termination"].update({
            "hit": int(success_hit.sum().item()),
            "out_of_bounds": int((out_z | out_xy).sum().item()),
            "nan_inf": int(nan_inf.sum().item()),
            "enemy_goal": int(enemy_goal_reached.sum().item()),
            "time_out": int(time_out.sum().item()),
        })
        return died, time_out

    """重置指定 env 的状态
        1.重置episode_sums
        2.重置友方位置/速度大小、方向/动力学状态/控制输入
          速度方向：先均匀球面随机一个方向，再反解 theta / psi_v
        3.重置敌方位置/速度"""
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # 若未创建地形，先建场景
        if not hasattr(self, "terrain"):
            self._setup_scene()

        # 归一化 env_ids
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # 清零本回合统计
        for k in list(self.episode_sums.keys()):
            self.episode_sums[k][env_ids] = 0.0

        # 出生点：各 env 原点 + 固定高度
        N = len(env_ids)
        origins = self.terrain.env_origins[env_ids]
        self.fr_pos[env_ids] = origins + torch.tensor([0.0, 0.0, self.cfg.flight_altitude], device=self.device)

        # 先生成敌方初始位置（后面要用来定朝向）
        self._spawn_enemy(env_ids)

        # —— 友方初始速度大小：Vm = 0 ——
        Vm0 = torch.zeros(N, device=self.device)
        self.Vm[env_ids] = Vm0
        # 初始速度（世界 z-up）：全零
        self.fr_vel_w[env_ids] = torch.zeros(N, 3, device=self.device)

        # —— 计算“指向敌方”的方向，并在 y-up 下反解 theta / psi —— 
        # 世界(z-up)下的相对向量：友方->敌方
        rel_w = self.enemy_pos[env_ids] - self.fr_pos[env_ids]                 # [N,3]
        # 映射到 y-up，再归一化
        rel_m = z_up_to_y_up(rel_w)                                           # [N,3] (xm,ym,zm)
        rel_m = rel_m / rel_m.norm(dim=1, keepdim=True).clamp_min(1e-9)

        # 由方向反解： Vx=Vm*cosθ*cosψ, Vy=Vm*sinθ, Vz=-Vm*cosθ*sinψ （y-up）
        sin_th = rel_m[:, 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta0 = torch.asin(sin_th)                                           # θ ∈ [-π/2, π/2]
        psi0   = torch.atan2(-rel_m[:, 2], rel_m[:, 0])                       # ψ ∈ (-π, π]

        # 赋初始姿态（roll=0 隐含）
        self.theta[env_ids] = theta0
        self.psi_v[env_ids] = psi0

        # 控制输入初始化（可选：不自发转向/抬头）
        self._ny[env_ids] = 0.0
        self._nz[env_ids] = 0.0

        # 敌机初始速度（保持你的逻辑：环上随机切向速度）
        phi = torch.rand(N, device=self.device) * 2.0 * math.pi
        spd = float(self.cfg.enemy_speed)
        self.enemy_vel[env_ids, 0] = spd * torch.cos(phi)
        self.enemy_vel[env_ids, 1] = spd * torch.sin(phi)
        self.enemy_vel[env_ids, 2] = 0.0

        # 可视化（位置 + 姿态）
        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(translations=self.fr_pos, orientations=fr_quats)
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self.enemy_pos)

        # prev_dist / 动作 / 回合步计数
        rel = self.enemy_pos[env_ids] - self.fr_pos[env_ids]
        if self.prev_dist is None:
            self.prev_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_dist[env_ids] = torch.linalg.norm(rel, dim=1)
        self.prev_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

    # 观测： [fr_pos(3) , fr_vel_w(3) , e_w(3)]
    def _get_observations(self) -> dict:
        pos_f = self.fr_pos                           # [N,3]
        vel_f = self.fr_vel_w                         # [N,3]
        rel_p_w = self.enemy_pos - self.fr_pos        # [N,3] 友方指向敌方的向量
        dist = torch.linalg.norm(rel_p_w, dim=1, keepdim=True).clamp_min(1e-6)
        e_w = rel_p_w / dist                          # [N,3] 友方指向敌方的单位向量
        obs = torch.cat([pos_f, vel_f, e_w], dim=-1)  # [N,9]
        return {"policy": obs, "odom": obs.clone()}

    # 工具
    def _spawn_enemy(self, env_ids: torch.Tensor):
        origins = self.terrain.env_origins[env_ids]
        origins_xy = origins[:, :2]
        R = float(self.cfg.enemy_spawn_radius)
        hmin = float(self.cfg.enemy_height_min)
        hmax = float(self.cfg.enemy_height_max)

        # 在各自 env 原点为圆心、半径 R 的圆环上随机一个角度，生成敌机初始水平位置
        ang = torch.rand(len(env_ids), device=self.device) * 2.0 * math.pi
        ex = origins_xy[:, 0] + R * torch.cos(ang)
        ey = origins_xy[:, 1] + R * torch.sin(ang)
        ez = origins[:, 2] + torch.rand(len(env_ids), device=self.device) * (hmax - hmin) + hmin

        self.enemy_pos[env_ids, 0] = ex
        self.enemy_pos[env_ids, 1] = ey
        self.enemy_pos[env_ids, 2] = ez

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self.friendly_visualizer is None:
                if HAS_AXIS_MARKER and AXIS_MARKER_CFG is not None:
                    f_cfg = AXIS_MARKER_CFG.copy()
                    f_cfg.prim_path = "/Visuals/FriendlyAxis"
                    self.friendly_visualizer = VisualizationMarkers(f_cfg)
                else:
                    f_cfg = CUBOID_MARKER_CFG.copy()
                    f_cfg.markers["cuboid"].size = (0.18, 0.18, 0.18)
                    f_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 1.0))
                    f_cfg.prim_path = "/Visuals/Friendly"
                    self.friendly_visualizer = VisualizationMarkers(f_cfg)
                self.friendly_visualizer.set_visibility(True)

            if self.cfg.debug_vis_enemy and self.enemy_visualizer is None:
                e_cfg = CUBOID_MARKER_CFG.copy()
                e_cfg.markers["cuboid"].size = (0.15, 0.15, 0.15)
                e_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                e_cfg.prim_path = "/Visuals/Enemy"
                self.enemy_visualizer = VisualizationMarkers(e_cfg)
                self.enemy_visualizer.set_visibility(True)
        else:
            if self.friendly_visualizer is not None:
                self.friendly_visualizer.set_visibility(False)
            if self.enemy_visualizer is not None:
                self.enemy_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(translations=self.fr_pos, orientations=fr_quats)
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self.enemy_pos)


# ---------------- Gym 注册 ----------------
from config import agents

gym.register(
    id="FAST-Intercept-Single",
    entry_point=FastInterceptionSingleEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FastInterceptionSingleEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:quadcopter_sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:quadcopter_skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Loitering_Munition_interception_single_rsl_rl_ppo_cfg:FASTInterceptSinglePPORunnerCfg",
    },
)
