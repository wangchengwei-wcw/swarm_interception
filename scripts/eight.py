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


@configclass
class QuadcopterBodyrateEnvCfg(DirectRLEnvCfg):
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # 敌方：从随机点出发，持续朝“各 env 原点上方固定高度”飞
    debug_vis_enemy = True
    enemy_spawn_radius = 12.0
    enemy_height_min = 1.0
    enemy_height_max = 3.0
    enemy_speed = 1.5
    enemy_seek_origin = True
    enemy_target_alt = 5.0

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
    flight_altitude = 1.0

    # 奖励相关
    capture_radius = 3.0
    success_distance_threshold = 1.0   # 进入1m即给“成功奖励”（不终止）
    hit_radius = 0.0                   # 真正命中并终止（此处禁用命中终止）
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


class QuadcopterBodyrateEnv(DirectRLEnv):
    cfg: QuadcopterBodyrateEnvCfg
    _is_closed = True

    def __init__(self, cfg: QuadcopterBodyrateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._is_closed = False

        # 世界位置/速度（z-up）
        self.fr_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.fr_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.enemy_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.enemy_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # 动力学（y-up）
        self.g0 = 9.81
        self.theta = torch.zeros(self.num_envs, device=self.device)
        self.psi_v = torch.zeros(self.num_envs, device=self.device)
        self.Vm = torch.zeros(self.num_envs, device=self.device)

        # 统计/动作
        self.episode_sums = {}
        self.prev_dist = None
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # 可视化器
        self.friendly_visualizer = None
        self.enemy_visualizer = None
        self.set_debug_vis(self.cfg.debug_vis)

        # ⚠️ 不要在这里手动调用 self._setup_scene()，基类会调用一次

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
        a = actions.clone().clamp(-1.0, 1.0)

        self._ny = a[:, 0] * self.cfg.ny_max_g
        self._nz = a[:, 1] * self.cfg.nz_max_g
        sp = (a[:, 2] + 1.0) * 0.5
        self.Vm = self.cfg.Vm_min + sp * (self.cfg.Vm_max - self.cfg.Vm_min)

    # 动力学 + 位置更新（无姿态计算/可视化仅平移）
    def _apply_action(self):
        dt = float(self.physics_dt)

        # 友机积分（y-up -> z-up）
        cos_th = torch.cos(self.theta).clamp_min(1e-6)  # 避免除零
        theta_dot = self.g0 * (self._ny - cos_th) / self.Vm.clamp_min(1e-6)
        psi_dot   = - self.g0 * self._nz / (self.Vm.clamp_min(1e-6) * cos_th)
        self.theta = self.theta + theta_dot * dt
        self.psi_v = (self.psi_v + psi_dot * dt + math.pi) % (2.0 * math.pi) - math.pi

        sin_th, cos_th = torch.sin(self.theta), torch.cos(self.theta)
        sin_ps, cos_ps = torch.sin(self.psi_v), torch.cos(self.psi_v)
        Vxm = self.Vm * cos_th * cos_ps
        Vym = self.Vm * sin_th
        Vzm = -self.Vm * cos_th * sin_ps
        V_m = torch.stack([Vxm, Vym, Vzm], dim=-1)
        self.fr_vel_w = y_up_to_z_up(V_m)
        self.fr_pos += self.fr_vel_w * dt

        # 敌机：朝各 env 原点上方目标飞
        if self.cfg.enemy_seek_origin:
            origins = self.terrain.env_origins
            goal = torch.stack([origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)], dim=-1)
            to_goal = goal - self.enemy_pos
            dist_to_goal = torch.linalg.norm(to_goal, dim=1, keepdim=True).clamp_min(1e-6)
            en_dir = to_goal / dist_to_goal
            self.enemy_vel = en_dir * float(self.cfg.enemy_speed)
        self.enemy_pos += self.enemy_vel * dt

        # 可视化（只更新位置）
        if self.friendly_visualizer is not None:
            self.friendly_visualizer.visualize(translations=self.fr_pos)
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self.enemy_pos)

    # 终止
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.linalg.norm(self.enemy_pos - self.fr_pos, dim=1)
        success_hit = dist <= self.cfg.hit_radius

        z = self.fr_pos[:, 2]
        out_z = (z < 0.0) | (z > 200.0)
        xy_rel = self.fr_pos[:, :2] - self.terrain.env_origins[:, :2]
        out_xy = torch.linalg.norm(xy_rel, dim=1) > 300.0
        nan_inf = torch.isnan(self.fr_pos).any(dim=1) | torch.isinf(self.fr_pos).any(dim=1)

        died = out_z | out_xy | nan_inf | success_hit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.extras.setdefault("termination", {})
        self.extras["termination"].update({
            "hit": int(success_hit.sum().item()),
            "out_of_bounds": int((out_z | out_xy).sum().item()),
            "nan_inf": int(nan_inf.sum().item()),
            "time_out": int(time_out.sum().item()),
        })
        return died, time_out

    # ---------- 奖励（仅两项：靠近 + 1m内距离形状） ----------
    def _get_rewards(self) -> torch.Tensor:
        # 敌我距离
        dist = torch.linalg.norm(self.enemy_pos - self.fr_pos, dim=1)  # [N]

        # 1) 靠近奖励：上一步距离 - 本步距离（>0 表示靠近，<0 表示远离）
        if self.prev_dist is None:
            approach = torch.zeros_like(dist)
        else:
            approach = self.prev_dist - dist

        # 2) 1米内距离奖励：距离越近奖励越大；>1米则为0（线性形状）
        near_1m = torch.clamp(1.0 - dist, min=0.0)

        # 合成奖励（如需权重，可在此处乘以系数）
        reward = self.cfg.approach_reward_weight * approach + near_1m

        # 日志统计（可选）
        self.episode_sums.setdefault("approach", torch.zeros_like(dist))
        self.episode_sums.setdefault("near_1m", torch.zeros_like(dist))
        self.episode_sums["approach"] += approach
        self.episode_sums["near_1m"] += near_1m

        # 记住当前距离供下一步比较
        self.prev_dist = dist

        return reward

    # 重置
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # —— 兜底保险：若还未创建 terrain，则先建场景（避免属性缺失）——
        if not hasattr(self, "terrain"):
            self._setup_scene()

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        for k in list(self.episode_sums.keys()):
            self.episode_sums[k][env_ids] = 0.0

        # 设置己方初始位置（固定位置）
        N = len(env_ids)
        origins = self.terrain.env_origins[env_ids]
        self.fr_pos[env_ids] = origins + torch.tensor([0.0, 0.0, self.cfg.flight_altitude], device=self.device)

        # 设置己方的初始角度、速度和过载量
        self.theta[env_ids] = (torch.rand(N, device=self.device) - 0.5) * math.radians(20.0)
        self.psi_v[env_ids] = (torch.rand(N, device=self.device) * 2.0 - 1.0) * math.pi
        self.Vm[env_ids] = self.cfg.Vm_min + torch.rand(N, device=self.device) * (self.cfg.Vm_max - self.cfg.Vm_min)
        self._ny = torch.zeros(self.num_envs, device=self.device)
        self._nz = torch.zeros(self.num_envs, device=self.device)

        # 敌机出生：环上随机
        self._spawn_enemy(env_ids)
        phi = torch.rand(N, device=self.device) * 2.0 * math.pi
        spd = float(self.cfg.enemy_speed)
        self.enemy_vel[env_ids, 0] = spd * torch.cos(phi)
        self.enemy_vel[env_ids, 1] = spd * torch.sin(phi)
        self.enemy_vel[env_ids, 2] = 0.0

        if self.friendly_visualizer is not None:
            self.friendly_visualizer.visualize(translations=self.fr_pos)
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self.enemy_pos)

        # 只更新被 reset 的 prev_dist
        rel = self.enemy_pos[env_ids] - self.fr_pos[env_ids]
        if self.prev_dist is None:
            self.prev_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_dist[env_ids] = torch.linalg.norm(rel, dim=1)
        self.prev_actions[env_ids] = 0.0

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
            self.friendly_visualizer.visualize(translations=self.fr_pos)
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self.enemy_pos)

    # 对外接口：设置 Vm
    def set_friendly_vm(self, vm: float | torch.Tensor, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(vm, (int, float)):
            v = torch.full((len(env_ids),), float(vm), device=self.device)
        else:
            v = vm.to(self.device)
            assert v.shape[0] == len(env_ids), "vm 长度需与 env_ids 数量一致"
        v = torch.clamp(v, self.cfg.Vm_min, self.cfg.Vm_max)
        self.Vm[env_ids] = v


# ---------------- Gym 注册 ----------------
from config import agents

gym.register(
    id="FAST-Quadcopter-Bodyrate",
    entry_point=QuadcopterBodyrateEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterBodyrateEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:quadcopter_sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:quadcopter_skrl_ppo_cfg.yaml",
    },
)
