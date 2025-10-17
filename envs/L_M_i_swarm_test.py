from __future__ import annotations

import math
import torch
import gymnasium as gym
import time
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import CUBOID_MARKER_CFG

# ---------------- 配置 ----------------
@configclass
class FastInterceptionSwarmEnvCfg(DirectRLEnvCfg):
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # ---------- 数量控制 ----------
    swarm_size: int = 5                 # 便捷参数：同时设置友机/敌机数量
    friendly_size: int = 6              # 显式设置（可选）
    enemy_size: int = 6                 # 显式设置（可选）

    # 敌机出生区域（圆盘）与最小间隔
    debug_vis_enemy = True
    enemy_height_min = 3.0
    enemy_height_max = 3.0
    enemy_speed = 1.5
    enemy_seek_origin = True
    enemy_target_alt = 3.0
    enemy_goal_radius = 0.5
    enemy_cluster_ring_radius: float = 25.0   # R：以 env 原点为圆心，在半径 R 的圆周上选簇中心
    enemy_cluster_radius: float = 5.0         # r：以簇中心为圆心的小圆半径
    enemy_min_separation: float = 4.0         # 敌机间最小 XY 间距

    # 友方“速度命令”限幅（m/s）
    Vm_min = 0.0
    Vm_max = 3.0

    formation_spacing = 0.8
    flight_altitude = 0.2

    # 单机观测/动作维度（实际 env * M）
    single_observation_space = 9
    single_action_space = 3               # 直接输出 [vx, vy, vz]（世界系，单位 m/s）

    # 占位（会在 __init__ 时按 M 覆盖）
    observation_space = 9
    state_space = 0
    action_space = 3
    clip_action = 1.0                     # 策略输出归一化范围 [-1,1]

    # 奖励相关
    hit_radius = 1.0
    centroid_approach_weight = 1.0
    hit_reward_weight: float = 1000.0
    heading_align_weight = 0.0            # 已不使用

    # 频率
    episode_length_s = 30.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

    # === 投影与射线可视化 ===
    proj_vis_enable: bool = False
    proj_max_envs: int = 8
    proj_ray_step: float = 0.2
    proj_ray_size: tuple[float,float,float] = (0.08, 0.08, 0.08)
    proj_friend_size: tuple[float,float,float] = (0.12, 0.12, 0.12)
    proj_enemy_size:  tuple[float,float,float] = (0.12, 0.12, 0.12)
    proj_centroid_size: tuple[float,float,float] = (0.16, 0.16, 0.16)

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

# ---------------- 环境 ----------------
class FastInterceptionSwarmEnv(DirectRLEnv):
    cfg: FastInterceptionSwarmEnvCfg
    _is_closed = True

    def __init__(self, cfg: FastInterceptionSwarmEnvCfg, render_mode: str | None = None, **kwargs):
        # 解析数量
        M = cfg.friendly_size if cfg.friendly_size is not None else cfg.swarm_size
        E = cfg.enemy_size if cfg.enemy_size is not None else cfg.swarm_size
        if M != E:
            raise ValueError(f"friendly_size({M}) 必须等于 enemy_size({E}) 或使用 swarm_size 统一设置。")

        single_obs_dim = 6 + 3 * E      # pos(3) + vel(3) + unit_to_all_enemies(3*E)
        cfg.single_observation_space = single_obs_dim
        cfg.observation_space = single_obs_dim * M
        cfg.action_space = cfg.single_action_space * M

        super().__init__(cfg, render_mode, **kwargs)
        self._is_closed = False

        # ---------- 维度 ----------
        self.M = int(M)  # 友机数
        self.E = int(E)  # 敌机数（=M）
        N = self.num_envs
        dev = self.device
        dtype = torch.float32

        # 友机状态 [N,M,3]
        self.fr_pos = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)
        self.fr_vel_w = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)   # 实际应用速度（世界系）
        self.fr_cmd_vel = torch.zeros(N, self.M, 3, device=dev, dtype=dtype) # 策略命令速度（世界系）

        # 敌机状态 [N,E,3]
        self.enemy_pos = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)
        self.enemy_vel = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)

        # —— 冻结掩码与命中位置（全对全）——
        self.friend_frozen = torch.zeros(N, self.M, device=dev, dtype=torch.bool)      # [N,M]
        self.enemy_frozen  = torch.zeros(N, self.E, device=dev, dtype=torch.bool)      # [N,E]
        self.friend_capture_pos = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)   # [N,M,3]
        self.enemy_capture_pos  = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)   # [N,E,3]

        # 统计/动作缓存
        self.episode_sums = {}
        self.episode_sums["align"]          = torch.zeros(self.num_envs, self.M, device=dev)  # 已不使用
        self.episode_sums["hit_bonus"]      = torch.zeros(self.num_envs,       device=dev)

        # 一次性奖励发放标记
        self._newly_frozen_friend = torch.zeros(N, self.M, dtype=torch.bool, device=dev)
        self._newly_frozen_enemy  = torch.zeros(N, self.E, dtype=torch.bool, device=dev)

        # 可视化器
        self.friendly_visualizer = None
        self.enemy_visualizer = None
        # —— 投影/射线可视化器 —— 
        self.centroid_marker = None
        self.ray_marker = None
        self.friend_proj_marker = None
        self.enemy_proj_marker = None

        self.set_debug_vis(self.cfg.debug_vis)
        self._profile_print = bool(getattr(self.cfg, "profile_print", False)) # 耗时打印开关

        # --- 缓存（每步只更新一次）---
        self._enemy_centroid_init = torch.zeros(self.num_envs, 3, device=self.device, dtype=self.fr_pos.dtype)
        self._enemy_centroid = torch.zeros(self.num_envs, 3, device=dev, dtype=dtype)      # [N,3]
        self._enemy_active = torch.zeros(self.num_envs, self.E, device=dev, dtype=torch.bool)  # [N,E]
        self._enemy_active_any = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)  # [N]
        self._goal_e = None                       # [N,3]
        self._axis_hat = torch.zeros(self.num_envs, 3, device=dev, dtype=dtype)  # goal_e->centroid 单位向量

    # —————————————————— ↓↓↓↓↓工具区↓↓↓↓↓ ——————————————————
    def _flatten_agents(self, X: torch.Tensor) -> torch.Tensor:
        return X.reshape(-1, X.shape[-1])

    def close(self):
        if getattr(self, "_is_closed", True):
            return
        super().close()
        self._is_closed = True

    # --------- ↓↓↓↓↓敌方生成相关↓↓↓↓↓ ---------
    def _rebuild_goal_e(self):
        origins = self.terrain.env_origins  # [N,3]
        self._goal_e = torch.stack(
            [origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)],
            dim=-1
        )  # [N,3]

    def _refresh_enemy_cache(self):
        if self._goal_e is None:
            self._rebuild_goal_e()

        enemy_active = ~self.enemy_frozen                        # [N,E]
        e_mask = enemy_active.unsqueeze(-1).float()              # [N,E,1]
        sum_pos = (self.enemy_pos * e_mask).sum(dim=1)           # [N,3]
        cnt     = e_mask.sum(dim=1).clamp_min(1.0)               # [N,1]
        centroid = sum_pos / cnt                                 # [N,3]

        self._enemy_centroid = centroid
        self._enemy_active   = enemy_active
        self._enemy_active_any = enemy_active.any(dim=1)         # [N]

        axis = centroid - self._goal_e                           # [N,3]
        norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._axis_hat = axis / norm                             # [N,3]

    def _spawn_enemy(self, env_ids: torch.Tensor):
        # （保持和你的版本一致，略去注释以节省篇幅）
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        dev = self.fr_pos.device
        env_ids = env_ids.to(dtype=torch.long, device=dev)

        origins_all = self.terrain.env_origins
        if origins_all.device != dev:
            origins_all = origins_all.to(dev)
        origins = origins_all[env_ids]  # [N,3]

        if self._goal_e is None:
            self._rebuild_goal_e()
        goal_e = self._goal_e[env_ids]  # [N,3]

        N, E = env_ids.shape[0], self.E
        s_min = float(self.cfg.enemy_min_separation)
        hmin, hmax = float(self.cfg.enemy_height_min), float(self.cfg.enemy_height_max)

        R_center = float(getattr(self.cfg, "enemy_cluster_ring_radius", 8.0))
        center_jitter = float(getattr(self.cfg, "enemy_center_jitter", 0.0))

        def tmpl_triangle(E, s):
            if E == 0:
                return torch.zeros(0, 2, device=dev)
            dy = s * math.sqrt(3.0) / 2.0
            xs, ys, cnt = [], [], 0
            R = int(math.ceil((math.sqrt(8.0 * E + 1.0) - 1.0) / 2.0))
            for r in range(R):
                num_in_row = r + 1
                x_r = r * dy
                y_start = -0.5 * r * s
                for k in range(num_in_row):
                    xs.append(x_r); ys.append(y_start + k * s)
                    cnt += 1
                    if cnt >= E:
                        xy = torch.stack([torch.tensor(xs, device=dev),
                                          torch.tensor(ys, device=dev)], dim=-1)
                        return xy
            xy = torch.stack([torch.tensor(xs, device=dev),
                              torch.tensor(ys, device=dev)], dim=-1)
            return xy[:E]

        templates = []
        for builder in (tmpl_triangle,):
            xy = builder(E, s_min)
            xy = xy - xy.mean(dim=0, keepdim=True)
            templates.append(xy)
        templates = torch.stack(templates, dim=0)  # [1, E, 2]
        F = templates.shape[0]
        f_idx = torch.randint(low=0, high=F, size=(N,), device=dev)  # 全 0
        local_xy = templates[f_idx, :, :]

        theta = 2.0 * math.pi * torch.rand(N, device=dev)
        centers = torch.stack([
            origins[:, 0] + R_center * torch.cos(theta),
            origins[:, 1] + R_center * torch.sin(theta)
        ], dim=1)                                                       # [N,2]
        if center_jitter > 0.0:
            centers = centers + (torch.rand(N, 2, device=dev) - 0.5) * (2.0 * center_jitter)

        goal_xy = goal_e[:, :2]
        head_vec = goal_xy - centers
        head = head_vec / head_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        c, s = head[:, 0], head[:, 1]
        Rm = torch.stack([torch.stack([c, -s], dim=-1),
                          torch.stack([s,  c], dim=-1)], dim=1)         # [N,2,2]
        rotated = torch.matmul(local_xy, Rm.transpose(1, 2))            # [N,E,2]
        xy = centers.unsqueeze(1) + rotated                             # [N,E,2]

        z = origins[:, 2:3].unsqueeze(1) \
            + (hmin + torch.rand(N, E, 1, device=dev) * (hmax - hmin))  # [N,E,1]
        enemy_pos = torch.cat([xy, z], dim=-1)  # [N,E,3]
        self.enemy_pos[env_ids] = enemy_pos

        if self._profile_print:
            self._cuda_sync_if_needed()
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _spawn_enemy(triangle): envs={N}, E={E}, s_min={s_min:.2f}, R={R_center:.2f} -> {dt:.2f} ms")

    # --------- ↓↓↓↓↓可视化相关↓↓↓↓↓ ---------
    def _build_ray_dots(self, c: torch.Tensor, g: torch.Tensor, step: float) -> torch.Tensor:
        dev = self.device
        v = g - c
        L = torch.linalg.norm(v).item()
        if L < 1e-6:
            return c.unsqueeze(0)
        n = max(2, int(L / max(step, 1e-6)) + 1)
        ts = torch.linspace(0.0, 1.0, n, device=dev, dtype=torch.float32).unsqueeze(1)  # [n,1]
        pts = c.unsqueeze(0) + ts * v.unsqueeze(0)  # [n,3]
        return pts

    def _update_projection_debug_vis(self):
        if not getattr(self.cfg, "proj_vis_enable", False):
            return
        if self._goal_e is None:
            self._rebuild_goal_e()

        N_draw = int(min(self.num_envs, getattr(self.cfg, "proj_max_envs", 8)))
        if N_draw <= 0:
            return

        centroid_pts, ray_pts, fr_proj_pts, en_proj_pts = [], [], [], []

        for ei in range(N_draw):
            enemy_active = self._enemy_active[ei]  # [E]
            if not enemy_active.any():
                continue

            centroid = self._enemy_centroid[ei]    # [3]
            g = self._goal_e[ei]                   # [3]
            ray_pts.append(self._build_ray_dots(centroid, g, float(self.cfg.proj_ray_step)))
            centroid_pts.append(centroid.unsqueeze(0))

            axis_hat = self._axis_hat[ei]          # [3]

            friend_active = (~self.friend_frozen[ei])          # [M]
            fr_pos = self.fr_pos[ei]                           # [M,3]
            s_f = ((fr_pos - centroid.unsqueeze(0)) * axis_hat.unsqueeze(0)).sum(dim=-1)  # [M]
            p_f = centroid.unsqueeze(0) + s_f.unsqueeze(1) * axis_hat.unsqueeze(0)        # [M,3]
            fr_proj_pts.append(p_f[friend_active])

            en_pos = self.enemy_pos[ei]                        # [E,3]
            s_e = ((en_pos - centroid.unsqueeze(0)) * axis_hat.unsqueeze(0)).sum(dim=-1)  # [E]
            p_e = centroid.unsqueeze(0) + s_e.unsqueeze(1) * axis_hat.unsqueeze(0)        # [E,3]
            en_proj_pts.append(p_e[enemy_active])

        if len(centroid_pts) > 0 and self.centroid_marker is not None:
            self.centroid_marker.visualize(translations=torch.cat(centroid_pts, dim=0))
        if len(ray_pts) > 0 and self.ray_marker is not None:
            self.ray_marker.visualize(translations=torch.cat(ray_pts, dim=0))
        if len(fr_proj_pts) > 0 and self.friend_proj_marker is not None:
            self.friend_proj_marker.visualize(translations=torch.cat(fr_proj_pts, dim=0))
        if len(en_proj_pts) > 0 and self.enemy_proj_marker is not None:
            self.enemy_proj_marker.visualize(translations=torch.cat(en_proj_pts, dim=0))

    def _set_debug_vis_impl(self, debug_vis: bool):
        """改成总是使用立方体标记（无需姿态）"""
        if debug_vis:
            if self.friendly_visualizer is None:
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

            if getattr(self.cfg, "proj_vis_enable", True):
                if self.centroid_marker is None:
                    c_cfg = CUBOID_MARKER_CFG.copy()
                    c_cfg.prim_path = "/Visuals/Proj/Centroid"
                    c_cfg.markers["cuboid"].size = tuple(self.cfg.proj_centroid_size)
                    c_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.9, 0.2))
                    self.centroid_marker = VisualizationMarkers(c_cfg)
                    self.centroid_marker.set_visibility(True)

                if self.ray_marker is None:
                    r_cfg = CUBOID_MARKER_CFG.copy()
                    r_cfg.prim_path = "/Visuals/Proj/Ray"
                    r_cfg.markers["cuboid"].size = tuple(self.cfg.proj_ray_size)
                    r_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.75))
                    self.ray_marker = VisualizationMarkers(r_cfg)
                    self.ray_marker.set_visibility(True)

                if self.friend_proj_marker is None:
                    fp_cfg = CUBOID_MARKER_CFG.copy()
                    fp_cfg.prim_path = "/Visuals/Proj/Friend"
                    fp_cfg.markers["cuboid"].size = tuple(self.cfg.proj_friend_size)
                    fp_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.5, 1.0))
                    self.friend_proj_marker = VisualizationMarkers(fp_cfg)
                    self.friend_proj_marker.set_visibility(True)

                if self.enemy_proj_marker is None:
                    ep_cfg = CUBOID_MARKER_CFG.copy()
                    ep_cfg.prim_path = "/Visuals/Proj/Enemy"
                    ep_cfg.markers["cuboid"].size = tuple(self.cfg.proj_enemy_size)
                    ep_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.25))
                    self.enemy_proj_marker = VisualizationMarkers(ep_cfg)
                    self.enemy_proj_marker.set_visibility(True)
        else:
            if self.friendly_visualizer is not None:
                self.friendly_visualizer.set_visibility(False)
            if self.enemy_visualizer is not None:
                self.enemy_visualizer.set_visibility(False)
            if self.centroid_marker is not None:
                self.centroid_marker.set_visibility(False)
            if self.ray_marker is not None:
                self.ray_marker.set_visibility(False)
            if self.friend_proj_marker is not None:
                self.friend_proj_marker.set_visibility(False)
            if self.enemy_proj_marker is not None:
                self.enemy_proj_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        """仅位置可视化（无姿态）"""
        if self.friendly_visualizer is not None:
            self.friendly_visualizer.visualize(translations=self._flatten_agents(self.fr_pos))
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

    def _setup_scene(self):
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._rebuild_goal_e()
    
    def _cuda_sync_if_needed(self):
        try:
            dev = getattr(self, "device", None)
            if isinstance(dev, torch.device):
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
            elif isinstance(dev, str):
                if dev.startswith("cuda"):
                    torch.cuda.synchronize(torch.device(dev))
        except Exception:
            pass
    
    # —————————————————— ↓↓↓↓↓主工作区↓↓↓↓↓ ——————————————————
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        解析动作为“世界系速度命令”：
        - 支持 [N, 3*M] / [N, M, 3] / [N, 3]（广播到 M）
        - 对已冻结者置零
        - 限幅：vel_cmd = clip(action, -1, 1) * Vm_max
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        if actions is None:
            return
        N, M = self.num_envs, self.M

        # 统一为 [N, M, 3]
        if actions.dim() == 2 and actions.shape[1] == 3*M:
            act = actions.view(N, M, 3)
        elif actions.dim() == 3 and actions.shape[1:] == (M, 3):
            act = actions
        elif actions.dim() == 2 and actions.shape[1] == 3:
            act = actions.view(N, 1, 3).repeat(1, M, 1)
        else:
            raise RuntimeError(f"Action shape mismatch. Got {tuple(actions.shape)}, expected [N,{3*M}] or [N,{M},3] or [N,3].")

        # 限幅到 [-1, 1]，再按 Vm_max 缩放到 m/s
        act = act.clamp(-float(self.cfg.clip_action), float(self.cfg.clip_action))
        v_scale = float(self.cfg.Vm_max)
        vel_cmd = act * v_scale  # [N,M,3]，世界系

        # 已冻结者速度置零
        if hasattr(self, "friend_frozen") and self.friend_frozen is not None:
            vel_cmd = torch.where((~self.friend_frozen).unsqueeze(-1), vel_cmd, torch.zeros_like(vel_cmd))

        self.fr_cmd_vel = vel_cmd  # 缓存到下一步推进使用

        if self._profile_print:
            self._cuda_sync_if_needed()
            print(f"[TIME] _pre_physics_step: {(time.perf_counter() - t0)*1000.0:.3f} ms")

    def _apply_action(self):
        """
        每个物理步推进一次（步首判定：距离<=1m即命中并冻结):
        0) 缓存步首状态
        0.5) 步首判定命中并冻结（记录捕获点）
        1) 友机速度 = fr_cmd_vel（冻结者=0）
        2) 敌机速度（可选寻标；冻结者=0）
        3) 积分到步末（pos += vel * dt）
        4) 覆盖冻结对象的位置/速度
        5) 刷新缓存 + 可视化
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        dt = float(self.physics_dt)
        is_first_substep = ((self._sim_step_counter - 1) % self.cfg.decimation) == 0
        if is_first_substep:
            self._newly_frozen_friend[:] = False
            self._newly_frozen_enemy[:]  = False

        N, M, E = self.num_envs, self.M, self.E
        r = float(self.cfg.hit_radius)

        # ---------- 0) 缓存步首状态 ----------
        fr_pos0 = self.fr_pos.clone()        # [N,M,3]
        en_pos0 = self.enemy_pos.clone()     # [N,E,3]
        fz0 = self.friend_frozen.clone()     # [N,M]
        ez0 = self.enemy_frozen.clone()      # [N,E]

        # ---------- 0.5) 步首命中判定 ----------
        active_pair0 = (~fz0).unsqueeze(2) & (~ez0).unsqueeze(1)  # [N,M,E]
        if active_pair0.any():
            diff0 = fr_pos0.unsqueeze(2) - en_pos0.unsqueeze(1)   # [N,M,E,3]
            dist0 = torch.linalg.norm(diff0, dim=-1)              # [N,M,E]
            hit_pair0 = (dist0 <= r) & active_pair0               # [N,M,E]

            fr_hit0 = hit_pair0.any(dim=2)  # [N,M]
            en_hit0 = hit_pair0.any(dim=1)  # [N,E]

            newly_fr = (~fz0) & fr_hit0
            newly_en = (~ez0) & en_hit0
            self._newly_frozen_friend |= newly_fr
            self._newly_frozen_enemy  |= newly_en

            if newly_en.any():
                self.enemy_capture_pos[newly_en] = en_pos0[newly_en]

            if newly_fr.any():
                INF = torch.tensor(float("inf"), device=self.device, dtype=dist0.dtype)
                dist_masked0 = torch.where(hit_pair0, dist0, INF)      # [N,M,E]
                j_star0 = dist_masked0.argmin(dim=2)                   # [N,M]
                batch_idx = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, M)  # [N,M]
                cap_for_friend0 = en_pos0[batch_idx, j_star0, :]       # [N,M,3]
                self.friend_capture_pos[newly_fr] = cap_for_friend0[newly_fr]

            self.friend_frozen |= fr_hit0
            self.enemy_frozen  |= en_hit0

        # 最新冻结掩码
        fz = self.friend_frozen
        ez = self.enemy_frozen

        # ---------- 1) 友机速度（直接采用命令，冻结=0） ----------
        fr_vel_w_step = torch.where(fz.unsqueeze(-1), torch.zeros_like(self.fr_cmd_vel), self.fr_cmd_vel)

        # ---------- 2) 敌机速度 ----------
        if self.cfg.enemy_seek_origin:
            if self._goal_e is None:
                self._rebuild_goal_e()
            goal = self._goal_e.unsqueeze(1).expand(-1, self.E, -1)     # [N,E,3]
            to_goal = goal - en_pos0                                    # [N,E,3]
            en_dir = to_goal / torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1e-6)
            enemy_vel_step = en_dir * float(self.cfg.enemy_speed)       # [N,E,3]
        else:
            enemy_vel_step = self.enemy_vel                              # [N,E,3]
        enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)

        # ---------- 3) 推进到步末 ----------
        fr_pos1 = fr_pos0 + fr_vel_w_step * dt   # [N,M,3]
        en_pos1 = en_pos0 + enemy_vel_step * dt  # [N,E,3]

        # ---------- 4) 覆盖冻结对象（位置=捕获点, 速度=0） ----------
        self.fr_vel_w = fr_vel_w_step
        self.enemy_vel = enemy_vel_step
        self.fr_pos = fr_pos1
        self.enemy_pos = en_pos1

        if fz.any():
            self.fr_vel_w[fz] = 0.0
            self.fr_pos[fz]   = self.friend_capture_pos[fz]
        if ez.any():
            self.enemy_vel[ez] = 0.0
            self.enemy_pos[ez] = self.enemy_capture_pos[ez]

        # ---------- 5) 刷新缓存 + 可视化 ----------
        self._refresh_enemy_cache()

        if self.friendly_visualizer is not None:
            self.friendly_visualizer.visualize(translations=self._flatten_agents(self.fr_pos))
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

        self._update_projection_debug_vis()

        if self._profile_print:
            self._cuda_sync_if_needed()
            print(f"[TIME] _apply_action (vel-only): {(time.perf_counter() - t0)*1000.0:.3f} ms")

    def _get_rewards(self) -> torch.Tensor:
        """
        奖励：
        - centroid_approach: 鼓励友机靠近“当前存活敌机”的质心（距离减小为正）。仅计未冻结友机
        - hit_bonus        : 以“本步新冻结的敌机”为准，一次性奖励
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        N, M = self.num_envs, self.M
        centroid_w  = float(getattr(self.cfg, "centroid_approach_weight", 1.0))
        hit_w       = float(getattr(self.cfg, "hit_reward_weight", 1000.0))

        friend_active     = (~self.friend_frozen)                   # [N,M]
        enemy_active_any  = self._enemy_active_any                  # [N]
        friend_active_f   = friend_active.float()
        centroid          = self._enemy_centroid                    # [N,3]

        c = centroid.unsqueeze(1).expand(N, M, 3)                   # [N,M,3]
        diff = c - self.fr_pos                                      # [N,M,3]
        dist_now = torch.linalg.norm(diff, dim=-1)                  # [N,M]

        if (not hasattr(self, "prev_dist_centroid")) or (self.prev_dist_centroid is None) \
           or (self.prev_dist_centroid.shape != dist_now.shape):
            self.prev_dist_centroid = dist_now.detach().clone()

        dist_now_safe = torch.where(enemy_active_any.unsqueeze(1), dist_now, self.prev_dist_centroid)
        d_delta_signed = self.prev_dist_centroid - dist_now_safe
        centroid_each = d_delta_signed * friend_active_f                  # [N,M]
        base_each = centroid_w * centroid_each
        reward = base_each.sum(dim=1)                                     # [N]

        new_hits_mask = self._newly_frozen_enemy                          # [N,E]
        hit_bonus = new_hits_mask.float().sum(dim=1) * hit_w              # [N]
        reward = reward + hit_bonus

        self._newly_frozen_enemy[:]  = False
        self._newly_frozen_friend[:] = False

        self.episode_sums.setdefault("centroid_approach", torch.zeros_like(centroid_each))
        self.episode_sums.setdefault("hit_bonus",         torch.zeros(self.num_envs, device=self.device, dtype=reward.dtype))
        self.episode_sums["centroid_approach"] += centroid_each
        self.episode_sums["hit_bonus"]         += hit_bonus

        self.prev_dist_centroid = dist_now_safe

        if self._profile_print:
            self._cuda_sync_if_needed()
            print(f"[TIME] _get_rewards: {(time.perf_counter() - t0)*1000.0:.3f} ms")

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 与原逻辑一致（略）
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        tol = float(getattr(self.cfg, "overshoot_tol", 0.4))
        r2_goal = float(self.cfg.enemy_goal_radius) ** 2
        xy_max2 = 25.0 ** 2

        if self._goal_e is None:
            self._rebuild_goal_e()

        success_all_enemies = self.enemy_frozen.all(dim=1)        # [N]

        z = self.fr_pos[..., 2]
        out_z_any = ((z < 0.0) | (z > 6.0)).any(dim=1)            # [N]

        origin_xy = self.terrain.env_origins[:, :2].unsqueeze(1)  # [N,1,2]
        dxy = self.fr_pos[..., :2] - origin_xy                    # [N,M,2]
        out_xy_any = (dxy.square().sum(dim=-1) > xy_max2).any(dim=1)  # [N]

        nan_inf_any = ~torch.isfinite(self.fr_pos).all(dim=(1, 2))     # [N]

        N = self.num_envs
        device = self.device
        enemy_goal_any = torch.zeros(N, dtype=torch.bool, device=device)
        overshoot_any  = torch.zeros(N, dtype=torch.bool, device=device)

        alive_mask = ~(success_all_enemies | out_z_any | out_xy_any | nan_inf_any)
        if alive_mask.any():
            idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)       # [K]

            diff_e = self.enemy_pos[idx] - self._goal_e[idx].unsqueeze(1)     # [K,E,3]
            enemy_goal_any[idx] = (diff_e.square().sum(dim=-1) < r2_goal).any(dim=1)

            friend_active = (~self.friend_frozen[idx])                  # [K,M]
            enemy_active  = self._enemy_active[idx]                     # [K,E]
            have_both = friend_active.any(dim=1) & enemy_active.any(dim=1)
            if have_both.any():
                k_idx = have_both.nonzero(as_tuple=False).squeeze(-1)
                centroid = self._enemy_centroid[idx][k_idx]             # [K2,3]
                gk = self._goal_e[idx][k_idx]                           # [K2,3]
                axis_hat = self._axis_hat[idx][k_idx]                   # [K2,3]

                sf = ((self.fr_pos[idx][k_idx]    - gk.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [K2,M]
                se = ((self.enemy_pos[idx][k_idx] - gk.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [K2,E]

                INF     = torch.tensor(float("inf"),  dtype=sf.dtype, device=sf.device)
                NEG_INF = torch.tensor(float("-inf"), dtype=sf.dtype, device=sf.device)
                sf_masked_for_min = torch.where(friend_active[k_idx], sf, INF)
                se_masked_for_max = torch.where(enemy_active[k_idx],  se, NEG_INF)
                friend_min = sf_masked_for_min.min(dim=1).values
                enemy_max  = se_masked_for_max.max(dim=1).values
                separated = friend_min > (enemy_max + tol)
                overshoot_any[idx[k_idx]] = separated

        died = out_z_any | out_xy_any | nan_inf_any | success_all_enemies | enemy_goal_any | overshoot_any
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        log_every = int(getattr(self.cfg, "log_termination_every", 1))
        if log_every and (int(self.episode_length_buf.max().item()) % log_every == 0):
            term = self.extras.setdefault("termination", {})
            stats = torch.stack([
                success_all_enemies.sum(),
                self.enemy_frozen.sum(),
                (out_z_any | out_xy_any).sum(),
                nan_inf_any.sum(),
                enemy_goal_any.sum(),
                overshoot_any.sum(),
                time_out.sum(),
            ]).to("cpu")
            term.update({
                "success_envs":      int(stats[0]),
                "hit_total_enemies": int(stats[1]),
                "out_of_bounds_any": int(stats[2]),
                "nan_inf_any":       int(stats[3]),
                "enemy_goal_any":    int(stats[4]),
                "overshoot_any":     int(stats[5]),
                "time_out":          int(stats[6]),
            })

        if self._profile_print:
            self._cuda_sync_if_needed()
            print(f"[TIME] _get_dones: {(time.perf_counter() - t0)*1000.0:.3f} ms")

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        重置指定 env：
            - 清零统计/缓存/冻结
            - 友机并排生成（Y 方向）
            - 敌机生成 & 初速度
            - 初始化“上一帧质心距离”缓存
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()
        if not hasattr(self, "terrain"):
            self._setup_scene()
        if self._goal_e is None:
            self._rebuild_goal_e()
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        N = len(env_ids)
        M = self.M
        dev = self.device
        origins = self.terrain.env_origins[env_ids]  # [N,3]

        for k in list(self.episode_sums.keys()):
            self.episode_sums[k][env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        self.friend_frozen[env_ids] = False
        self.enemy_frozen[env_ids]  = False
        self.friend_capture_pos[env_ids] = 0.0
        self.enemy_capture_pos[env_ids]  = 0.0
        self._newly_frozen_friend[env_ids] = False
        self._newly_frozen_enemy[env_ids]  = False

        # 友方并排生成（固定沿 Y 方向）
        spacing = float(getattr(self.cfg, "formation_spacing", 0.8))
        idx = torch.arange(M, device=dev).float() - (M - 1) / 2.0
        offsets_xy = torch.stack([torch.zeros_like(idx), idx * spacing], dim=-1)  # [M,2]
        offsets_xy = offsets_xy.unsqueeze(0).expand(N, M, 2)                      # [N,M,2]
        fr0 = torch.empty(N, M, 3, device=dev)
        fr0[..., :2] = origins[:, :2].unsqueeze(1) + offsets_xy
        fr0[...,  2] = origins[:, 2].unsqueeze(1) + float(self.cfg.flight_altitude)
        self.fr_pos[env_ids]   = fr0
        self.fr_vel_w[env_ids] = 0.0
        self.fr_cmd_vel[env_ids] = 0.0

        # 敌机在圆盘内随机出生（带最小间隔约束）
        self._spawn_enemy(env_ids)

        # 敌机初速度（环向寻标或常值）
        phi = torch.rand(N, device=dev) * 2.0 * math.pi
        spd = float(self.cfg.enemy_speed)
        self.enemy_vel[env_ids, :, 0] = spd * torch.cos(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 1] = spd * torch.sin(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 2] = 0.0

        # 初始化“友机到活敌质心”的上一帧距离缓存
        enemy_active   = (~self.enemy_frozen[env_ids])                        # [N,E]
        e_mask         = enemy_active.float().unsqueeze(-1)                   # [N,E,1]
        sum_pos        = (self.enemy_pos[env_ids] * e_mask).sum(dim=1)        # [N,3]
        cnt            = e_mask.sum(dim=1).clamp_min(1.0)                     # [N,1]
        centroid       = sum_pos / cnt                                        # [N,3]
        c              = centroid.unsqueeze(1).expand(-1, self.M, 3)          # [N,M,3]
        dist0          = torch.linalg.norm(c - self.fr_pos[env_ids], dim=-1)  # [N,M]
        if not hasattr(self, "prev_dist_centroid") or self.prev_dist_centroid is None \
           or self.prev_dist_centroid.shape != (self.num_envs, self.M):
            self.prev_dist_centroid = torch.zeros(self.num_envs, self.M, device=self.device)
        self.prev_dist_centroid[env_ids] = dist0

        self._refresh_enemy_cache()

        if self.friendly_visualizer is not None:
            self.friendly_visualizer.visualize(translations=self._flatten_agents(self.fr_pos))
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

        if self._profile_print:
            self._cuda_sync_if_needed()
            print(f"[TIME] _reset_idx: {(time.perf_counter() - t0)*1000.0:.3f} ms")

    def _get_observations(self) -> dict:
        """
        观测：
            对每个友机：
                [ fr_pos(3) | fr_vel_w(3) | e_hat_to_all_enemies(3*E) ]
            其中“已冻结敌机”的方向置零，维度不变。
            拼成 [N, M * (6 + 3E)]
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        eps = 1e-6
        N, M, E = self.num_envs, self.M, self.E

        rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)                # [N,M,E,3]
        dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)      # [N,M,E,1]
        e_hat_all = rel_all / dist_all                                                  # [N,M,E,3]

        if hasattr(self, "enemy_frozen") and self.enemy_frozen is not None:
            enemy_active = (~self.enemy_frozen).unsqueeze(1).unsqueeze(-1).float()      # [N,1,E,1]
            e_hat_all = e_hat_all * enemy_active

        e_hat_flat = e_hat_all.reshape(N, M, 3 * E)                                     # [N,M,3E]
        obs_each = torch.cat([self.fr_pos, self.fr_vel_w, e_hat_flat], dim=-1)          # [N,M,6+3E]
        obs = obs_each.reshape(N, -1)

        if self._profile_print:
            self._cuda_sync_if_needed()
            print(f"[TIME] _get_observations: {(time.perf_counter() - t0)*1000.0:.3f} ms")

        return {"policy": obs, "odom": obs.clone()}

# ---------------- Gym 注册 ----------------
from config import agents

gym.register(
    id="FAST-Intercept-Swarm-Test",
    entry_point=FastInterceptionSwarmEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FastInterceptionSwarmEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:quadcopter_sb3_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:Loitering_Munition_interception_swarm_skrl_ppo_cfg.yaml",
    },
)
