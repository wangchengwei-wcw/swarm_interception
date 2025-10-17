# Loitering_Munition_interception_swarm.py
# MARL 版：使用 DirectMARLEnv / DirectMARLEnvCfg，按 agent 字典进行交互
from __future__ import annotations

import math
import torch
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
from isaaclab.utils import configclass
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, ViewerCfg
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
    xm = vec_m[..., 0]
    ym = vec_m[..., 1]
    zm = vec_m[..., 2]
    return torch.stack([xm, -zm, ym], dim=-1)

def z_up_to_y_up(vec_w: torch.Tensor) -> torch.Tensor:
    xw = vec_w[..., 0]
    yw = vec_w[..., 1]
    zw = vec_w[..., 2]
    return torch.stack([xw, zw, -yw], dim=-1)


@configclass
class FastInterceptionSwarmMARLCfg(DirectMARLEnvCfg):
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # ---------- 数量控制 ----------
    swarm_size: int = 5                 # 便捷参数：同时设置友机/敌机数量
    friendly_size: int | None = None    # 若为 None 就用 swarm_size
    enemy_size: int | None = None

    # 敌机出生区域（圆盘）与最小间隔
    debug_vis_enemy = True
    enemy_height_min = 3.0
    enemy_height_max = 3.0
    enemy_speed = 1.5
    enemy_seek_origin = True
    enemy_target_alt = 3.0
    enemy_goal_radius = 0.5
    enemy_cluster_ring_radius: float = 25.0
    enemy_cluster_radius: float = 6.0
    enemy_min_separation: float = 4.0

    # 友方控制/速度范围/位置间隔
    Vm_min = 1.5
    Vm_max = 3.0
    ny_max_g = 3.0
    nz_max_g = 3.0
    formation_spacing = 0.8
    flight_altitude = 0.2

    # —— 单 agent 观测/动作维（用于 MARL 的 per-agent 空间）——
    single_observation_space: int = 9     # 将在 __post_init__ 基于 E 自动覆盖为 6 + 3E
    single_action_space: int = 3

    # —— Multi-agent 所需的字典空间（在 __post_init__ 填充）——
    possible_agents: list[str] | None = None
    action_spaces: dict[str, int] | None = None
    observation_spaces: dict[str, int] | None = None

    # 奖励相关
    hit_radius = 1.0
    centroid_approach_weight = 1.0
    hit_reward_weight: float = 1000.0
    heading_align_weight = 0.5  # 目前未启用

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

    # —— 关键：类级别一定要是“可序列化”的 Gym Space（占位即可）——
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space      = spaces.Box(low=-1.0,   high=1.0,   shape=(1,), dtype=np.float32)
    state_space       = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    clip_action       = 1.0

    # 多智能体的 spaces：类级别给“空字典”，让序列化器能安全遍历（得到 {}）
    possible_agents: list[str] = []
    action_spaces: dict[str, gym.Space] = {}
    observation_spaces: dict[str, gym.Space] = {}

    # 单 agent 维度描述（实例期会覆盖）
    single_action_space: int = 3
    single_observation_space: int = 9

    def __post_init__(self):
        M = self.friendly_size if self.friendly_size is not None else self.swarm_size
        E = self.enemy_size    if self.enemy_size    is not None else self.swarm_size
        self.single_observation_space = 6 + 3 * int(E)
        # 注意：不要在这里填 action_spaces/observation_spaces 的 dict，留到 Env.__init__()
        return


class FastInterceptionSwarmMARLEnv(DirectMARLEnv):
    """多智能体（分布式）版拦截环境：按 agent 字典进行 obs/action/reward/done 交互。"""
    cfg: FastInterceptionSwarmMARLCfg
    _is_closed = True

    # ---------- ↓↓↓ 原文件中的成员与逻辑保持一致（只改成 MARL 交互） ↓↓↓ ----------
    def __init__(self, cfg: FastInterceptionSwarmMARLCfg, render_mode: str | None = None, **kwargs):
        M = cfg.friendly_size if cfg.friendly_size is not None else cfg.swarm_size
        E = cfg.enemy_size    if cfg.enemy_size    is not None else cfg.swarm_size

        single_obs_dim = 6 + 3 * E
        cfg.single_observation_space = single_obs_dim

        # —— 真正的 multi-agent 空间（按 agent 填充成Gym Space）——
        agents = [f"drone_{i}" for i in range(M)]
        cfg.possible_agents = agents
        cfg.action_spaces = {a: spaces.Box(low=-1.0, high=1.0, shape=(cfg.single_action_space,), dtype=np.float32)
                             for a in agents}
        cfg.observation_spaces = {a: spaces.Box(low=-np.inf, high=np.inf, shape=(single_obs_dim,), dtype=np.float32)
                                  for a in agents}

        # 可选：集中式 state（供 MAPPO 等用），设为拼接后的维度
        cfg.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(M * single_obs_dim,), dtype=np.float32)

        # 单智能体那三个占位保持即可，不再使用；也可以同步成：
        cfg.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(cfg.single_action_space,), dtype=np.float32)
        cfg.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(single_obs_dim,), dtype=np.float32)

        super().__init__(cfg, render_mode, **kwargs)
        self._is_closed = False

        # 关键：把“多智能体标识 & 空间字典”挂到 env 实例上，便于 SKRL 检测
        self.is_multi_agent = True                     # 一些包装器会看这个
        self.possible_agents = cfg.possible_agents     # list[str]
        self.action_spaces = cfg.action_spaces         # dict[str, gym.Space]
        self.observation_spaces = cfg.observation_spaces  # dict[str, gym.Space]
        # 如果你有集中式状态（MAPPO 用），也一并挂上（非必须）
        if getattr(cfg, "state_space", None) is not None:
            self.state_space = cfg.state_space

        # ---------- 维度 ----------
        self.M = int(M)  # 友机数（= agents 数）
        self.E = int(E)  # 敌机数（=M）
        N = self.num_envs
        dev = self.device
        dtype = torch.float32

        # 友机状态 [N,M,3]
        self.fr_pos = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)
        self.fr_vel_w = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)

        # 敌机状态 [N,E,3]
        self.enemy_pos = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)
        self.enemy_vel = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)

        # 友机动力学（y-up）[N,M]
        self.g0 = 9.81
        self.theta = torch.zeros(N, self.M, device=dev, dtype=dtype)
        self.psi_v = torch.zeros(N, self.M, device=dev, dtype=dtype)
        self.Vm = torch.zeros(N, self.M, device=dev, dtype=dtype)
        self._ny = torch.zeros(N, self.M, device=dev, dtype=dtype)
        self._nz = torch.zeros(N, self.M, device=dev, dtype=dtype)

        # —— 冻结掩码与命中位置（全对全）——
        self.friend_frozen = torch.zeros(N, self.M, device=dev, dtype=torch.bool)
        self.enemy_frozen  = torch.zeros(N, self.E, device=dev, dtype=torch.bool)
        self.friend_capture_pos = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)
        self.enemy_capture_pos  = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)

        # 统计/动作缓存
        self.episode_sums = {}
        self.episode_sums["align"]          = torch.zeros(self.num_envs, self.M, device=dev)
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
        self._profile_print = bool(getattr(self.cfg, "profile_print", False))
        self._enemy_centroid_init = torch.zeros(self.num_envs, 3, device=self.device, dtype=self.fr_pos.dtype)
        # --- 缓存（每步只更新一次）---
        self._enemy_centroid = torch.zeros(self.num_envs, 3, device=dev, dtype=dtype)
        self._enemy_active = torch.zeros(self.num_envs, self.E, device=dev, dtype=torch.bool)
        self._enemy_active_any = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        self._goal_e = None
        self._axis_hat = torch.zeros(self.num_envs, 3, device=dev, dtype=dtype)

        # 为 MARL 方便：保存 agent 名称顺序
        self.possible_agents = list(self.cfg.possible_agents)

    # —————————————————— ↓↓↓↓↓工具/可视化区（与原版一致）↓↓↓↓↓ ——————————————————
    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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
        half = 0.5 * psi
        return torch.stack([torch.cos(half), torch.zeros_like(psi), torch.sin(half), torch.zeros_like(psi)], dim=-1)

    def _qz(self, theta: torch.Tensor) -> torch.Tensor:
        half = 0.5 * theta
        return torch.stack([torch.cos(half), torch.zeros_like(theta), torch.zeros_like(theta), torch.sin(half)], dim=-1)

    def _qx_plus_90(self, *shape_prefix) -> torch.Tensor:
        cx = math.sqrt(0.5)
        sx = cx
        base = torch.tensor([cx, sx, 0.0, 0.0], device=self.device, dtype=self.fr_pos.dtype)
        if len(shape_prefix) == 0:
            return base
        rep = int(torch.tensor(shape_prefix).prod().item())
        return base.repeat(rep, 1).reshape(*shape_prefix, 4)

    def _friendly_world_quats(self) -> torch.Tensor:
        q_m = self._quat_mul(self._qy(self.psi_v), self._qz(self.theta))
        q_w = self._quat_mul(self._qx_plus_90(self.num_envs, self.M), q_m)
        return self._quat_normalize(q_w)

    def _flatten_agents(self, X: torch.Tensor) -> torch.Tensor:
        return X.reshape(-1, X.shape[-1])

    def close(self):
        if getattr(self, "_is_closed", True):
            return
        super().close()
        self._is_closed = True

    def _rebuild_goal_e(self):
        origins = self.terrain.env_origins
        self._goal_e = torch.stack(
            [origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)],
            dim=-1
        )

    def _refresh_enemy_cache(self):
        if self._goal_e is None:
            self._rebuild_goal_e()

        enemy_active = ~self.enemy_frozen
        e_mask = enemy_active.unsqueeze(-1).float()
        sum_pos = (self.enemy_pos * e_mask).sum(dim=1)
        cnt     = e_mask.sum(dim=1).clamp_min(1.0)
        centroid = sum_pos / cnt

        self._enemy_centroid = centroid
        self._enemy_active   = enemy_active
        self._enemy_active_any = enemy_active.any(dim=1)

        axis = centroid - self._goal_e
        norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._axis_hat = axis / norm

    # —— 敌机生成（保持与原版一致）——
    def _spawn_enemy(self, env_ids: torch.Tensor):
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        dev = self.device
        env_ids = env_ids.to(dtype=torch.long, device=dev)
        N = env_ids.numel()
        if N == 0:
            return

        E = self.E
        R_big   = float(self.cfg.enemy_cluster_ring_radius)
        r_small = float(self.cfg.enemy_cluster_radius)
        s_min   = float(self.cfg.enemy_min_separation)
        hmin    = float(self.cfg.enemy_height_min)
        hmax    = float(self.cfg.enemy_height_max)

        eta = float(getattr(self.cfg, "enemy_poisson_eta", 0.7))

        origins_all = self.terrain.env_origins
        if origins_all.device != dev:
            origins_all = origins_all.to(dev)
        origins = origins_all[env_ids]

        two_pi = 2.0 * math.pi
        theta = two_pi * torch.rand(N, device=dev)
        centers = origins[:, :2] + R_big * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

        r_needed = 0.5 * s_min * math.sqrt(E / max(eta, 1e-6))
        r_env = torch.full((N,), max(r_small, r_needed * 1.02), device=dev)

        s2 = s_min * s_min
        pts = torch.zeros(N, E, 2, device=dev)
        filled = torch.zeros(N, dtype=torch.long, device=dev)
        stagn  = torch.zeros(N, dtype=torch.long, device=dev)
        BATCH = 128
        MAX_ROUNDS = 256
        GROW_FACTOR = 1.05
        STAGN_ROUNDS = 5
        ar_e = torch.arange(E, device=dev)

        for _ in range(MAX_ROUNDS):
            if (filled >= E).all():
                break

            u = torch.rand(N, BATCH, device=dev)
            v = torch.rand(N, BATCH, device=dev)
            rr  = r_env.unsqueeze(1) * torch.sqrt(u.clamp_min(1e-12))
            ang = two_pi * v
            cand = centers[:, None, :] + torch.stack([rr * torch.cos(ang),
                                                    rr * torch.sin(ang)], dim=-1)

            diff_valid = (ar_e.unsqueeze(0) < filled.unsqueeze(1)).unsqueeze(1)
            pts_eff = torch.where(diff_valid.unsqueeze(-1), pts[:, None, :, :], cand[:, :, None, :])
            diff = cand[:, :, None, :] - pts_eff
            sq   = (diff ** 2).sum(dim=-1).masked_fill(~diff_valid, float("inf"))
            min_sq, _ = sq.min(dim=-1)
            ok = min_sq >= s2

            idxs = torch.arange(BATCH, device=dev).unsqueeze(0).expand(N, -1)
            first_idx = torch.where(ok, idxs, torch.full_like(idxs, BATCH)).min(dim=1).values
            can_take = (first_idx < BATCH) & (filled < E)

            env_take = torch.nonzero(can_take, as_tuple=False).squeeze(1)
            if env_take.numel() > 0:
                pos = filled[env_take]
                pts[env_take, pos, :] = cand[env_take, first_idx[env_take], :]
                filled[env_take] += 1
                stagn[env_take] = 0

            stagn[~can_take] += 1
            grow_mask = stagn >= STAGN_ROUNDS
            if grow_mask.any():
                r_env[grow_mask] *= GROW_FACTOR
                stagn[grow_mask] = 0

        if (filled < E).any():
            EXTRA_GROW_STEPS = 8
            for _ in range(EXTRA_GROW_STEPS):
                need_mask = filled < E
                if not need_mask.any():
                    break
                r_env[need_mask] *= GROW_FACTOR

                u = torch.rand(N, BATCH, device=dev)
                v = torch.rand(N, BATCH, device=dev)
                rr  = r_env.unsqueeze(1) * torch.sqrt(u)
                ang = two_pi * v
                cand = centers[:, None, :] + torch.stack([rr * torch.cos(ang),
                                                        rr * torch.sin(ang)], dim=-1)

                diff  = cand[:, :, None, :] - pts[:, None, :, :]
                valid = (ar_e.unsqueeze(0) < filled.unsqueeze(1)).unsqueeze(1)
                sq    = (diff ** 2).sum(dim=-1).masked_fill(~valid, float("inf"))
                min_sq, _ = sq.min(dim=-1)
                ok = min_sq >= s2

                idxs = torch.arange(BATCH, device=dev).unsqueeze(0).expand(N, -1)
                first_idx = torch.where(ok, idxs, torch.full_like(idxs, BATCH)).min(dim=1).values
                can_take = (first_idx < BATCH) & (filled < E)

                env_take = torch.nonzero(can_take, as_tuple=False).squeeze(1)
                if env_take.numel() > 0:
                    pos = filled[env_take]
                    pts[env_take, pos, :] = cand[env_take, first_idx[env_take], :]
                    filled[env_take] += 1

            if (filled < E).any():
                not_full = int((filled < E).sum().item())
                raise RuntimeError(
                    f"Poisson sampling failed after radius growth for {not_full}/{N} envs. "
                    f"Consider increasing enemy_cluster_radius or reducing enemy_min_separation/E."
                )

        # 写回
        pts = torch.nan_to_num(pts)
        self.enemy_pos[env_ids, :, 0:2] = pts
        z = origins[:, 2:3].unsqueeze(1) + (hmin + torch.rand(N, E, 1, device=dev) * (hmax - hmin))
        self.enemy_pos[env_ids, :, 2:3] = z

        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _spawn_enemy : {dt_ms:.3f} ms")

    def _build_ray_dots(self, c: torch.Tensor, g: torch.Tensor, step: float) -> torch.Tensor:
        dev = self.device
        v = g - c
        L = torch.linalg.norm(v).item()
        if L < 1e-6:
            return c.unsqueeze(0)
        n = max(2, int(L / max(step, 1e-6)) + 1)
        ts = torch.linspace(0.0, 1.0, n, device=dev, dtype=torch.float32).unsqueeze(1)
        pts = c.unsqueeze(0) + ts * v.unsqueeze(0)
        return pts

    def _update_projection_debug_vis(self):
        if not getattr(self.cfg, "proj_vis_enable", False):
            return
        if self._goal_e is None:
            self._rebuild_goal_e()

        dev = self.device
        N_draw = int(min(self.num_envs, getattr(self.cfg, "proj_max_envs", 8)))
        if N_draw <= 0:
            return

        centroid_pts, ray_pts, fr_proj_pts, en_proj_pts = [], [], [], []
        for ei in range(N_draw):
            enemy_active = self._enemy_active[ei]
            if not enemy_active.any():
                continue

            centroid = self._enemy_centroid[ei]
            g = self._goal_e[ei]
            ray_pts.append(self._build_ray_dots(centroid, g, float(self.cfg.proj_ray_step)))
            centroid_pts.append(centroid.unsqueeze(0))

            axis_hat = self._axis_hat[ei]

            friend_active = (~self.friend_frozen[ei])
            fr_pos = self.fr_pos[ei]
            s_f = ((fr_pos - centroid.unsqueeze(0)) * axis_hat.unsqueeze(0)).sum(dim=-1)
            p_f = centroid.unsqueeze(0) + s_f.unsqueeze(1) * axis_hat.unsqueeze(0)
            fr_proj_pts.append(p_f[friend_active])

            en_pos = self.enemy_pos[ei]
            s_e = ((en_pos - centroid.unsqueeze(0)) * axis_hat.unsqueeze(0)).sum(dim=-1)
            p_e = centroid.unsqueeze(0) + s_e.unsqueeze(1) * axis_hat.unsqueeze(0)
            en_proj_pts.append(p_e[enemy_active])

        if len(centroid_pts) > 0 and self.centroid_marker is not None:
            self.centroid_marker.visualize(translations=torch.cat(centroid_pts, dim=0))
        if len(ray_pts) > 0 and self.ray_marker is not None:
            self.ray_marker.visualize(translations=torch.cat(ray_pts, dim=0))
        if len(fr_proj_pts) > 0 and self.friend_proj_marker is not None:
            self.friend_proj_marker.visualize(translations=torch.cat(fr_proj_pts, dim=0) if fr_proj_pts else torch.empty(0,3,device=dev))
        if len(en_proj_pts) > 0 and self.enemy_proj_marker is not None:
            self.enemy_proj_marker.visualize(translations=torch.cat(en_proj_pts, dim=0) if en_proj_pts else torch.empty(0,3,device=dev))

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
        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
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

    # —————————————————— ↑↑↑ 工具/可视化区 ↑↑↑ ——————————————————

    # ============================ MARL 交互实现 ============================

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """
        接收字典动作：每个 agent 一个 [num_envs, 3] 向量（ny, nz, throttle）。
        已冻结友机的动作会被屏蔽。
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        if actions is None:
            return
        N, M = self.num_envs, self.M

        # 组装到 [N, M, 3]
        act = torch.zeros(N, M, 3, device=self.device, dtype=self.fr_pos.dtype)
        for i, agent in enumerate(self.possible_agents):
            a = actions[agent].to(self.device)  # [N,3]
            act[:, i, :] = a

        # 屏蔽冻结
        if hasattr(self, "friend_frozen") and self.friend_frozen is not None:
            active_mask_f = (~self.friend_frozen).float().unsqueeze(-1)  # [N,M,1]
            act = act * active_mask_f

        # 规范化与映射
        ny = act[..., 0].clamp(-1.0, 1.0)
        nz = act[..., 1].clamp(-1.0, 1.0)
        throttle = act[..., 2].clamp(0.0, 1.0)

        self._ny = ny * self.cfg.ny_max_g
        self._nz = nz * self.cfg.nz_max_g
        self.Vm = self.cfg.Vm_min + throttle * (self.cfg.Vm_max - self.cfg.Vm_min)

        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _pre_physics_step: {dt_ms:.3f} ms")

    def _apply_action(self):
        """
        与原版一致：步首命中→冻结→推进→写回→缓存更新/可视化
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

        fr_pos0 = self.fr_pos.clone()
        en_pos0 = self.enemy_pos.clone()
        fz0 = self.friend_frozen.clone()
        ez0 = self.enemy_frozen.clone()

        # 步首命中
        active_pair0 = (~fz0).unsqueeze(2) & (~ez0).unsqueeze(1)
        if active_pair0.any():
            diff0 = fr_pos0.unsqueeze(2) - en_pos0.unsqueeze(1)
            dist0 = torch.linalg.norm(diff0, dim=-1)
            hit_pair0 = (dist0 <= r) & active_pair0

            fr_hit0 = hit_pair0.any(dim=2)
            en_hit0 = hit_pair0.any(dim=1)

            newly_fr = (~fz0) & fr_hit0
            newly_en = (~ez0) & en_hit0
            self._newly_frozen_friend |= newly_fr
            self._newly_frozen_enemy  |= newly_en

            if newly_en.any():
                self.enemy_capture_pos[newly_en] = en_pos0[newly_en]

            if newly_fr.any():
                INF = torch.tensor(float("inf"), device=self.device, dtype=dist0.dtype)
                dist_masked0 = torch.where(hit_pair0, dist0, INF)
                j_star0 = dist_masked0.argmin(dim=2)
                batch_idx = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, M)
                cap_for_friend0 = en_pos0[batch_idx, j_star0, :]
                self.friend_capture_pos[newly_fr] = cap_for_friend0[newly_fr]

            self.friend_frozen |= fr_hit0
            self.enemy_frozen  |= en_hit0

        fz = self.friend_frozen
        ez = self.enemy_frozen

        # 友机姿态/速度（冻结为0）
        cos_th_now = torch.cos(self.theta).clamp_min(1e-6)
        Vm_eff = torch.where(fz, torch.zeros_like(self.Vm), self.Vm)
        Vm_eps = Vm_eff.clamp_min(1e-6)
        theta_rate = self.g0 * (self._ny - cos_th_now) / Vm_eps
        psi_rate   = - self.g0 * self._nz / (Vm_eps * cos_th_now)
        theta_rate = torch.where(fz, torch.zeros_like(theta_rate), theta_rate)
        psi_rate   = torch.where(fz, torch.zeros_like(psi_rate),   psi_rate)

        THETA_RATE_LIMIT = 1.0
        PSI_RATE_LIMIT   = 1.0
        theta_rate = torch.clamp(theta_rate, -THETA_RATE_LIMIT, THETA_RATE_LIMIT)
        psi_rate   = torch.clamp(psi_rate,   -PSI_RATE_LIMIT,   PSI_RATE_LIMIT)

        self.theta = self.theta + theta_rate * dt
        self.psi_v = (self.psi_v + psi_rate * dt + math.pi) % (2.0 * math.pi) - math.pi

        sin_th, cos_th = torch.sin(self.theta), torch.cos(self.theta)
        sin_ps, cos_ps = torch.sin(self.psi_v), torch.cos(self.psi_v)
        Vxm = Vm_eff * cos_th * cos_ps
        Vym = Vm_eff * sin_th
        Vzm = -Vm_eff * cos_th * sin_ps
        V_m = torch.stack([Vxm, Vym, Vzm], dim=-1)
        fr_vel_w_step = y_up_to_z_up(V_m)

        badv = ~torch.isfinite(fr_vel_w_step).all(dim=-1)
        if badv.any():
            nidx, midx = badv.nonzero(as_tuple=False)[0].tolist()
            print(f"[NaN vel] env={nidx} agent={midx}",
                f"Vm={self.Vm[nidx, midx].item():.6g}",
                f"ny={self._ny[nidx, midx].item():.6g}",
                f"nz={self._nz[nidx, midx].item():.6g}",
                f"theta={self.theta[nidx, midx].item():.6g}",
                f"psi={self.psi_v[nidx, midx].item():.6g}",
                f"fr_vel_w_step={fr_vel_w_step[nidx, midx]}")

        # 敌机速度
        if self.cfg.enemy_seek_origin:
            if self._goal_e is None:
                self._rebuild_goal_e()
            goal = self._goal_e.unsqueeze(1).expand(-1, self.E, -1)
            to_goal = goal - en_pos0
            en_dir = to_goal / torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1e-6)
            enemy_vel_step = en_dir * float(self.cfg.enemy_speed)
        else:
            enemy_vel_step = self.enemy_vel
        enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)

        # 推进
        fr_pos1 = fr_pos0 + fr_vel_w_step * dt
        en_pos1 = en_pos0 + enemy_vel_step * dt

        badp = ~torch.isfinite(fr_pos1).all(dim=-1)
        if badp.any():
            nidx, midx = badp.nonzero(as_tuple=False)[0].tolist()
            print(f"[NaN pos] env={nidx} agent={midx}",
                f"fr_pos0={fr_pos0[nidx, midx]}",
                f"fr_vel_w_step={fr_vel_w_step[nidx, midx]}",
                f"dt={dt}")
            fr_pos1[nidx, midx] = fr_pos0[nidx, midx]

        # 冻结写回
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

        # 刷新缓存 + 可视化
        self._refresh_enemy_cache()

        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

        self._update_projection_debug_vis()

        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _apply_action (pre-step 1m check): {dt_ms:.3f} ms")

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """
        为每个 agent 返回一个 reward 向量（长度 num_envs）。
        这里将“命中奖励”在所有友机间均分（也可替换成只奖新命中的友机）。
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        N, M = self.num_envs, self.M

        centroid_w  = float(getattr(self.cfg, "centroid_approach_weight", 1.0))
        hit_w       = float(getattr(self.cfg, "hit_reward_weight", 1000.0))

        friend_active     = (~self.friend_frozen)
        enemy_active_any  = self._enemy_active_any
        centroid          = self._enemy_centroid

        c = centroid.unsqueeze(1).expand(N, M, 3)
        diff = c - self.fr_pos
        dist_now = torch.linalg.norm(diff, dim=-1)

        if (not hasattr(self, "prev_dist_centroid")) or (self.prev_dist_centroid is None) \
        or (self.prev_dist_centroid.shape != dist_now.shape):
            self.prev_dist_centroid = dist_now.detach().clone()

        dist_now_safe = torch.where(enemy_active_any.unsqueeze(1), dist_now, self.prev_dist_centroid)
        d_delta_signed = self.prev_dist_centroid - dist_now_safe
        centroid_each = d_delta_signed * friend_active.float()

        # 命中奖励：按“本步新冻结的敌机”计数，均分给 M 个友机
        new_hits_mask = self._newly_frozen_enemy
        hit_bonus_env = new_hits_mask.float().sum(dim=1) * hit_w  # [N]
        per_agent_hit = (hit_bonus_env / max(M, 1)).unsqueeze(1).expand(-1, M)  # [N,M]

        # 合成每个 agent 的 reward
        rewards = {}
        for i, agent in enumerate(self.possible_agents):
            r_i = centroid_w * centroid_each[:, i] + per_agent_hit[:, i]
            rewards[agent] = r_i

        # 统计（可选）
        self.episode_sums.setdefault("centroid_approach", torch.zeros_like(centroid_each))
        self.episode_sums.setdefault("hit_bonus",         torch.zeros(self.num_envs, device=self.device, dtype=hit_bonus_env.dtype))
        self.episode_sums["centroid_approach"] += centroid_each
        self.episode_sums["hit_bonus"]         += hit_bonus_env

        self.prev_dist_centroid = dist_now_safe

        # 命中标记用过就清（保持一次性）
        self._newly_frozen_enemy[:]  = False
        self._newly_frozen_friend[:] = False

        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _get_rewards: {dt_ms:.3f} ms")

        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        为每个 agent 返回统一的 done / time_out 掩码（与 swarm_vel_env 的风格一致）。
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        tol = float(getattr(self.cfg, "overshoot_tol", 0.4))
        r2_goal = float(self.cfg.enemy_goal_radius) ** 2
        xy_max2 = 25.0 ** 2

        if self._goal_e is None:
            self._rebuild_goal_e()

        success_all_enemies = self.enemy_frozen.all(dim=1)

        z = self.fr_pos[..., 2]
        out_z_any = ((z < 0.0) | (z > 6.0)).any(dim=1)

        origin_xy = self.terrain.env_origins[:, :2].unsqueeze(1)
        dxy = self.fr_pos[..., :2] - origin_xy
        out_xy_any = (dxy.square().sum(dim=-1) > xy_max2).any(dim=1)

        nan_inf_any = ~torch.isfinite(self.fr_pos).all(dim=(1, 2))

        N = self.num_envs
        device = self.device
        enemy_goal_any = torch.zeros(N, dtype=torch.bool, device=device)
        overshoot_any  = torch.zeros(N, dtype=torch.bool, device=device)

        alive_mask = ~(success_all_enemies | out_z_any | out_xy_any | nan_inf_any)
        if alive_mask.any():
            idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)

            diff_e = self.enemy_pos[idx] - self._goal_e[idx].unsqueeze(1)
            enemy_goal_any[idx] = (diff_e.square().sum(dim=-1) < r2_goal).any(dim=1)

            friend_active = (~self.friend_frozen[idx])
            enemy_active  = self._enemy_active[idx]
            have_both = friend_active.any(dim=1) & enemy_active.any(dim=1)
            if have_both.any():
                k_idx = have_both.nonzero(as_tuple=False).squeeze(-1)
                centroid = self._enemy_centroid[idx][k_idx]
                gk = self._goal_e[idx][k_idx]
                axis_hat = self._axis_hat[idx][k_idx]

                sf = ((self.fr_pos[idx][k_idx]    - gk.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)
                se = ((self.enemy_pos[idx][k_idx] - gk.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)

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

        dones = {agent: died for agent in self.possible_agents}
        truncs = {agent: time_out for agent in self.possible_agents}

        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _get_dones: {dt_ms:.3f} ms")

        return dones, truncs

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        与原版一致，但无机器人关节，保留粒子动力学/可视化。
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
        origins = self.terrain.env_origins[env_ids]

        # 清零 episode 统计
        for k in list(self.episode_sums.keys()):
            self.episode_sums[k][env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        # 清空冻结状态与捕获点
        self.friend_frozen[env_ids] = False
        self.enemy_frozen[env_ids]  = False
        self.friend_capture_pos[env_ids] = 0.0
        self.enemy_capture_pos[env_ids]  = 0.0
        self._newly_frozen_friend[env_ids] = False
        self._newly_frozen_enemy[env_ids]  = False

        # 友方并排沿 Y 生成
        spacing = float(getattr(self.cfg, "formation_spacing", 0.8))
        idx = torch.arange(M, device=dev).float() - (M - 1) / 2.0
        offsets_xy = torch.stack([torch.zeros_like(idx), idx * spacing], dim=-1)
        offsets_xy = offsets_xy.unsqueeze(0).expand(N, M, 2)
        fr0 = torch.empty(N, M, 3, device=dev)
        fr0[..., :2] = origins[:, :2].unsqueeze(1) + offsets_xy
        fr0[...,  2] = origins[:, 2].unsqueeze(1) + float(self.cfg.flight_altitude)
        self.fr_pos[env_ids]  = fr0
        self.fr_vel_w[env_ids] = 0.0

        # 敌机生成
        self._spawn_enemy(env_ids)

        # 敌机初速度（环向）
        phi = torch.rand(N, device=dev) * 2.0 * math.pi
        spd = float(self.cfg.enemy_speed)
        self.enemy_vel[env_ids, :, 0] = spd * torch.cos(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 1] = spd * torch.sin(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 2] = 0.0

        # 友机朝向质心
        self.Vm[env_ids] = 0.0
        en_pos = self.enemy_pos[env_ids]
        centroid = en_pos.mean(dim=1)
        rel_w = centroid.unsqueeze(1) - self.fr_pos[env_ids]
        rel_m = z_up_to_y_up(rel_w)
        rel_m = rel_m / rel_m.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        sin_th = rel_m[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta0 = torch.asin(sin_th)
        psi0   = torch.atan2(-rel_m[..., 2], rel_m[..., 0])
        self.theta[env_ids] = theta0
        self.psi_v[env_ids] = psi0
        self._ny[env_ids] = 0.0
        self._nz[env_ids] = 0.0

        # 初始化“友机到活敌质心”的上一帧距离缓存
        enemy_active   = (~self.enemy_frozen[env_ids])
        e_mask         = enemy_active.float().unsqueeze(-1)
        sum_pos        = (self.enemy_pos[env_ids] * e_mask).sum(dim=1)
        cnt            = e_mask.sum(dim=1).clamp_min(1.0)
        centroid       = sum_pos / cnt
        c              = centroid.unsqueeze(1).expand(-1, self.M, 3)
        dist0          = torch.linalg.norm(c - self.fr_pos[env_ids], dim=-1)
        if not hasattr(self, "prev_dist_centroid") or self.prev_dist_centroid is None \
        or self.prev_dist_centroid.shape != (self.num_envs, self.M):
            self.prev_dist_centroid = torch.zeros(self.num_envs, self.M, device=self.device)
        self.prev_dist_centroid[env_ids] = dist0

        self._refresh_enemy_cache()

        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _reset_idx: {dt_ms:.3f} ms")

        bad = ~torch.isfinite(self.fr_pos[env_ids]).all(dim=-1)
        if bad.any():
            nidx = env_ids[bad.any(dim=1).nonzero(as_tuple=False)[0,0]].item()
            midx = bad[nidx == env_ids].nonzero(as_tuple=False)[0,1].item()
            print(f"[NaN after reset] env={nidx} agent={midx} fr_pos={self.fr_pos[nidx, midx]}")

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        为每个 agent 返回自己的局部观测：
            obs_i = [ fr_pos_i(3) | fr_vel_i(3) | dir_to_all_enemies(3*E, 冻结敌机置零) ]
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        eps = 1e-6
        N, M, E = self.num_envs, self.M, self.E

        rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)   # [N,M,E,3]
        dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)
        e_hat_all = rel_all / dist_all

        if hasattr(self, "enemy_frozen") and self.enemy_frozen is not None:
            enemy_active = (~self.enemy_frozen).unsqueeze(1).unsqueeze(-1).float()
            e_hat_all = e_hat_all * enemy_active

        e_hat_flat = e_hat_all.reshape(N, M, 3 * E)
        obs_each = torch.cat([self.fr_pos, self.fr_vel_w, e_hat_flat], dim=-1)  # [N,M, 6+3E]

        obs_dict: dict[str, torch.Tensor] = {}
        for i, agent in enumerate(self.possible_agents):
            obs_dict[agent] = obs_each[:, i, :]  # [N, 6+3E]

        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _get_observations: {dt_ms:.3f} ms")

        return obs_dict

    def _get_states(self) -> torch.Tensor:
        """
        提供集中式状态（供 MAPPO 等使用）：简单拼接所有友机的 [pos, vel] 与敌机方向。
        也可以直接复用集中式观测：这里用每个友机的 obs 串接。
        """
        N, M, E = self.num_envs, self.M, self.E
        eps = 1e-6
        rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)   # [N,M,E,3]
        dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)
        e_hat_all = rel_all / dist_all
        if hasattr(self, "enemy_frozen") and self.enemy_frozen is not None:
            enemy_active = (~self.enemy_frozen).unsqueeze(1).unsqueeze(-1).float()
            e_hat_all = e_hat_all * enemy_active
        e_hat_flat = e_hat_all.reshape(N, M, 3 * E)
        obs_each = torch.cat([self.fr_pos, self.fr_vel_w, e_hat_flat], dim=-1)  # [N,M, 6+3E]
        return obs_each.reshape(N, -1)

# ---------------- Gym 注册 ----------------
from config import agents

gym.register(
    id="FAST-Intercept-Swarm-Distributed",
    entry_point=FastInterceptionSwarmMARLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FastInterceptionSwarmMARLCfg,
        # 与 swarm_vel_env 一致，提供多种算法配置入口（若只用其中部分也没问题）
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:L_M_interception_swarm_skrl_mappo_cfg.yaml",
    },
)
