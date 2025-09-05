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
class FastInterceptionSwarmEnvCfg(DirectRLEnvCfg):
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # ---------- 数量控制 ----------
    swarm_size: int = 4                 # 便捷参数：同时设置友机/敌机数量
    friendly_size: int = 5             # 显式设置（可选）
    enemy_size: int = 5                # 显式设置（可选）

    # 敌机出生区域（圆盘）与最小间隔
    debug_vis_enemy = True
    enemy_height_min = 1.0
    enemy_height_max = 3.0
    enemy_speed = 1.5
    enemy_seek_origin = True
    enemy_target_alt = 3.0
    enemy_goal_radius = 0.5
    enemy_cluster_ring_radius: float = 20.0   # R：以 env 原点为圆心，在半径 R 的圆周上选簇中心
    enemy_cluster_radius: float = 5.0         # r：以簇中心为圆心的小圆半径
    enemy_min_separation: float = 1.0         # 敌机间最小 XY 间距（放不下会自适应稍微放宽）

    # 友方控制/速度范围/位置间隔
    Vm_min = 1.5
    Vm_max = 3.0
    ny_max_g = 3.0
    nz_max_g = 3.0
    formation_spacing = 0.8

    # 单机观测/动作维度（实际 env * M）
    single_observation_space = 9
    single_action_space = 3

    # 占位（会在 __init__ 时按 M 覆盖）
    observation_space = 9
    state_space = 0
    action_space = 3
    clip_action = 1.0
    flight_altitude = 0.2

    # 奖励相关
    capture_radius = 3.0
    success_distance_threshold = 1.0
    hit_radius = 1.0
    approach_reward_weight = 1.0
    hit_reward_weight: float = 2000.0        # 单对首次命中奖励

    # 频率
    episode_length_s = 30.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

    # —— 可视化数字标签（使用小方块点阵渲染）——
    label_enable: bool = False          # 开关
    label_z_offset: float = 0.28       # 标签离机体的高度偏移(m)
    label_max_envs: int = 8            # 只给前 K 个 env 画，避免太多点
    label_cell: float = 0.135          # 点阵像素尺寸(方块边长)
    label_gap: float = 0.015           # 像素之间的间隔
    label_char_gap: float = 0.02       # 字符之间水平间隔
    
    # === 投影与射线可视化 ===
    proj_vis_enable: bool = False         # 开关：是否显示投影与射线
    proj_max_envs: int = 8               # 最多可视化的前 K 个 env
    proj_ray_step: float = 0.2           # 射线虚线步距(米)
    proj_ray_size: tuple[float,float,float] = (0.08, 0.08, 0.08)  # 射线方块大小
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

class FastInterceptionSwarmEnv(DirectRLEnv):
    cfg: FastInterceptionSwarmEnvCfg
    _is_closed = True

    DIGIT_BITMAP = {
        "0": ["111","101","101","101","111"],
        "1": ["010","110","010","010","111"],
        "2": ["111","001","111","100","111"],
        "3": ["111","001","111","001","111"],
        "4": ["101","101","111","001","001"],
        "5": ["111","100","111","001","111"],
        "6": ["111","100","111","101","111"],
        "7": ["111","001","010","100","100"],
        "8": ["111","101","111","101","111"],
        "9": ["111","101","111","001","111"],
        "F": ["111","100","110","100","100"],
        "E": ["111","100","110","100","111"],
    }

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
        # 友机冻结：命中后“该友机”停止；敌机冻结：被击落后“该敌机”停止
        self.friend_frozen = torch.zeros(N, self.M, device=dev, dtype=torch.bool)      # [N,M]
        self.enemy_frozen  = torch.zeros(N, self.E, device=dev, dtype=torch.bool)      # [N,E]
        self.friend_capture_pos = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)   # [N,M,3]
        self.enemy_capture_pos  = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)   # [N,E,3]

        # 统计/动作缓存
        self.episode_sums = {}
        self.prev_dist = torch.zeros(N, self.M, device=dev, dtype=dtype)   # 存“每友机的最小敌距”
        self.prev_actions = torch.zeros(N, self.M, self.cfg.single_action_space, device=dev, dtype=dtype)

        # ---- episode 统计缓冲 ----
        self.episode_sums["approach"]       = torch.zeros(self.num_envs, self.M, device=dev)  # 按友机累计
        self.episode_sums["near_1m"]        = torch.zeros(self.num_envs, self.M, device=dev)
        self.episode_sums["align"]          = torch.zeros(self.num_envs, self.M, device=dev)  # 记录未加权余弦
        self.episode_sums["closing_speed"]  = torch.zeros(self.num_envs, self.M, device=dev)
        self.episode_sums["hit_bonus"]      = torch.zeros(self.num_envs,       device=dev)    # 按 env 累计
        self.episode_sums["success_bonus"]  = torch.zeros(self.num_envs,       device=dev)

        # 一次性奖励发放标记
        self.hit_given_enemy = torch.zeros(N, self.E, device=dev, dtype=torch.bool)    # 敌机是否已发过“命中奖励”

        # 可视化器
        self.friendly_visualizer = None
        self.enemy_visualizer = None
        self.friendly_digit_marker = None
        self.enemy_digit_marker = None
        self._warned_no_text_marker = True  
        # —— 投影/射线可视化器 —— 
        self.centroid_marker = None
        self.ray_marker = None
        self.friend_proj_marker = None
        self.enemy_proj_marker = None
        self.set_debug_vis(self.cfg.debug_vis)

    # ----------------- ↓↓↓↓↓工具区↓↓↓↓↓ -----------------
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
        q_m = self._quat_mul(self._qy(self.psi_v), self._qz(self.theta))         # [N,M,4]
        q_w = self._quat_mul(self._qx_plus_90(self.num_envs, self.M), q_m)       # [N,M,4]
        return self._quat_normalize(q_w)

    def _flatten_agents(self, X: torch.Tensor) -> torch.Tensor:
        return X.reshape(-1, X.shape[-1])

    def close(self):
        if getattr(self, "_is_closed", True):
            return
        super().close()
        self._is_closed = True

    def _poisson_disk_in_circle(self,
                            center_xy: torch.Tensor,
                            radius: float,
                            num_points: int,
                            min_sep: float,
                            device: torch.device,
                            batch: int = 64,
                            max_rounds: int = 256) -> torch.Tensor:
        """
        在圆盘内采样 num_points 个点，尽量满足两两距离 >= min_sep
        若密度过高放不下，会逐渐放宽间隔（每若干轮衰减 5%），最终一定返回 num_points
        返回形状 [num_points, 2]
        """
        if num_points <= 0:
            return torch.zeros(0, 2, device=device)

        pts = torch.empty(0, 2, device=device)
        s = max(min_sep, 1e-6)
        stagnation = 0

        # 估算容量，过密则预先小幅放宽（经验 packing≈0.7）
        cap = 0.7 * (math.pi * radius * radius) / (0.25 * math.pi * s * s)  # ≈ 0.7 * 4R^2 / s^2
        if num_points > cap:
            scale = math.sqrt(num_points / max(cap, 1e-6))
            s = s / (1.05 * scale)

        two_pi = 2.0 * math.pi
        for _ in range(max_rounds):
            if pts.shape[0] >= num_points:
                break

            # 圆盘均匀采样 batch 个候选
            u = torch.rand(batch, device=device)
            v = torch.rand(batch, device=device)
            r = radius * torch.sqrt(u)
            ang = two_pi * v
            cand = torch.stack([center_xy[0] + r * torch.cos(ang),
                                center_xy[1] + r * torch.sin(ang)], dim=-1)  # [B,2]

            if pts.shape[0] == 0:
                need = num_points
                take = min(need, cand.shape[0])
                pts = torch.cat([pts, cand[:take]], dim=0)
                continue

            d = torch.cdist(cand, pts)             # [B,K]
            min_d = d.min(dim=1).values
            mask = min_d >= s

            if mask.any():
                sel = cand[mask]
                need = num_points - pts.shape[0]
                take = min(need, sel.shape[0])
                pts = torch.cat([pts, sel[:take]], dim=0)
                stagnation = 0
            else:
                stagnation += 1

            # 若持续卡住，逐步放宽间隔
            if stagnation >= 5:
                s *= 0.95
                stagnation = 0

        # 兜底补满
        if pts.shape[0] < num_points:
            need = num_points - pts.shape[0]
            u = torch.rand(need, device=device)
            v = torch.rand(need, device=device)
            r = radius * torch.sqrt(u)
            ang = two_pi * v
            fill = torch.stack([center_xy[0] + r * torch.cos(ang),
                                center_xy[1] + r * torch.sin(ang)], dim=-1)
            pts = torch.cat([pts, fill], dim=0)

        return pts[:num_points, :]

    def _spawn_enemy(self, env_ids: torch.Tensor):
        """
        对每个 env:
        1) 以 env 原点为圆心、半径 R=cfg.enemy_cluster_ring_radius 的圆周上随机取一点作为“簇中心”；
        2) 以该簇中心为圆心、半径 r=cfg.enemy_cluster_radius 的小圆内，使用 Poisson 采样生成 E 架敌机；
        3) 敌机间距尽量 >= cfg.enemy_min_separation(放不下会自适应微放宽)。
        """
        dev = self.device
        E = self.E # 敌机数
        R_big = float(self.cfg.enemy_cluster_ring_radius)   # 大圆半径 R
        r_small = float(self.cfg.enemy_cluster_radius)      # 小圆半径 r
        s_min = float(self.cfg.enemy_min_separation)        # 最小间距
        hmin = float(self.cfg.enemy_height_min)
        hmax = float(self.cfg.enemy_height_max)

        origins = self.terrain.env_origins[env_ids]  # [N,3]

        for local_i, env_id in enumerate(env_ids.tolist()):
            origin_xy = origins[local_i, :2]  # [2]
            # 在大圆 R 的“这一圈”上随机采样一个点作为簇中心
            theta = 2.0 * math.pi * float(torch.rand((), device=dev).item())
            cx = origin_xy[0] + R_big * math.cos(theta)
            cy = origin_xy[1] + R_big * math.sin(theta)
            center = torch.tensor([cx, cy], device=dev)

            # 在小圆 r 内做 Poisson 采样（最小间距 s_min）
            pts = self._poisson_disk_in_circle(center, r_small, E, s_min, device=dev)  # [E,2]

            # 赋值（XY 来自采样；Z 随机在高度区间内）
            self.enemy_pos[env_id, :, 0] = pts[:, 0]
            self.enemy_pos[env_id, :, 1] = pts[:, 1]
            self.enemy_pos[env_id, :, 2] = origins[local_i, 2] + torch.rand(E, device=dev) * (hmax - hmin) + hmin

    def _label_strings_for_friendly(self) -> list[str]:
        """每个 env 内：F0..F{M-1}；跨 env 重复。长度=N*M。"""
        return [f"F{i}" for _ in range(self.num_envs) for i in range(self.M)]

    def _label_strings_for_enemy(self) -> list[str]:
        """每个 env 内：E0..E{E-1}；跨 env 重复。长度=N*E。"""
        return [f"E{i}" for _ in range(self.num_envs) for i in range(self.E)]

    def _digit_points_for_text(self, text: str, base_xy: torch.Tensor,
                            cell: float, gap: float, char_gap: float,
                            device: torch.device) -> torch.Tensor:
        """
        将字符串渲染为二维点阵（XY 平面，Z=0），中心对齐 base_xy。
        返回 [K,2]（K 为像素数）。
        """
        chars = list(text)
        pts = []
        w_char = 3 * cell + 2 * gap
        h_char = 5 * cell + 4 * gap
        width_total = len(chars) * w_char + (len(chars) - 1) * char_gap
        x0 = -0.5 * width_total  # 水平居中
        for ci, ch in enumerate(chars):
            bmp = self.DIGIT_BITMAP.get(ch, self.DIGIT_BITMAP["0"])
            left = x0 + ci * (w_char + char_gap)
            for r in range(5):
                row = bmp[r]
                for c in range(3):
                    if row[c] == "1":
                        x = left + c * (cell + gap)
                        y = (5 - 1 - r) * (cell + gap) - 0.5 * h_char  # 垂直居中
                        pts.append((x, y))
        if not pts:
            return torch.empty(0, 2, device=device, dtype=torch.float32)
        arr = torch.tensor(pts, dtype=torch.float32, device=device)  # [K,2]
        return arr + base_xy.unsqueeze(0)

    def _build_digit_cloud_for_agents(self, agent_pos: torch.Tensor,
                                    labels: list[str],
                                    max_envs: int,
                                    z_offset: float,
                                    cell: float,
                                    gap: float,
                                    char_gap: float) -> torch.Tensor:
        """
        为一组代理（[N,M,3] 或 [N,E,3]）构造点阵方块的位置云，返回 [K,3]（K 为所有像素总和）。
        只为前 max_envs 个 env 生成，防止过重。
        """
        dev = self.device
        N_draw = min(self.num_envs, int(max_envs))
        pts3_all = []
        idx_label = 0
        for ei in range(N_draw):
            for ai in range(agent_pos.shape[1]):
                base = agent_pos[ei, ai, :]  # [3]
                base_xy = base[:2]
                txt = labels[idx_label]
                idx_label += 1
                pts2 = self._digit_points_for_text(txt, base_xy, cell, gap, char_gap, dev)  # [K,2]
                if pts2.numel() == 0:
                    continue
                z = base[2] + float(z_offset)
                zcol = torch.full((pts2.shape[0], 1), z, device=dev, dtype=torch.float32)
                pts3 = torch.cat([pts2, zcol], dim=1)  # [K,3]
                pts3_all.append(pts3)
        if len(pts3_all) == 0:
            return torch.empty(0, 3, device=dev, dtype=torch.float32)
        return torch.cat(pts3_all, dim=0)

    def _build_ray_dots(self, c: torch.Tensor, g: torch.Tensor, step: float) -> torch.Tensor:
        """给定起点c与终点g，按步距step在直线上采样点（含首尾），返回 [K,3]。c,g:[3] on device."""
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
        """在 _apply_action 推进后调用：绘制质心→目标的射线、友/敌在线上的投影点、质心点。
           仅对前 cfg.proj_max_envs 个 env 绘制，避免性能压力。"""
        if not getattr(self.cfg, "proj_vis_enable", True):
            return

        dev = self.device
        N_draw = int(min(self.num_envs, getattr(self.cfg, "proj_max_envs", 8)))
        if N_draw <= 0:
            return

        # 聚合点容器
        centroid_pts = []
        ray_pts      = []
        fr_proj_pts  = []
        en_proj_pts  = []

        origins = self.terrain.env_origins  # [N,3]
        goal_e = torch.stack(
            [origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)],
            dim=-1
        )  # [N,3]

        for ei in range(N_draw):
            # 只在“存在活敌”时绘制
            enemy_active = (~self.enemy_frozen[ei])  # [E]
            if not enemy_active.any():
                continue

            # 敌群质心（仅活敌）
            e_mask  = enemy_active.float().unsqueeze(-1)       # [E,1]
            sum_pos = (self.enemy_pos[ei] * e_mask).sum(dim=0) # [3]
            cnt     = e_mask.sum(dim=0).clamp_min(1e-6)        # [1]
            centroid = sum_pos / cnt                           # [3]

            # 射线：质心 -> 敌目标点
            g = goal_e[ei]                                     # [3]
            ray_pts.append(self._build_ray_dots(centroid, g, float(self.cfg.proj_ray_step)))
            centroid_pts.append(centroid.unsqueeze(0))

            # 轴向（单位向量）
            axis = g - centroid
            axis_norm = torch.linalg.norm(axis).clamp_min(1e-6)
            axis_hat = axis / axis_norm                        # [3]

            # —— 友机投影 ——（可选：仅活友）
            friend_active = (~self.friend_frozen[ei])          # [M]
            fr_pos = self.fr_pos[ei]                           # [M,3]
            # 标量投影 s = (p - c)·axis_hat
            s_f = ((fr_pos - centroid.unsqueeze(0)) * axis_hat.unsqueeze(0)).sum(dim=-1)  # [M]
            p_f = centroid.unsqueeze(0) + s_f.unsqueeze(1) * axis_hat.unsqueeze(0)        # [M,3]
            fr_proj_pts.append(p_f[friend_active])  # 只画活友

            # —— 敌机投影 ——（只画活敌）
            en_pos = self.enemy_pos[ei]                        # [E,3]
            s_e = ((en_pos - centroid.unsqueeze(0)) * axis_hat.unsqueeze(0)).sum(dim=-1)  # [E]
            p_e = centroid.unsqueeze(0) + s_e.unsqueeze(1) * axis_hat.unsqueeze(0)        # [E,3]
            en_proj_pts.append(p_e[enemy_active])

        # 拼接并下发到可视化器
        if len(centroid_pts) > 0 and self.centroid_marker is not None:
            self.centroid_marker.visualize(translations=torch.cat(centroid_pts, dim=0))
        if len(ray_pts) > 0 and self.ray_marker is not None:
            self.ray_marker.visualize(translations=torch.cat(ray_pts, dim=0))
        if len(fr_proj_pts) > 0 and self.friend_proj_marker is not None:
            self.friend_proj_marker.visualize(translations=torch.cat(fr_proj_pts, dim=0) if fr_proj_pts else torch.empty(0,3,device=dev))
        if len(en_proj_pts) > 0 and self.enemy_proj_marker is not None:
            self.enemy_proj_marker.visualize(translations=torch.cat(en_proj_pts, dim=0) if en_proj_pts else torch.empty(0,3,device=dev))

    def _set_debug_vis_impl(self, debug_vis: bool):
        """一次性的“开/关与创建”函数"""
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

            # —— 方块点阵数字 ——（友方蓝 / 敌方红）
            if self.cfg.label_enable:
                if self.friendly_digit_marker is None:
                    f_cfg = CUBOID_MARKER_CFG.copy()
                    f_cfg.prim_path = "/Visuals/FriendlyDigitLabels"
                    f_cfg.markers["cuboid"].size = (
                        self.cfg.label_cell, self.cfg.label_cell, self.cfg.label_cell
                    )
                    f_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 0.5, 1.0)
                    )
                    self.friendly_digit_marker = VisualizationMarkers(f_cfg)
                    self.friendly_digit_marker.set_visibility(True)

                if self.enemy_digit_marker is None:
                    e_cfg = CUBOID_MARKER_CFG.copy()
                    e_cfg.prim_path = "/Visuals/EnemyDigitLabels"
                    e_cfg.markers["cuboid"].size = (
                        self.cfg.label_cell, self.cfg.label_cell, self.cfg.label_cell
                    )
                    e_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.2, 0.2)
                    )
                    self.enemy_digit_marker = VisualizationMarkers(e_cfg)
                    self.enemy_digit_marker.set_visibility(True)

            # —— 投影 / 射线 ——
            if getattr(self.cfg, "proj_vis_enable", True):
                # 质心（黄）
                if self.centroid_marker is None:
                    c_cfg = CUBOID_MARKER_CFG.copy()
                    c_cfg.prim_path = "/Visuals/Proj/Centroid"
                    c_cfg.markers["cuboid"].size = tuple(self.cfg.proj_centroid_size)
                    c_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.9, 0.2))
                    self.centroid_marker = VisualizationMarkers(c_cfg)
                    self.centroid_marker.set_visibility(True)

                # 射线虚线（灰）
                if self.ray_marker is None:
                    r_cfg = CUBOID_MARKER_CFG.copy()
                    r_cfg.prim_path = "/Visuals/Proj/Ray"
                    r_cfg.markers["cuboid"].size = tuple(self.cfg.proj_ray_size)
                    r_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.75))
                    self.ray_marker = VisualizationMarkers(r_cfg)
                    self.ray_marker.set_visibility(True)

                # 友机投影（蓝）
                if self.friend_proj_marker is None:
                    fp_cfg = CUBOID_MARKER_CFG.copy()
                    fp_cfg.prim_path = "/Visuals/Proj/Friend"
                    fp_cfg.markers["cuboid"].size = tuple(self.cfg.proj_friend_size)
                    fp_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.5, 1.0))
                    self.friend_proj_marker = VisualizationMarkers(fp_cfg)
                    self.friend_proj_marker.set_visibility(True)

                # 敌机投影（红）
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
            if self.friendly_digit_marker is not None:
                self.friendly_digit_marker.set_visibility(False)
            if self.enemy_digit_marker is not None:
                self.enemy_digit_marker.set_visibility(False)
            if self.centroid_marker is not None:
                self.centroid_marker.set_visibility(False)
            if self.ray_marker is not None:
                self.ray_marker.set_visibility(False)
            if self.friend_proj_marker is not None:
                self.friend_proj_marker.set_visibility(False)
            if self.enemy_proj_marker is not None:
                self.enemy_proj_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        """每帧“更新内容”的函数.每 sim 迭代调用一次，负责将当前状态下的可视化数据下发到各可视化器。"""
        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))
        
        # —— 方块数字同步 ——
        if self.cfg.label_enable:
            if self.friendly_digit_marker is not None:
                f_pts = self._build_digit_cloud_for_agents(
                    self.fr_pos, self._label_strings_for_friendly(),
                    self.cfg.label_max_envs, self.cfg.label_z_offset,
                    self.cfg.label_cell, self.cfg.label_gap, self.cfg.label_char_gap
                )
                self.friendly_digit_marker.visualize(translations=f_pts)
            if self.enemy_digit_marker is not None:
                e_pts = self._build_digit_cloud_for_agents(
                    self.enemy_pos, self._label_strings_for_enemy(),
                    self.cfg.label_max_envs, self.cfg.label_z_offset,
                    self.cfg.label_cell, self.cfg.label_gap, self.cfg.label_char_gap
                )
                self.enemy_digit_marker.visualize(translations=e_pts)

    def _setup_scene(self):
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _cuda_sync_if_needed(self):
        """在计时前/后调用。兼容 self.device 是 torch.device 或 str 或不存在。"""
        try:
            dev = getattr(self, "device", None)
            if isinstance(dev, torch.device):
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
            elif isinstance(dev, str):
                # "cuda" 或 "cuda:0" 都算
                if dev.startswith("cuda"):
                    torch.cuda.synchronize(torch.device(dev))
            else:
                # 没有 self.device：不强行同步；如果你想确保同步当前设备，可改为：
                # if torch.cuda.is_available(): torch.cuda.synchronize()
                pass
        except Exception:
            # 同步失败时不影响主流程
            pass

    # ----------------- ↑↑↑↑↑工具区↑↑↑↑↑ -----------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        控制输入预处理：
        - 支持 3 种形状的动作： [N, 3*M] / [N, M, 3] / [N, 3](广播到 M)
        - 对已冻结的配对(friend_frozen=True)屏蔽动作，不再对其施加控制
        - 将规范化动作映射到物理量:ny/nz ∈ [-1,1] -> g 值;throttle ∈ [0,1] -> Vm ∈ [Vm_min, Vm_max]
        - 打印“活跃配对”的平均空速，便于调试
        """
        self._cuda_sync_if_needed()
        t0 = time.perf_counter()

        if actions is None:
            return
        N = self.num_envs
        M = self.M

        # --- 统一动作形状为 [N, M, 3] ---
        if actions.dim() == 2 and actions.shape[1] == 3*M:
            act = actions.view(N, M, 3)
        elif actions.dim() == 3 and actions.shape[1:] == (M, 3):
            act = actions
        elif actions.dim() == 2 and actions.shape[1] == 3:
            act = actions.view(N, 1, 3).repeat(1, M, 1)  # 广播到每个友机
        else:
            raise RuntimeError(f"Action shape mismatch. Got {tuple(actions.shape)}, expected [N,{3*M}] or [N,{M},3].")

        # --- 对已冻结配对屏蔽动作（冻结对不再响应控制）---
        if hasattr(self, "friend_frozen") and self.friend_frozen is not None:
            active_mask_f = (~self.friend_frozen).float().unsqueeze(-1)  # [N,M,1]
            act = act * active_mask_f

        # --- 规范化与映射 ---
        ny = act[..., 0].clamp(-1.0, 1.0)          # 法向过载指令（归一化）
        nz = act[..., 1].clamp(-1.0, 1.0)
        throttle = act[..., 2].clamp(0.0, 1.0)     # 油门（归一化）

        # 映射到物理量
        self._ny = ny * self.cfg.ny_max_g
        self._nz = nz * self.cfg.nz_max_g
        self.Vm = self.cfg.Vm_min + throttle * (self.cfg.Vm_max - self.cfg.Vm_min)

        # --- 仅统计活跃友机的平均空速，便于调试 ---
        if hasattr(self, "friend_frozen") and self.friend_frozen is not None:
            active_mask_f = (~self.friend_frozen).float()
            denom = active_mask_f.sum().clamp_min(1.0)
            vm_mean_active = (self.Vm * active_mask_f).sum() / denom
        #     print(f"Vm(mean-active): {vm_mean_active.item():.2f} m/s", end="\r")
        # else:
        #     print(f"Vm(mean): {self.Vm.mean().item():.2f} m/s", end="\r")

        #打印时间
        self._cuda_sync_if_needed()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[TIME] _pre_physics_step: {dt_ms:.3f} ms")

    def _apply_action(self):
        """
        每物理步推进(全对全+CCD):
        1) 仅对“未冻结”的友/敌计算本子步恒速，缓存步首状态；
        2) 对所有友×敌做连续碰撞检测：解 || d0 + v_rel * (t*dt) || = R,t∈[0,1];
        3) 在本步内的所有候选命中对中，按“最早命中时间”做一对一匹配（贪心/同时最小）：
            — 选 t_{i*,j*} 同时是该友机行最小 & 敌机列最小（容差内），避免一敌多友冲突；
        4) 对匹配命中对：两端冻结，并把抓捕点写入 friend_capture_pos / enemy_capture_pos;
        5) 推进到步末，对冻结对象覆盖“速度=0,位置=捕获点”；
        6) 可视化。
        """
        self._cuda_sync_if_needed()
        t0 = time.perf_counter()

        dt = float(self.physics_dt)
        dev = self.device
        eps = 1e-12
        BIG = 1e9
        TOL = 1e-6

        N, M, E = self.num_envs, self.M, self.E

        # ---------- 0) 缓存步首状态 ----------
        fr_pos0 = self.fr_pos.clone()          # [N,M,3]
        en_pos0 = self.enemy_pos.clone()       # [N,E,3]
        fz = self.friend_frozen                # [N,M]
        ez = self.enemy_frozen                 # [N,E]

        # ---------- 1) 计算本步恒速（冻结屏蔽） ----------
        # 友机角速率/姿态
        cos_th_now = torch.cos(self.theta).clamp_min(1e-6)
        Vm_eff = torch.where(fz, torch.zeros_like(self.Vm), self.Vm)   # 冻结友机速度=0
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

        # 友机速度(y-up)->世界(z-up)
        sin_th, cos_th = torch.sin(self.theta), torch.cos(self.theta)
        sin_ps, cos_ps = torch.sin(self.psi_v), torch.cos(self.psi_v)
        Vxm = Vm_eff * cos_th * cos_ps
        Vym = Vm_eff * sin_th
        Vzm = -Vm_eff * cos_th * sin_ps
        V_m = torch.stack([Vxm, Vym, Vzm], dim=-1)               # [N,M,3] (y-up)
        fr_vel_w_step = y_up_to_z_up(V_m)                        # [N,M,3] (z-up, 本步恒速)

        # 敌机速度（冻结敌机=0；未冻结目标指向 env 原点+高度）
        if self.cfg.enemy_seek_origin:
            origins = self.terrain.env_origins                                        # [N,3]
            goal = torch.stack([origins[:,0], origins[:,1], origins[:,2] + float(self.cfg.enemy_target_alt)], dim=-1)
            goal = goal.unsqueeze(1).expand(-1, self.E, -1)                           # [N,E,3]
            to_goal = goal - en_pos0                                                  # [N,E,3]
            en_dir = to_goal / torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1e-6)
            enemy_vel_step = en_dir * float(self.cfg.enemy_speed)                     # [N,E,3]
        else:
            enemy_vel_step = self.enemy_vel                                           # [N,E,3]
        enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)

        # ---------- 2) 全对全连续碰撞检测 ----------
        r = float(self.cfg.hit_radius)

        fr_pos0_ = fr_pos0.unsqueeze(2)            # [N,M,1,3]
        en_pos0_ = en_pos0.unsqueeze(1)            # [N,1,E,3]
        d0    = fr_pos0_ - en_pos0_                # [N,M,E,3]
        v_rel = fr_vel_w_step.unsqueeze(2) - enemy_vel_step.unsqueeze(1)  # [N,M,E,3]

        a = (v_rel * v_rel).sum(dim=-1)            # [N,M,E]
        b = 2.0 * (d0 * v_rel).sum(dim=-1)         # [N,M,E]
        c = (d0 * d0).sum(dim=-1) - (r * r)        # [N,M,E]

        a_dt = a * (dt * dt)
        b_dt = b * dt
        c_dt = c

        disc = b_dt * b_dt - 4.0 * a_dt * c_dt
        has_quad = a_dt > eps
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))

        t1 = (-b_dt - sqrt_disc) / (2.0 * torch.clamp(a_dt, min=eps))
        t2 = (-b_dt + sqrt_disc) / (2.0 * torch.clamp(a_dt, min=eps))
        t_hit_quad = torch.minimum(t1, t2)                         # [N,M,E]
        inside_start = c_dt <= 0.0

        # 仅对“未冻结友/敌”的对儿有效
        active_pair = (~fz).unsqueeze(2) & (~ez).unsqueeze(1)      # [N,M,E]
        valid_quad  = has_quad & (disc >= 0.0) & (t_hit_quad >= 0.0) & (t_hit_quad <= 1.0)
        hit_ccd = (inside_start | valid_quad) & active_pair

        # 命中时间矩阵（无命中置 INF；起始在内置 0）
        t_mat = torch.where(hit_ccd, torch.where(inside_start, torch.zeros_like(t_hit_quad), t_hit_quad), 
                            torch.full_like(t_hit_quad, BIG))

        # ---------- 3) 一对一选择：严格唯一“行/列 argmin”交叉 ----------
        # 先把非命中的置成 BIG，避免被 argmin 选中
        t_valid = torch.where(t_mat < BIG, t_mat, torch.full_like(t_mat, BIG))

        # 每友机的唯一最早命中敌机索引 j_star: [N,M]
        t_friend_min, j_star = t_valid.min(dim=2)
        # 每敌机的唯一最早被命中友机索引 i_star: [N,E]
        t_enemy_min,  i_star = t_valid.min(dim=1)

        # 行/列唯一位置的布尔掩码
        mask_row = torch.zeros_like(t_valid, dtype=torch.bool)          # [N,M,E]
        mask_row.scatter_(2, j_star.unsqueeze(-1), True)

        mask_col = torch.zeros_like(t_valid, dtype=torch.bool)          # [N,M,E]
        mask_col.scatter_(1, i_star.unsqueeze(-1), True)

        # 最终一对一匹配
        selected_pairs = mask_row & mask_col & (t_valid < BIG)          # [N,M,E]

        # ---------- 3.1 命中位置：敌机在 t_hit 处的位置 ----------
        # 注意：inside_start 的 t_mat 已是 0，非命中是 BIG；t_valid < BIG 的都是有效命中
        cap_pos = en_pos0_ + enemy_vel_step.unsqueeze(1) * (dt * torch.clamp(t_valid, 0.0, 1.0)).unsqueeze(-1)  # [N,M,E,3]

        # ---------- 3.2 用显式 (n,i,j) 三元组索引来写入与冻结（避免布尔掩码展平顺序不一致） ----------
        idx = selected_pairs.nonzero(as_tuple=False)   # [K,3], 列为 [n, i, j]
        if idx.numel() > 0:
            n_idx = idx[:, 0]
            i_idx = idx[:, 1]
            j_idx = idx[:, 2]

            # 取 K 个抓捕点
            cap_points = cap_pos[n_idx, i_idx, j_idx, :]    # [K,3]

            # 写入抓捕点
            self.friend_capture_pos[n_idx, i_idx, :] = cap_points
            self.enemy_capture_pos[n_idx, j_idx, :] = cap_points

            # 冻结命中两端
            self.friend_frozen[n_idx, i_idx] = True
            self.enemy_frozen[n_idx, j_idx] = True

        # 同步冻结掩码的局部引用
        fz = self.friend_frozen
        ez = self.enemy_frozen

        # ---------- 4) 推进到步末，并覆盖冻结对象 ----------
        self.fr_vel_w = fr_vel_w_step
        self.enemy_vel = enemy_vel_step

        self.fr_pos = fr_pos0 + fr_vel_w_step * dt
        self.enemy_pos = en_pos0 + enemy_vel_step * dt

        # 覆盖冻结：速度=0，位置=捕获点
        self.fr_vel_w[fz] = 0.0
        self.enemy_vel[ez] = 0.0
        self.fr_pos[fz] = self.friend_capture_pos[fz]
        self.enemy_pos[ez] = self.enemy_capture_pos[ez]

        # ---------- 5) 可视化 ----------
        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))
        self._update_projection_debug_vis()
        # 数字标签
        if self.cfg.label_enable:
            if self.friendly_digit_marker is not None:
                f_pts = self._build_digit_cloud_for_agents(
                    self.fr_pos, self._label_strings_for_friendly(),
                    self.cfg.label_max_envs, self.cfg.label_z_offset,
                    self.cfg.label_cell, self.cfg.label_gap, self.cfg.label_char_gap
                )
                self.friendly_digit_marker.visualize(translations=f_pts)
            if self.enemy_digit_marker is not None:
                e_pts = self._build_digit_cloud_for_agents(
                    self.enemy_pos, self._label_strings_for_enemy(),
                    self.cfg.label_max_envs, self.cfg.label_z_offset,
                    self.cfg.label_cell, self.cfg.label_gap, self.cfg.label_char_gap
                )
                self.enemy_digit_marker.visualize(translations=e_pts)

        # ==== 计时打印 ====
        self._cuda_sync_if_needed()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[TIME] _apply_action: {dt_ms:.3f} ms")

    def _get_rewards(self) -> torch.Tensor:
        """
        奖励：
        - centroid_approach: 鼓励友机靠近“当前存活敌机”的质心（距离减小为正）。仅计未冻结友机。
        - hit_bonus        : 若任一友机与任一“存活敌机”的距离 <= hit_radius（1m），
                            则该敌机本回合记为“命中一次”（一次性发放，每敌机只发一次）。
        """
        self._cuda_sync_if_needed()
        t0 = time.perf_counter()

        eps = 1e-6
        N, M, E = self.num_envs, self.M, self.E

        # --- 配置权重 ---
        centroid_w  = float(getattr(self.cfg, "centroid_approach_weight", 1.0))
        hit_w       = float(getattr(self.cfg, "hit_reward_weight", 2000.0))
        hit_r       = float(getattr(self.cfg, "hit_radius", 1.0))   # 这里应为 1.0m

        # --- 活跃掩码 ---
        friend_active     = (~self.friend_frozen)                  # [N,M]
        enemy_active      = (~self.enemy_frozen)                   # [N,E]
        friend_active_f   = friend_active.float()
        enemy_active_any  = enemy_active.any(dim=1)                # [N]

        # --- 质心（仅活敌）---
        e_mask  = enemy_active.float().unsqueeze(-1)               # [N,E,1]
        sum_pos = (self.enemy_pos * e_mask).sum(dim=1)             # [N,3]
        cnt     = e_mask.sum(dim=1).clamp_min(1.0)                 # [N,1]
        centroid = sum_pos / cnt                                   # [N,3]

        # --- 友机到质心的距离减小 ---
        c = centroid.unsqueeze(1).expand(N, M, 3)                  # [N,M,3]
        diff = c - self.fr_pos                                     # [N,M,3]
        dist_now = torch.linalg.norm(diff, dim=-1)                 # [N,M]

        if (not hasattr(self, "prev_dist_centroid")) or (self.prev_dist_centroid is None) \
        or (self.prev_dist_centroid.shape != dist_now.shape):
            self.prev_dist_centroid = dist_now.detach().clone()

        # 无活敌 -> 不计逼近（冻结 delta）
        dist_now_safe = torch.where(enemy_active_any.unsqueeze(1), dist_now, self.prev_dist_centroid)
        d_delta = torch.clamp(self.prev_dist_centroid - dist_now_safe, min=0.0)

        centroid_each = d_delta * friend_active_f                  # [N,M]
        base_each = centroid_w * centroid_each
        base_each = base_each * enemy_active_any.unsqueeze(1).float()  # 无活敌 -> 0
        reward = base_each.sum(dim=1)                               # [N]
        # 如需按活跃友机数平均，可用：
        # num_friend_active = friend_active_f.sum(dim=1).clamp_min(1.0)
        # reward = base_each.sum(dim=1) / num_friend_active

        # --- 命中判定：几何 1m 半径 ---
        # 全对全距离
        rel_all  = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)         # [N,M,E,3]
        dist_all = torch.linalg.norm(rel_all, dim=-1)                              # [N,M,E]
        # 只对“活跃友 × 活跃敌”考虑命中
        pair_active = friend_active.unsqueeze(2) & enemy_active.unsqueeze(1)       # [N,M,E]，pair_active[n, m, e] = True 表示第 n 个环境中，第 m 个友机和第 e 个敌机均为活跃。
        geo_hit_pairs = (dist_all <= hit_r) & pair_active                          # [N,M,E]

        # 按敌机聚合：该敌机是否被任一友机命中
        hit_now_enemy = geo_hit_pairs.any(dim=1)                                   # [N,E]
        # 一次性：该敌机此前未奖励过
        new_hit_enemies = hit_now_enemy & (~self.hit_given_enemy)                  # [N,E]
        hit_bonus = new_hit_enemies.float().sum(dim=1) * hit_w                     # [N]
        reward = reward + hit_bonus
        # 记录已发过命中奖励
        self.hit_given_enemy |= new_hit_enemies

        # --- 统计（便于日志可视化）---
        self.episode_sums.setdefault("centroid_approach", torch.zeros_like(centroid_each))
        self.episode_sums.setdefault("hit_bonus",         torch.zeros(self.num_envs, device=self.device, dtype=reward.dtype))
        self.episode_sums["centroid_approach"] += centroid_each
        self.episode_sums["hit_bonus"]         += hit_bonus

        # --- 更新缓存 ---
        self.prev_dist_centroid = dist_now

        # ==== 计时打印 ====
        self._cuda_sync_if_needed()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[TIME] _get_rewards: {dt_ms:.3f} ms")
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        终止条件：
        1) 所有敌机被击落(success_all_enemies)
        2) 友机 z 轴越界 / xy 越界
        3) 位置数据 NaN/Inf
        4) 任一敌人到达目标（若仍启用敌机寻标）
        5) 友方整体“越线投影”（避免掉头打击）
        6) episode 超时
        """
        self._cuda_sync_if_needed()
        t0 = time.perf_counter()

        BIG = 1e9
        tol = float(getattr(self.cfg, "overshoot_tol", 0.2))
        r2_goal = float(self.cfg.enemy_goal_radius) ** 2          # 目标半径平方
        xy_max2 = 300.0 ** 2                                      # 越界半径平方

        # 1) 便宜项：不做 sqrt、不做大广播
        success_all_enemies = self.enemy_frozen.all(dim=1)        # [N]

        z = self.fr_pos[..., 2]
        out_z_any = ((z < 0.0) | (z > 5.5)).any(dim=1)          # [N]

        # XY 越界用平方距离
        origin_xy = self.terrain.env_origins[:, :2].unsqueeze(1)  # [N,1,2]
        dxy = self.fr_pos[..., :2] - origin_xy                    # [N,M,2]
        out_xy_any = (dxy.square().sum(dim=-1) > xy_max2).any(dim=1)  # [N]

        # 直接用 isfinite，一步搞定 nan/inf
        nan_inf_any = ~torch.isfinite(self.fr_pos).all(dim=(1, 2))     # [N]

        # 先默认较重项为 False；仅对“还活着”的 env 再细算
        N = self.num_envs
        device = self.device
        enemy_goal_any = torch.zeros(N, dtype=torch.bool, device=device)
        overshoot_any  = torch.zeros(N, dtype=torch.bool, device=device)

        alive_mask = ~(success_all_enemies | out_z_any | out_xy_any | nan_inf_any)
        if alive_mask.any():
            idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)       # [K]

            # 目标点（建议缓存：如果 origins/alt 不变，可在 __init__/reset 后保存 self._goal_e）
            origins = self.terrain.env_origins                          # [N,3]
            goal_e = torch.stack(
                [origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)],
                dim=-1
            )  # [N,3]

            # 2) 敌人到达目标：只算 alive 的环境；仍用平方距离
            diff_e = self.enemy_pos[idx] - goal_e[idx].unsqueeze(1)     # [K,E,3]
            enemy_goal_any[idx] = (diff_e.square().sum(dim=-1) < r2_goal).any(dim=1)

            # 3) “越线投影”：只对 alive 且“友/敌都仍有活体”的环境算
            friend_active = (~self.friend_frozen[idx])                  # [K,M]
            enemy_active  = (~self.enemy_frozen[idx])                   # [K,E]
            have_both = friend_active.any(dim=1) & enemy_active.any(dim=1)
            if have_both.any():
                k_idx = have_both.nonzero(as_tuple=False).squeeze(-1)   # 子索引
                # 敌群“活体”质心
                e_mask = enemy_active[k_idx].unsqueeze(-1).float()      # [K2,E,1]
                sum_pos = (self.enemy_pos[idx][k_idx] * e_mask).sum(dim=1)   # [K2,3]
                cnt = e_mask.sum(dim=1).clamp_min(1.0)                  # [K2,1]
                centroid = sum_pos / cnt

                # 轴：goal_e -> centroid
                gk = goal_e[idx][k_idx]                                  # [K2,3]
                axis = centroid - gk                                     # [K2,3]
                axis_hat = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)

                # 投影（点乘）：s = (p - g)·axis_hat
                sf = ((self.fr_pos[idx][k_idx]    - gk.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [K2,M]
                se = ((self.enemy_pos[idx][k_idx] - gk.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [K2,E]

                BIG_T = torch.tensor(BIG, dtype=sf.dtype, device=sf.device)
                sf_masked = torch.where(friend_active[k_idx], sf, BIG_T)  # 非活体置 +INF，便于 min 忽略
                se_masked = torch.where(enemy_active[k_idx],  se, BIG_T)

                friend_front = sf_masked.min(dim=1).values
                enemy_front  = se_masked.min(dim=1).values

                overshoot = friend_front > (enemy_front - tol)
                overshoot_any[idx[k_idx]] = overshoot

        # 4) 汇总
        died = out_z_any | out_xy_any | nan_inf_any | success_all_enemies | enemy_goal_any | overshoot_any
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 5) 降低 CPU 同步：把统计更新做成“可选且降频”
        log_every = int(getattr(self.cfg, "log_termination_every", 1))  # 0 表示关闭
        if log_every and (int(self.episode_length_buf.max().item()) % log_every == 0):
            term = self.extras.setdefault("termination", {})
            # 只做一次同步，把七个统计合在一起搬到 CPU
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
        # ==== 计时打印 ====
        self._cuda_sync_if_needed()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[TIME] _get_dones: {dt_ms:.3f} ms")

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        重置指定 env 的运行状态（只影响 env_ids 指定的那些环境）
            1) 统计与计数
            2) 捕获/配对流程相关缓存
            3) 友机初始状态（位置/速度）
            4) 敌机初始状态（位置/速度）
            5) 友机姿态/控制量初始化
            6) RL 缓存与上一步距离
            7) 可视化
        """
        self._cuda_sync_if_needed()
        t0 = time.perf_counter()

        if not hasattr(self, "terrain"):
            self._setup_scene()

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # 初始化变量
        N = len(env_ids) #需要重置的环境数量
        M = self.M #友机数量
        dev = self.device
        origins = self.terrain.env_origins[env_ids]  # [N,3]

        # 清零 episode 统计
        for k in list(self.episode_sums.keys()):
            self.episode_sums[k][env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        # 清空冻结状态与捕获点
        self.friend_frozen[env_ids] = False
        self.enemy_frozen[env_ids]  = False
        self.friend_capture_pos[env_ids] = 0.0
        self.enemy_capture_pos[env_ids]  = 0.0
        self.hit_given_enemy[env_ids]    = False

        # 友方并排生成
        spacing = getattr(self.cfg, "formation_spacing", 0.8)
        idx = torch.arange(M, device=dev).float() - (M - 1) / 2.0
        offsets_xy = torch.stack([idx * spacing, torch.zeros_like(idx)], dim=-1)  # [M,2]
        offsets_xy = offsets_xy.unsqueeze(0).expand(N, M, 2)                      # [N,M,2]
        fr0 = torch.empty(N, M, 3, device=dev)
        fr0[..., :2] = origins[:, :2].unsqueeze(1) + offsets_xy
        fr0[...,  2] = origins[:,  2].unsqueeze(1) + float(self.cfg.flight_altitude)
        self.fr_pos[env_ids] = fr0
        self.fr_vel_w[env_ids] = 0.0

        # 敌机在圆盘内随机出生（带最小间隔约束）
        self._spawn_enemy(env_ids)

        # 敌机初速度（环向寻标）
        phi = torch.rand(N, device=dev) * 2.0 * math.pi
        spd = float(self.cfg.enemy_speed)
        self.enemy_vel[env_ids, :, 0] = spd * torch.cos(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 1] = spd * torch.sin(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 2] = 0.0

        # 友方初始速度（自动面向团中心）/姿态
        self.Vm[env_ids] = 0.0
        en_pos = self.enemy_pos[env_ids]                # [N,E,3]
        centroid = en_pos.mean(dim=1)                   # [N,3]

        # 每个友机指向质心的相对向量（世界系 z-up）
        rel_w = centroid.unsqueeze(1) - self.fr_pos[env_ids]   # [N,M,3]

        # 转到机体使用的 y-up 表达，并单位化
        rel_m = z_up_to_y_up(rel_w)
        rel_m = rel_m / rel_m.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        # 由方向向量解初始姿态
        sin_th = rel_m[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta0 = torch.asin(sin_th)
        psi0   = torch.atan2(-rel_m[..., 2], rel_m[..., 0])
        self.theta[env_ids] = theta0
        self.psi_v[env_ids] = psi0
        self._ny[env_ids] = 0.0
        self._nz[env_ids] = 0.0

        # 初始化“友机到活敌质心”的上一帧距离缓存
        enemy_active   = (~self.enemy_frozen[env_ids])                        # [N,E] 此时通常全 True
        e_mask         = enemy_active.float().unsqueeze(-1)                   # [N,E,1]
        sum_pos        = (self.enemy_pos[env_ids] * e_mask).sum(dim=1)        # [N,3]
        cnt            = e_mask.sum(dim=1).clamp_min(1.0)                      # [N,1]
        centroid       = sum_pos / cnt                                        # [N,3]
        c              = centroid.unsqueeze(1).expand(-1, self.M, 3)          # [N,M,3]
        dist0          = torch.linalg.norm(c - self.fr_pos[env_ids], dim=-1)  # [N,M]
        if not hasattr(self, "prev_dist_centroid") or self.prev_dist_centroid is None \
        or self.prev_dist_centroid.shape != (self.num_envs, self.M):
            self.prev_dist_centroid = torch.zeros(self.num_envs, self.M, device=self.device)
        self.prev_dist_centroid[env_ids] = dist0

        # RL 缓存（放在 fr_pos/enemy_pos 之后）
        # prev_dist: 设为“每友机的最小敌距”
        rel_full = self.enemy_pos[env_ids].unsqueeze(1) - self.fr_pos[env_ids].unsqueeze(2)   # [N,1,E,3] - [N,M,1,3] => [N,M,E,3]
        dist_full = torch.linalg.norm(rel_full, dim=-1)                                       # [N,M,E]
        min_dist, _ = dist_full.min(dim=2)                                                    # [N,M]

        if (self.prev_dist is None) or (self.prev_dist.shape != (self.num_envs, self.M)):
            self.prev_dist = torch.zeros(self.num_envs, self.M, device=self.device)
        self.prev_dist[env_ids] = min_dist

        if (self.prev_actions is None) or (self.prev_actions.shape != (self.num_envs, self.M, self.cfg.single_action_space)):
            self.prev_actions = torch.zeros(self.num_envs, self.M, self.cfg.single_action_space, device=self.device)
        self.prev_actions[env_ids] = 0.0

        # 可视化
        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

        # ==== 计时打印 ====
        self._cuda_sync_if_needed()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[TIME] _reset_idx: {dt_ms:.3f} ms")

    def _get_observations(self) -> dict:
        """
        观测（固定维度；不依赖编号配对）：
        对每个友机，拼接：
            [ fr_pos(3) | fr_vel_w(3) | e_hat_to_all_enemies(3*E) ]
        其中 e_hat_to_all_enemies 会对“已冻结敌机”置零，从而不改变维度。
        最终把 M 个友机的观测串接： [N, M * (6 + 3E)]。
        """
        self._cuda_sync_if_needed()
        t0 = time.perf_counter()

        eps = 1e-6
        N, M, E = self.num_envs, self.M, self.E

        # 全对全相对向量： enemy - friend  -> [N, M, E, 3] 表示在 N 个环境中，每个友机（共 M 个）到每个敌机（共 E 个）的单位方向向量（每个向量为 3 维）
        rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)                # [N,M,E,3]
        dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)      # [N,M,E,1]
        e_hat_all = rel_all / dist_all                                                  # [N,M,E,3] 单位向量

        # 屏蔽“已冻结敌机”的方向（置零，不改变维度）
        if hasattr(self, "enemy_frozen") and self.enemy_frozen is not None:
            enemy_active = (~self.enemy_frozen).unsqueeze(1).unsqueeze(-1).float()      # [N,1,E,1]
            e_hat_all = e_hat_all * enemy_active

        #（可选）也可把冻结友机的方向置零，通常没必要；如需则解除注释
        # if hasattr(self, "friend_frozen") and self.friend_frozen is not None:
        #     friend_active = (~self.friend_frozen).unsqueeze(2).unsqueeze(-1).float()   # [N,M,1,1]
        #     e_hat_all = e_hat_all * friend_active

        # 展平每个友机的所有敌机方向 -> [N,M, 3E]
        e_hat_flat = e_hat_all.reshape(N, M, 3 * E)

        # 每个友机 6 + 3E 维： [pos(3), vel(3), e_hat_to_all_enemies(3E)]
        obs_each = torch.cat([self.fr_pos, self.fr_vel_w, e_hat_flat], dim=-1)          # [N,M, 6+3E]
        # 串接 M 个友机 -> [N, M*(6+3E)]
        obs = obs_each.reshape(N, -1)
        # ==== 计时打印 ====
        self._cuda_sync_if_needed()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[TIME] _get_observations: {dt_ms:.3f} ms")

        return {"policy": obs, "odom": obs.clone()}

# ---------------- Gym 注册 ----------------
from config import agents

gym.register(
    id="FAST-Intercept-Swarm",
    entry_point=FastInterceptionSwarmEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FastInterceptionSwarmEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:quadcopter_sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:quadcopter_skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Loitering_Munition_interception_swarm_rsl_rl_ppo_cfg:FASTInterceptSwarmPPORunnerCfg",
    },
)
