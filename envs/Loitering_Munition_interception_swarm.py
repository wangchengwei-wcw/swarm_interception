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
    xw = vec_m[..., 0]
    yw = vec_m[..., 2]
    zw = vec_m[..., 1]
    return torch.stack([xw, yw, zw], dim=-1)

def z_up_to_y_up(vec_w: torch.Tensor) -> torch.Tensor:
    xm = vec_w[..., 0]
    ym = vec_w[..., 2]
    zm = vec_w[..., 1]
    return torch.stack([xm, ym, zm], dim=-1)

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
    enemy_target_alt = 5.0
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
    success_reward_weight = 600.0
    approach_reward_weight = 1.0
    hit_reward_weight: float = 200.0        # 单对首次命中奖励
    heading_align_weight: float = 2.0

    # 频率
    episode_length_s = 30.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

    # —— 可视化数字标签（使用小方块点阵渲染）——
    label_enable: bool = True          # 开关
    label_z_offset: float = 0.28       # 标签离机体的高度偏移(m)
    label_max_envs: int = 8            # 只给前 K 个 env 画，避免太多点
    label_cell: float = 0.135          # 点阵像素尺寸(方块边长)
    label_gap: float = 0.015           # 像素之间的间隔
    label_char_gap: float = 0.02       # 字符之间水平间隔

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
        cfg.observation_space = cfg.single_observation_space * M
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

        # 冻结掩码与命中位置
        self.pair_frozen = torch.zeros(N, self.M, device=dev, dtype=torch.bool)     # [N,M]
        self.pair_capture_pos = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)  # [N,M,3]

        # 统计/动作缓存
        self.episode_sums = {}
        self.prev_dist = torch.zeros(N, self.M, device=dev, dtype=dtype)     # 按配对 i↔i
        self.prev_actions = torch.zeros(N, self.M, self.cfg.single_action_space, device=dev, dtype=dtype)

        # ---- episode 统计缓冲 ----
        self.episode_sums = {}
        self.episode_sums["approach"]      = torch.zeros(self.num_envs, self.M, device=dev)
        self.episode_sums["near_1m"]       = torch.zeros(self.num_envs, self.M, device=dev)
        self.episode_sums["hit_bonus"]     = torch.zeros(self.num_envs,           device=dev)
        self.episode_sums["success_bonus"] = torch.zeros(self.num_envs,           device=dev)

        self.hit_given = torch.zeros(N, self.M, device=dev, dtype=torch.bool)  # [N,M]：该配对是否发过“命中奖励”
        self.success_given = torch.zeros(N, device=dev, dtype=torch.bool)      # [N]  ：该 env 是否发过“全歼奖励”

        # 可视化器
        self.friendly_visualizer = None
        self.enemy_visualizer = None
        self.friendly_digit_marker = None
        self.enemy_digit_marker = None
        self._warned_no_text_marker = True  
        self.set_debug_vis(self.cfg.debug_vis)

    # ----------------- 工具区 -----------------
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
        else:
            if self.friendly_visualizer is not None:
                self.friendly_visualizer.set_visibility(False)
            if self.enemy_visualizer is not None:
                self.enemy_visualizer.set_visibility(False)
            if self.friendly_digit_marker is not None:
                self.friendly_digit_marker.set_visibility(False)
            if self.enemy_digit_marker is not None:
                self.enemy_digit_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))
        
        # —— 数字同步 ----
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

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        控制输入预处理：
        - 支持 3 种形状的动作： [N, 3*M] / [N, M, 3] / [N, 3](广播到 M)
        - 对已冻结的配对(pair_frozen=True)屏蔽动作，不再对其施加控制
        - 将规范化动作映射到物理量:ny/nz ∈ [-1,1] -> g 值;throttle ∈ [0,1] -> Vm ∈ [Vm_min, Vm_max]
        - 打印“活跃配对”的平均空速，便于调试
        """
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
        if hasattr(self, "pair_frozen") and self.pair_frozen is not None:
            active_mask_f = (~self.pair_frozen).float().unsqueeze(-1)  # [N,M,1]
            act = act * active_mask_f

        # --- 规范化与映射 ---
        ny = act[..., 0].clamp(-1.0, 1.0)          # 法向过载指令（归一化）
        nz = act[..., 1].clamp(-1.0, 1.0)
        throttle = act[..., 2].clamp(0.0, 1.0)     # 油门（归一化）

        # 映射到物理量
        self._ny = ny * self.cfg.ny_max_g
        self._nz = nz * self.cfg.nz_max_g
        self.Vm = self.cfg.Vm_min + throttle * (self.cfg.Vm_max - self.cfg.Vm_min)

        # --- 仅统计活跃配对的平均空速，便于调试 ---
        if hasattr(self, "pair_frozen") and self.pair_frozen is not None:
            active_mask_f = (~self.pair_frozen).float()
            denom = active_mask_f.sum().clamp_min(1.0)
            vm_mean_active = (self.Vm * active_mask_f).sum() / denom
            print(f"Vm(mean-active): {vm_mean_active.item():.2f} m/s", end="\r")
        else:
            print(f"Vm(mean): {self.Vm.mean().item():.2f} m/s", end="\r")

    def _apply_action(self):
        """
        每物理步推进(带连续命中判定 CCD):
        1) 缓存子步开始状态（友机/敌机位置），基于当前控制计算本子步“将使用”的速度；
        2) 以 d(t) = d0 + v_rel * (t*dt) 的匀速相对运动近似，解 ||d(t)|| = r 的二次方程做连续命中判定：
            - 若起始时刻就在半径内(c <= 0) => 立即命中(t_hit=0);
            - 否则判别式 >= 0 且存在 t_hit ∈ [0,1] => 子步内命中；
            - 否则不命中。
        3) 对新命中的配对：记录捕获点（取敌机在 t_hit 处的位置）、置冻结；
        4) 正常推进到子步末:pos = pos0 + vel * dt;
        5) 对冻结配对做覆盖：速度=0、位置=捕获点（含本步新命中与历史已冻结）。
        说明：
        - 命中与奖励仍按 i↔i 编号配对进行(友机 i 对敌机 i)。
        - 冻结配对的推进（姿态/速度/位置）都会被屏蔽。
        - 为了数值稳定，对 a≈0(相对速度极小)与判别式<0 做了防护。
        """
        dt = float(self.physics_dt)
        dev = self.device
        eps = 1e-12

        # ====== 0) 子步开始：缓存“步首”位置（用于 CCD） ======
        # 仅对前 M 个敌机与友机一一配对做命中判定；其余敌机正常推进但不参与配对命中。
        fr_pos0 = self.fr_pos.clone()                  # [N,M,3]
        en_pos0_full = self.enemy_pos.clone()          # [N,E,3]
        en_pos0 = en_pos0_full[:, :self.M, :].clone()  # [N,M,3]

        # ====== 1) 根据控制计算本子步将使用的速度（冻结配对屏蔽推进） ======
        freeze = self.pair_frozen                      # [N,M] 布尔掩码

        # --- 角速度计算（与原逻辑一致，但对冻结对置零）---
        cos_th_now = torch.cos(self.theta).clamp_min(1e-6)
        # 冻结对的有效空速为 0（既用于角速度公式也用于生成线速度）
        Vm_eff = torch.where(freeze, torch.zeros_like(self.Vm), self.Vm)
        Vm_eps = Vm_eff.clamp_min(1e-6)

        theta_rate = self.g0 * (self._ny - cos_th_now) / Vm_eps
        psi_rate   = - self.g0 * self._nz / (Vm_eps * cos_th_now)

        theta_rate = torch.where(freeze, torch.zeros_like(theta_rate), theta_rate)
        psi_rate   = torch.where(freeze, torch.zeros_like(psi_rate),   psi_rate)

        # 限幅
        THETA_RATE_LIMIT = 1.0  # rad/s
        PSI_RATE_LIMIT   = 1.0  # rad/s
        theta_rate = torch.clamp(theta_rate, -THETA_RATE_LIMIT, THETA_RATE_LIMIT)
        psi_rate   = torch.clamp(psi_rate,   -PSI_RATE_LIMIT,   PSI_RATE_LIMIT)

        # 姿态积分到“步末姿态”
        self.theta = self.theta + theta_rate * dt
        self.psi_v = (self.psi_v + psi_rate * dt + math.pi) % (2.0 * math.pi) - math.pi

        # 由“步末姿态 + Vm_eff”生成本子步近似恒定的 y-up 速度，再转到世界 z-up
        sin_th, cos_th = torch.sin(self.theta), torch.cos(self.theta)
        sin_ps, cos_ps = torch.sin(self.psi_v), torch.cos(self.psi_v)
        Vxm = Vm_eff * cos_th * cos_ps
        Vym = Vm_eff * sin_th
        Vzm = -Vm_eff * cos_th * sin_ps
        V_m = torch.stack([Vxm, Vym, Vzm], dim=-1)         # [N,M,3] (y-up)
        fr_vel_w_step = y_up_to_z_up(V_m)                  # [N,M,3] (z-up)，作为本子步恒定速度

        # 敌机速度（本子步恒定）
        if self.cfg.enemy_seek_origin:
            origins = self.terrain.env_origins                                        # [N,3]
            goal = torch.stack([origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)], dim=-1)
            goal = goal.unsqueeze(1).expand(-1, self.E, -1)                           # [N,E,3]
            to_goal = goal - en_pos0_full                                             # [N,E,3] 用“步首位置”定方向更物理
            dist_to_goal = torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1e-6)
            en_dir = to_goal / dist_to_goal
            enemy_vel_step_full = en_dir * float(self.cfg.enemy_speed)                # [N,E,3]
        else:
            enemy_vel_step_full = self.enemy_vel                                      # [N,E,3] 保持上一步速度

        enemy_vel_step = enemy_vel_step_full[:, :self.M, :]                            # [N,M,3] 与友机配对的前 M 个

        # ====== 2) 连续命中判定（CCD） ======
        # 相对运动： d(t) = (fr_pos0 - en_pos0) + (fr_vel - en_vel) * (t * dt), t ∈ [0,1]
        # 求 ||d(t)||^2 = r^2 的根。
        r = float(self.cfg.hit_radius)

        d0    = fr_pos0 - en_pos0                       # [N,M,3]
        v_rel = fr_vel_w_step - enemy_vel_step          # [N,M,3]

        a = (v_rel * v_rel).sum(dim=-1)                 # [N,M]
        b = 2.0 * (d0 * v_rel).sum(dim=-1)              # [N,M]
        c = (d0 * d0).sum(dim=-1) - (r * r)             # [N,M]

        # 归一化到 t ∈ [0,1] 的多项式：把 dt 吸收入系数，避免尺度过大/过小
        a_dt = a * (dt * dt)                            # [N,M]
        b_dt = b * dt                                   # [N,M]
        c_dt = c                                        # [N,M]

        disc = b_dt * b_dt - 4.0 * a_dt * c_dt          # 判别式
        has_quad = a_dt > eps                           # 相对速度足够大
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))

        # 两根（选更早的进入时刻）
        t1 = (-b_dt - sqrt_disc) / (2.0 * torch.clamp(a_dt, min=eps))
        t2 = (-b_dt + sqrt_disc) / (2.0 * torch.clamp(a_dt, min=eps))
        t_hit_quad = torch.minimum(t1, t2)              # [N,M]

        # 起始即在半径内：立即命中
        inside_start = c_dt <= 0.0                      # [N,M]

        # 有效根：有二次项、判别式非负、t ∈ [0,1]
        valid_quad = has_quad & (disc >= 0.0) & (t_hit_quad >= 0.0) & (t_hit_quad <= 1.0)

        # 组合：本子步内命中事件
        t_hit = torch.where(inside_start, torch.zeros_like(t_hit_quad), t_hit_quad)  # inside_start -> 0
        hit_ccd = inside_start | valid_quad                                          # [N,M] 是否命中（CCD）

        # 仅对“未冻结”的配对判定为新命中
        new_hits = hit_ccd & (~freeze)                                               # [N,M]

        # 捕获点（取敌机在 t_hit 的位置；也可用相对中点）
        cap_pos = en_pos0 + enemy_vel_step * (dt * t_hit.clamp(0.0, 1.0)).unsqueeze(-1)  # [N,M,3]
        if new_hits.any():
            self.pair_capture_pos[new_hits] = cap_pos[new_hits]
            self.pair_frozen |= new_hits
            freeze = self.pair_frozen  # 更新冻结掩码（后续推进用）

        # ====== 3) 推进到子步末（先正常推进，再对冻结配对覆盖到捕获点） ======
        # 更新缓冲中的“当前速度”（供外部可视化/观测使用）
        self.fr_vel_w = fr_vel_w_step
        self.enemy_vel = enemy_vel_step_full

        # 位置推进（基于“步首位置 + 本子步恒速 * dt”）
        self.fr_pos = fr_pos0 + fr_vel_w_step * dt
        self.enemy_pos = en_pos0_full + enemy_vel_step_full * dt

        # 对冻结配对强制覆盖：速度=0，位置=捕获点
        self.fr_vel_w[freeze] = 0.0
        self.fr_pos[freeze]   = self.pair_capture_pos[freeze]
        self.enemy_vel[:, :self.M, :][freeze] = 0.0
        self.enemy_pos[:, :self.M, :][freeze] = self.pair_capture_pos[freeze]

        # ====== 4) 可视化 ======
        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()     # [N,M,4]
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))
        # ---- 数字标签（方块点阵）----
        if self.cfg.label_enable:
            if self.friendly_digit_marker is not None:
                f_pts = self._build_digit_cloud_for_agents(
                    self.fr_pos, self._label_strings_for_friendly(),
                    self.cfg.label_max_envs, self.cfg.label_z_offset,
                    self.cfg.label_cell, self.cfg.label_gap, self.cfg.label_char_gap
                )  # [K,3]
                self.friendly_digit_marker.visualize(translations=f_pts)

            if self.enemy_digit_marker is not None:
                e_pts = self._build_digit_cloud_for_agents(
                    self.enemy_pos, self._label_strings_for_enemy(),
                    self.cfg.label_max_envs, self.cfg.label_z_offset,
                    self.cfg.label_cell, self.cfg.label_gap, self.cfg.label_char_gap
                )
                self.enemy_digit_marker.visualize(translations=e_pts)

    def _get_rewards(self) -> torch.Tensor:
        """
        奖励（按 i↔i 配对，集中式）：
        - approach   ：距离减小的增量奖励（仅未冻结配对）
        - near_1m    ：距离 < 1m 的近距离奖励（仅未冻结配对）
        - align      ：速度方向与“指向敌机”的单位向量对齐的奖励（仅未冻结且速度>阈值的配对）
                        使用余弦相似度 cos(u, e) ∈ [-1,1]，对齐奖励为正，背离为负。
        - hit_bonus  ：单对首次命中一次性奖励（当该配对首次从未冻结→冻结）
        - success    ：全歼一次性奖励（当该 env 首次所有配对都冻结）

        注:所有逐配对奖励(approach/near_1m/align)对“未冻结配对”求平均得到每个 env 的基础奖励。
        """
        eps = 1e-6

        # ---- 距离与掩码 ----
        rel  = self.enemy_pos[:, :self.M, :] - self.fr_pos          # [N,M,3]
        dist = torch.linalg.norm(rel, dim=-1)                       # [N,M]
        if self.prev_dist is None or self.prev_dist.shape != dist.shape:
            self.prev_dist = dist.clone()

        active_mask   = (~self.pair_frozen)                         # [N,M] 仍在交战
        active_mask_f = active_mask.float()
        active_cnt    = active_mask_f.sum(dim=1).clamp_min(1.0)     # [N] 用于求平均

        # ---- 1) 逼近奖励 ----
        approach = (self.prev_dist - dist) * active_mask_f          # [N,M]

        # ---- 2) 近距离奖励（<1m）----
        near_1m  = torch.clamp(1.0 - dist, min=0.0) * active_mask_f # [N,M]

        # ---- 3) 朝向对齐奖励：cos(u, e) ----
        # 视线单位向量 e_w
        e_w = rel / torch.clamp(dist.unsqueeze(-1), min=eps)        # [N,M,3]
        # 速度方向 u（用 fr_vel_w；也可用期望速度向量）
        speed = torch.linalg.norm(self.fr_vel_w, dim=-1)            # [N,M]
        move_mask = (speed > 1e-3)                                  # 速度过小则不计
        u_dir = self.fr_vel_w / torch.clamp(speed.unsqueeze(-1), min=eps)  # [N,M,3]
        # 余弦相似度：对齐为 +1，反向为 -1，垂直为 0
        cos_align = (u_dir * e_w).sum(dim=-1)                       # [N,M]
        # 仅对未冻结且在移动的配对计分
        align_term = cos_align * active_mask_f * move_mask.float()  # [N,M]
        align_reward_each = float(self.cfg.heading_align_weight) * align_term

        # ---- 基础奖励：对“仍在交战”的配对求平均 ----
        base_each = self.cfg.approach_reward_weight * approach + near_1m + align_reward_each  # [N,M]
        reward = base_each.sum(dim=1) / active_cnt                                            # [N]

        # ---- 4) 单对首次命中奖励（一次性）----
        new_hit_pairs = self.pair_frozen & (~self.hit_given)          # [N,M] 本步新冻结的配对
        hit_weight = float(getattr(self.cfg, "hit_reward_weight", 200.0))
        hit_bonus = new_hit_pairs.float().sum(dim=1) * hit_weight     # [N]
        reward = reward + hit_bonus
        self.hit_given |= new_hit_pairs

        # ---- 5) 全歼一次性奖励 ----
        success_hit_all = self.pair_frozen.all(dim=1)                 # [N]
        new_success_env = success_hit_all & (~self.success_given)     # [N]
        success_weight = float(getattr(self.cfg, "success_reward_weight", 600.0))
        success_bonus = new_success_env.float() * success_weight      # [N]
        reward = reward + success_bonus
        self.success_given |= new_success_env

        # ---- 6) 统计累计（便于日志/评估），确保键存在 ----
        self.episode_sums.setdefault("approach",      torch.zeros_like(approach))
        self.episode_sums.setdefault("near_1m",       torch.zeros_like(near_1m))
        self.episode_sums.setdefault("align",         torch.zeros_like(align_term))
        self.episode_sums.setdefault("hit_bonus",     torch.zeros(self.num_envs, device=self.device, dtype=reward.dtype))
        self.episode_sums.setdefault("success_bonus", torch.zeros(self.num_envs, device=self.device, dtype=reward.dtype))

        self.episode_sums["approach"]      += approach        # [N,M]
        self.episode_sums["near_1m"]       += near_1m         # [N,M]
        self.episode_sums["align"]         += align_term      # [N,M]（记录未加权的 cos，以便可视化）
        self.episode_sums["hit_bonus"]     += hit_bonus       # [N]
        self.episode_sums["success_bonus"] += success_bonus   # [N]

        # ---- 7) 更新 prev_dist ----
        self.prev_dist = dist
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        终止条件：
           1.成功击中所有敌人(success_hit_all)
           2.己方单位 z 轴越界(out_z_any)
           3.己方单位 xy 平面越界(out_xy_any)
           4.位置数据出现 NaN 或无穷值(nan_inf_any)
           5.任一敌人到达目标(enemy_goal_any)
           6.episode 超时(time_out)
        """
        success_hit_all = self.pair_frozen.all(dim=1)  # [N]

        z = self.fr_pos[..., 2]
        out_z_any = ((z < 0.0) | (z > 200.0)).any(dim=1)

        xy_rel = self.fr_pos[..., :2] - self.terrain.env_origins[:, :2].unsqueeze(1)
        out_xy_any = (torch.linalg.norm(xy_rel, dim=-1) > 300.0).any(dim=1)

        nan_inf_any = (torch.isnan(self.fr_pos).any(dim=(1,2)) | torch.isinf(self.fr_pos).any(dim=(1,2)))

        origins = self.terrain.env_origins
        goal = torch.stack([origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)], dim=-1)
        goal = goal.unsqueeze(1).expand(-1, self.E, -1)
        enemy_goal_dist = torch.linalg.norm(goal - self.enemy_pos, dim=-1)   # [N,E]
        enemy_goal_any = (enemy_goal_dist < float(self.cfg.enemy_goal_radius)).any(dim=1)

        died = out_z_any | out_xy_any | nan_inf_any | success_hit_all | enemy_goal_any
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.extras.setdefault("termination", {})
        self.extras["termination"].update({
            "hit_all_envs": int(success_hit_all.sum().item()),
            "hit_total_pairs": int(self.pair_frozen.sum().item()),
            "out_of_bounds_any": int((out_z_any | out_xy_any).sum().item()),
            "nan_inf_any": int(nan_inf_any.sum().item()),
            "enemy_goal_any": int(enemy_goal_any.sum().item()),
            "time_out": int(time_out.sum().item()),
        })
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
        if not hasattr(self, "terrain"):
            self._setup_scene()

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # 初始化变量
        N = len(env_ids) #需要重置的环境数量
        M = self.M #友机数量
        dev = self.device
        origins = self.terrain.env_origins[env_ids]  # [N,3]

        # 清零episode统计
        for k in list(self.episode_sums.keys()):
            self.episode_sums[k][env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        # 清空各种状态
        self.pair_frozen[env_ids] = False
        self.pair_capture_pos[env_ids] = 0.0
        self.episode_sums["approach"][env_ids]      = 0.0
        self.episode_sums["near_1m"][env_ids]       = 0.0
        self.episode_sums["success_bonus"][env_ids] = 0.0
        self.episode_length_buf[env_ids]            = 0
        self.hit_given[env_ids] = False
        self.success_given[env_ids] = False

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

        # 友方初始速度（自动面向敌机）/姿态
        self.Vm[env_ids] = 0.0
        rel_w = self.enemy_pos[env_ids, :M, :] - self.fr_pos[env_ids]     # [N,M,3]
        rel_m = z_up_to_y_up(rel_w)
        rel_m = rel_m / rel_m.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        sin_th = rel_m[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta0 = torch.asin(sin_th)
        psi0   = torch.atan2(-rel_m[..., 2], rel_m[..., 0])
        self.theta[env_ids] = theta0
        self.psi_v[env_ids] = psi0
        self._ny[env_ids] = 0.0
        self._nz[env_ids] = 0.0

        # 敌机初速度（环向寻标）
        phi = torch.rand(N, device=dev) * 2.0 * math.pi
        spd = float(self.cfg.enemy_speed)
        self.enemy_vel[env_ids, :, 0] = spd * torch.cos(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 1] = spd * torch.sin(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 2] = 0.0

        # RL 缓存（必须放在rel_w之后）
        if (self.prev_dist is None) or (self.prev_dist.shape != (self.num_envs, self.M)):
            self.prev_dist = torch.zeros(self.num_envs, self.M, device=self.device)
        self.prev_dist[env_ids] = torch.linalg.norm(rel_w, dim=-1)  # [N,M]
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

    def _get_observations(self) -> dict:
        """观测组成（集中式，按友机逐个串接）：
            - 每台友机 9 维： [ fr_pos(3) | fr_vel_w(3) | e_w(3) ]
            * fr_pos(3): 世界系(z-up)下友机位置
            * fr_vel_w(3): 世界系(z-up)下友机速度
            * e_w(3): 友机指向“配对敌机”的单位向量（相对位置方向）"""
        rel = self.enemy_pos[:, :self.M, :] - self.fr_pos                     # [N,M,3]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
        e_w = rel / dist # [N,M,3] 归一化的指向配对敌机的方向向量
        obs_each = torch.cat([self.fr_pos, self.fr_vel_w, e_w], dim=-1)       # [N,M,9]
        obs = obs_each.reshape(self.num_envs, -1)                             # [N, 9*M]
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
