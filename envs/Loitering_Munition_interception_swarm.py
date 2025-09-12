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
    swarm_size: int = 5                 # 便捷参数：同时设置友机/敌机数量
    friendly_size: int = 6             # 显式设置（可选）
    enemy_size: int = 6                # 显式设置（可选）

    # 敌机出生区域（圆盘）与最小间隔
    debug_vis_enemy = True
    enemy_height_min = 3.0
    enemy_height_max = 3.0
    enemy_speed = 1.5
    enemy_seek_origin = True
    enemy_target_alt = 3.0
    enemy_goal_radius = 0.5
    enemy_cluster_ring_radius: float = 20.0   # R：以 env 原点为圆心，在半径 R 的圆周上选簇中心
    enemy_cluster_radius: float = 5.0         # r：以簇中心为圆心的小圆半径
    enemy_min_separation: float = 4.0         # 敌机间最小 XY 间距（放不下会自适应稍微放宽）

    # 友方控制/速度范围/位置间隔
    Vm_min = 1.5
    Vm_max = 3.0
    ny_max_g = 3.0
    nz_max_g = 3.0
    formation_spacing = 0.8
    flight_altitude = 0.2

    # 单机观测/动作维度（实际 env * M）
    single_observation_space = 9
    single_action_space = 3

    # 占位（会在 __init__ 时按 M 覆盖）
    observation_space = 9
    state_space = 0
    action_space = 3
    clip_action = 1.0

    # 奖励相关
    hit_radius = 1.0
    centroid_approach_weight = 1.0
    hit_reward_weight: float = 1000.0        # 单对首次命中奖励
    heading_align_weight = 0.5
    # 频率
    episode_length_s = 30.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

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
        self.friend_frozen = torch.zeros(N, self.M, device=dev, dtype=torch.bool)      # [N,M]
        self.enemy_frozen  = torch.zeros(N, self.E, device=dev, dtype=torch.bool)      # [N,E]
        self.friend_capture_pos = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)   # [N,M,3]
        self.enemy_capture_pos  = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)   # [N,E,3]

        # 统计/动作缓存
        self.episode_sums = {}
        self.episode_sums["align"]          = torch.zeros(self.num_envs, self.M, device=dev)  # 记录未加权余弦
        self.episode_sums["hit_bonus"]      = torch.zeros(self.num_envs,       device=dev)    # 按 env 累计

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
        self._enemy_centroid = torch.zeros(self.num_envs, 3, device=dev, dtype=dtype)      # [N,3]
        self._enemy_active = torch.zeros(self.num_envs, self.E, device=dev, dtype=torch.bool)  # [N,E]
        self._enemy_active_any = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)  # [N]
        self._goal_e = None                       # [N,3] 在 setup/reset 时构建
        self._axis_hat = torch.zeros(self.num_envs, 3, device=dev, dtype=dtype)  # goal_e->centroid 的单位向量
    
    # —————————————————— ↓↓↓↓↓工具区↓↓↓↓↓ ——————————————————
    # --------- ↓↓↓↓↓友方坐标轴可视化相关↓↓↓↓↓ ---------
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
        """将多维张量 X(通常表示多个批次中多个代理的特征）展平为二维张量，方便后续统一处理所有代理的数据"""
        return X.reshape(-1, X.shape[-1])
    
    # --------- ↑↑↑↑↑友方坐标轴可视化相关↑↑↑↑↑ ---------

    def close(self):
        if getattr(self, "_is_closed", True):
            return
        super().close()
        self._is_closed = True

    # --------- ↓↓↓↓↓敌方生成相关↓↓↓↓↓ ---------
    def _rebuild_goal_e(self):
        """重构敌方目标点,origins 或 enemy_target_alt 改变时重建一次。"""
        origins = self.terrain.env_origins  # [N,3]
        self._goal_e = torch.stack(
            [origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)],
            dim=-1
        )  # [N,3]

    def _refresh_enemy_cache(self):
        """每步结束后更新：活敌掩码 + 质心 + 轴向单位向量(goal_e->centroid)"""
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

        axis = centroid - self._goal_e                           # [N,3] 轴：goal_e -> centroid
        norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._axis_hat = axis / norm                             # [N,3]



    def _spawn_enemy(self, env_ids: torch.Tensor):
        """
        固定三角阵（等边三角形，尖角朝 +X):
        - 阵型中心位于 origin + (R_big, 0)，无随机；
        - 横向间距 d = enemy_min_separation;
        - 纵向(沿尖角方向的法向)行间距 dy = d*sqrt(3)/2;
        - 按 1,2,3,... 行数递增放置，直到放满 E;
        - 高度固定为 (hmin+hmax)/2。
        """
        import math, time
        dev = self.device
        N = env_ids.shape[0]
        E = self.E

        # 参数
        R_big = float(self.cfg.enemy_cluster_ring_radius)       # 阵型整体沿 +X 的偏移
        d     = float(self.cfg.enemy_min_separation)            # 横向间距（同一行内相邻敌机的距离）
        hmin, hmax = float(self.cfg.enemy_height_min), float(self.cfg.enemy_height_max)
        z_fix = (hmin + hmax) * 0.5

        origins = self.terrain.env_origins[env_ids]             # [N,3]

        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        # ---------------- 预生成“局部三角阵” [E,2]（与 env 无关，后续平移到各 env） ----------------
        # 行距（等边三角形几何）
        dy = d * math.sqrt(3.0) / 2.0

        # 计算需要的行数 R：1+2+...+R >= E
        # R*(R+1)/2 >= E  -> R = ceil((sqrt(8E+1)-1)/2)
        R = int(math.ceil((math.sqrt(8.0 * max(E, 1) + 1.0) - 1.0) / 2.0))

        xs, ys = [], []
        count = 0
        for r in range(R):
            # 第 r 行（从 0 开始）有 r+1 个点；沿 Y 居中摆放，X 方向为“前后”方向
            num_in_row = r + 1
            # 这一行的 X：沿 +X 推进，尖角在 x=0，下一行 x 增加 dy
            x_r = r * dy
            # 这一行的 Y：关于 0 对称，间距为 d
            # 让最左的 y 为 -r*d/2，依次加 d，共 r+1 个
            y_start = -0.5 * r * d
            for k in range(num_in_row):
                if count >= E:
                    break
                xs.append(x_r)
                ys.append(y_start + k * d)
                count += 1
            if count >= E:
                break

        # 若恰好填满 E，OK；通常已满足
        # 将局部阵列形状为 [E,2]
        local_xy = torch.stack([
            torch.tensor(xs, dtype=torch.float32, device=dev),
            torch.tensor(ys, dtype=torch.float32, device=dev)
        ], dim=-1)  # [E,2]

        # 让阵型的几何中心落在 (R_big, 0)：这里把局部坐标以其几何中心为基准再平移
        # （也可以选择让“尖角”对齐到 (R_big,0)，如需此效果，去掉下面的中心化即可）
        center_xy = local_xy.mean(dim=0, keepdim=True)          # [1,2]
        local_xy_centered = local_xy - center_xy                # [E,2]

        # ---------------- 广播到所有 env：平移到 origin+(R_big,0) ----------------
        # 阵型参考点：每个 env 的原点沿 +X 偏移 R_big
        centers_xy = torch.stack([
            origins[:, 0] + R_big,       # x
            origins[:, 1]                # y
        ], dim=1)                        # [N,2]

        xy = centers_xy.unsqueeze(1) + local_xy_centered.unsqueeze(0)   # [N,E,2]

        # 高度固定
        z = origins[:, 2:3].unsqueeze(1) + z_fix                        # [N,1,1] + scalar
        z = z.expand(-1, E, 1)                                          # [N,E,1]

        enemy_pos = torch.cat([xy, z], dim=-1)                          # [N,E,3]
        self.enemy_pos[env_ids] = enemy_pos

        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _spawn_enemy triangle: envs={N}, E={E}, d={d:.3f}, dy={dy:.3f}, R_big={R_big:.3f} -> {dt_ms:.3f} ms")

    # --------- ↑↑↑↑↑敌方生成相关↑↑↑↑↑ ---------
    # --------- ↓↓↓↓↓可视化相关↓↓↓↓↓ ---------
    def _build_ray_dots(self, c: torch.Tensor, g: torch.Tensor, step: float) -> torch.Tensor:
        """给定起点c与终点g,按步距step在直线上采样点(含首尾),返回 [K,3]。c,g:[3] on device."""
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
        if not getattr(self.cfg, "proj_vis_enable", False):
            return
        if self._goal_e is None:
            self._rebuild_goal_e()

        dev = self.device
        N_draw = int(min(self.num_envs, getattr(self.cfg, "proj_max_envs", 8)))
        if N_draw <= 0:
            return

        # 聚合点容器
        centroid_pts = []
        ray_pts      = []
        fr_proj_pts  = []
        en_proj_pts  = []

        for ei in range(N_draw):
            # 只在“存在活敌”时绘制
            enemy_active = self._enemy_active[ei]  # [E]
            if not enemy_active.any():
                continue

            # 敌群质心（缓存）
            centroid = self._enemy_centroid[ei]    # [3]

            # 射线：质心 -> 敌目标点
            g = self._goal_e[ei]                   # [3]
            ray_pts.append(self._build_ray_dots(centroid, g, float(self.cfg.proj_ray_step)))
            centroid_pts.append(centroid.unsqueeze(0))

            # 轴向（单位向量，缓存）
            axis_hat = self._axis_hat[ei]          # [3]

            # —— 友机投影 ——（仅活友）
            friend_active = (~self.friend_frozen[ei])          # [M]
            fr_pos = self.fr_pos[ei]                           # [M,3]
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

    def _setup_scene(self):
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # 目标点缓存
        self._rebuild_goal_e()
    
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
                pass
        except Exception:
            pass
    
    # --------- ↑↑↑↑↑可视化相关↑↑↑↑↑ ---------
    # —————————————————— ↑↑↑↑↑工具区↑↑↑↑↑ ——————————————————
    # —————————————————— ↓↓↓↓↓主工作区↓↓↓↓↓ ——————————————————
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        控制输入预处理：
        - 支持 3 种形状的动作： [N, 3*M] / [N, M, 3] / [N, 3](广播到 M)
        - 对已冻结的配对(friend_frozen=True)屏蔽动作，不再对其施加控制
        - 将规范化动作映射到物理量:ny/nz ∈ [-1,1] -> g 值;throttle ∈ [0,1] -> Vm ∈ [Vm_min, Vm_max]
        """
        if self._profile_print:
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

        # 映射到物理量，ny/nz 在物理里表示法向过载指令（单位是“g”，即几倍重力的加速度）。要把无量纲的 [-1,1] 映射成有物理上限的指令区间。ny ∈ [-1,1] 被缩放为 [-ny_max_g, +ny_max_g]（单位：g）
        self._ny = ny * self.cfg.ny_max_g
        self._nz = nz * self.cfg.nz_max_g
        self.Vm = self.cfg.Vm_min + throttle * (self.cfg.Vm_max - self.cfg.Vm_min)

        # 打印时间
        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _pre_physics_step: {dt_ms:.3f} ms")

    def _apply_action(self):
        """
        每个物理步推进一次（步首判定：距离<=1m即命中并冻结):
        0) 缓存步首状态
        0.5) 在步首用 fr_pos0/en_pos0 判定并冻结，记录捕获点
        1) 更新友机姿态与速度（已冻结者速度=0)
        2) 计算敌机速度（已冻结者速度=0)
        3) 推进到步末
        4) 覆盖冻结对象的位置/速度
        5) 刷新缓存 + 可视化
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        dt = float(self.physics_dt)
        self._newly_frozen_friend[:] = False
        self._newly_frozen_enemy[:]  = False

        N, M, E = self.num_envs, self.M, self.E
        r = float(self.cfg.hit_radius)  # 1.0m

        # ---------- 0) 缓存步首状态 ----------
        fr_pos0 = self.fr_pos.clone()        # [N,M,3]
        en_pos0 = self.enemy_pos.clone()     # [N,E,3]
        fz0 = self.friend_frozen.clone()     # [N,M]
        ez0 = self.enemy_frozen.clone()      # [N,E]

        # ---------- 0.5) 步首命中：用 fr_pos0/en_pos0 判定 ----------
        # 仅考虑“未冻结的友/敌”对
        active_pair0 = (~fz0).unsqueeze(2) & (~ez0).unsqueeze(1)  # [N,M,E]
        if active_pair0.any():
            diff0 = fr_pos0.unsqueeze(2) - en_pos0.unsqueeze(1)   # [N,M,E,3]
            dist0 = torch.linalg.norm(diff0, dim=-1)              # [N,M,E]
            hit_pair0 = (dist0 <= r) & active_pair0               # [N,M,E]

            # 命中的友/敌（任一配对命中即算）
            fr_hit0 = hit_pair0.any(dim=2)  # [N,M]
            en_hit0 = hit_pair0.any(dim=1)  # [N,E]

            # 本步新冻结标记（用于奖励）
            newly_fr = (~fz0) & fr_hit0
            newly_en = (~ez0) & en_hit0
            self._newly_frozen_friend |= newly_fr # 左右两边只要有一个是true则左边被赋值为true，布尔张量
            self._newly_frozen_enemy  |= newly_en

            # 敌机捕获点：取步首位置
            if newly_en.any():
                self.enemy_capture_pos[newly_en] = en_pos0[newly_en] # 在 PyTorch 里，用一个二维的布尔掩码去索引一个三维张量时，这个掩码会作用在前两个维度上

            # 友机捕获点：在其命中集合里选“最近的敌机”的步首位置。用“敌机索引”去取敌机的位置，再赋值给“这个友机”的捕获点；
            if newly_fr.any():
                INF = torch.tensor(float("inf"), device=self.device, dtype=dist0.dtype) # 创建无穷大值，用于屏蔽非命中对的距离
                dist_masked0 = torch.where(hit_pair0, dist0, INF)      # [N,M,E] 将非命中对的距离置为无穷大，保留命中对的实际距离
                j_star0 = dist_masked0.argmin(dim=2)                   # [N,M] 沿敌方维度找到每个友方物体命中的最近敌方物体的索引，每个友机对应的“最近命中敌机的下标”
                batch_idx = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, M)  # [N,M] 创建批次索引张量，形状 [N, M]，用于索引 en_pos0
                cap_for_friend0 = en_pos0[batch_idx, j_star0, :]       # [N,M,3] 根据 (n, i, j*) 提取最近敌方物体的步首位置
                self.friend_capture_pos[newly_fr] = cap_for_friend0[newly_fr] # 将新冻结友方物体的捕获位置存储到全局张量中

            # 更新冻结掩码（立即生效，后续推进将不再移动这些体）
            self.friend_frozen |= fr_hit0
            self.enemy_frozen  |= en_hit0

        # 用最新冻结掩码（步首后）参与后续动力学
        fz = self.friend_frozen
        ez = self.enemy_frozen

        # ---------- 1) 友机姿态/速度（冻结=0） ----------
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
        V_m = torch.stack([Vxm, Vym, Vzm], dim=-1)          # [N,M,3]
        fr_vel_w_step = y_up_to_z_up(V_m)                   # [N,M,3]

        # ---------- 2) 敌机速度（冻结=0） ----------
        if self.cfg.enemy_seek_origin:
            if self._goal_e is None:
                self._rebuild_goal_e()
            goal = self._goal_e.unsqueeze(1).expand(-1, self.E, -1)  # [N,E,3] 将目标位置张量 self._goal_e 扩展到与敌方物体数量 self.E 匹配的形状
            to_goal = goal - en_pos0                                 # [N,E,3]
            en_dir = to_goal / torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1e-6)
            enemy_vel_step = en_dir * float(self.cfg.enemy_speed)    # [N,E,3]
        else:
            enemy_vel_step = self.enemy_vel                          # [N,E,3]
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
        self._refresh_enemy_cache() # 更新质心等

        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

        self._update_projection_debug_vis() # 投影可视化

        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _apply_action (pre-step 1m check): {dt_ms:.3f} ms")

    def _get_rewards(self) -> torch.Tensor:
        """
        奖励：
        - centroid_approach: 鼓励友机靠近“当前存活敌机”的质心（距离减小为正）。仅计未冻结友机
        - hit_bonus        : 若任一友机与任一“存活敌机”的距离 <= hit_radius(1m)
                            则该敌机本回合记为“命中一次”（一次性发放，每敌机只发一次）
        - align_bonus      : 鼓励友机机头/速度方向对准“敌团质心”方向。仅计未冻结友机
        """
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        N, M = self.num_envs, self.M

        # --- 配置权重 ---
        centroid_w  = float(getattr(self.cfg, "centroid_approach_weight", 1.0))
        hit_w       = float(getattr(self.cfg, "hit_reward_weight", 1000.0))
        # align_w     = float(getattr(self.cfg, "heading_align_weight", 1.0))

        # --- 活跃掩码 / 质心（用缓存） ---
        friend_active     = (~self.friend_frozen)                   # [N,M]
        enemy_active      = self._enemy_active                      # [N,E]
        enemy_active_any  = self._enemy_active_any                  # [N]
        friend_active_f   = friend_active.float()
        centroid          = self._enemy_centroid                    # [N,3]

        # --- 友机到质心的距离减小 ---
        c = centroid.unsqueeze(1).expand(N, M, 3)                  # [N,M,3]
        diff = c - self.fr_pos                                     # [N,M,3]
        dist_now = torch.linalg.norm(diff, dim=-1)                 # [N,M] 友机与敌方团质心的距离

        if (not hasattr(self, "prev_dist_centroid")) or (self.prev_dist_centroid is None) \
        or (self.prev_dist_centroid.shape != dist_now.shape):
            self.prev_dist_centroid = dist_now.detach().clone()

        # 无活敌 -> 不计逼近（冻结 delta）
        dist_now_safe = torch.where(enemy_active_any.unsqueeze(1), dist_now, self.prev_dist_centroid)
        d_delta_signed = self.prev_dist_centroid - dist_now_safe          # 可能为正或负
        centroid_each = d_delta_signed * friend_active_f                  # [N,M]
        base_each = centroid_w * centroid_each
        reward = base_each.sum(dim=1)                                     # [N]

        # --- 命中判定：几何 1m 半径 ---
        # --- 命中奖励：以“本步新冻结的敌机”为准
        new_hits_mask = self._newly_frozen_enemy                          # [N,E]
        hit_bonus = new_hits_mask.float().sum(dim=1) * hit_w              # [N]
        reward = reward + hit_bonus

        # === 朝向对齐奖励：让机头/速度方向对准“敌团质心” ===
        # to_centroid = self._enemy_centroid.unsqueeze(1) - self.fr_pos      # [N,M,3]
        # to_centroid = torch.nn.functional.normalize(to_centroid, dim=-1)   # [N,M,3]
        # vdir = torch.nn.functional.normalize(self.fr_vel_w + 1e-6, dim=-1) # [N,M,3]
        # align = (to_centroid * vdir).sum(dim=-1).clamp(-1.0, 1.0)          # 余弦 ∈ [-1,1] 两个单位向量的点积等于夹角的余弦值

        # align_bonus = align_w * (align * friend_active.float()).sum(dim=1)
        # reward = reward + align_bonus

        # --- 统计（便于日志可视化）---
        self.episode_sums.setdefault("centroid_approach", torch.zeros_like(centroid_each))
        self.episode_sums.setdefault("hit_bonus",         torch.zeros(self.num_envs, device=self.device, dtype=reward.dtype))
        # self.episode_sums.setdefault("align_bonus",       torch.zeros(self.num_envs, device=self.device, dtype=reward.dtype))
        self.episode_sums["centroid_approach"] += centroid_each
        self.episode_sums["hit_bonus"]         += hit_bonus
        # self.episode_sums["align_bonus"]       += align_bonus
        
        # --- 更新缓存 ---
        self.prev_dist_centroid = dist_now_safe

        # ==== 计时打印 ====
        if self._profile_print:
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
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        BIG = 1e9
        tol = float(getattr(self.cfg, "overshoot_tol", 0.2))
        r2_goal = float(self.cfg.enemy_goal_radius) ** 2          # 目标半径平方
        xy_max2 = 20.0 ** 2                                       # 越界半径平方

        if self._goal_e is None:
            self._rebuild_goal_e()

        # 1. 全部被拦截
        success_all_enemies = self.enemy_frozen.all(dim=1)        # [N]

        # 2. 友机 Z 越界
        z = self.fr_pos[..., 2]
        out_z_any = ((z < 0.0) | (z > 6.0)).any(dim=1)            # [N]（示例高度限制，可按需调整）

        # 3. 友机 XY 越界
        origin_xy = self.terrain.env_origins[:, :2].unsqueeze(1)  # [N,1,2]
        dxy = self.fr_pos[..., :2] - origin_xy                    # [N,M,2]
        out_xy_any = (dxy.square().sum(dim=-1) > xy_max2).any(dim=1)  # [N]

        # 4. 友机位置无效
        nan_inf_any = ~torch.isfinite(self.fr_pos).all(dim=(1, 2))     # [N]

        # 5. 友机投影越线 / 敌人到达目标
        N = self.num_envs
        device = self.device
        enemy_goal_any = torch.zeros(N, dtype=torch.bool, device=device)
        overshoot_any  = torch.zeros(N, dtype=torch.bool, device=device)

        alive_mask = ~(success_all_enemies | out_z_any | out_xy_any | nan_inf_any)
        if alive_mask.any():
            idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)       # [K] 活跃环境的索引

            # 1) 敌人到达目标：只算 alive 的环境；仍用平方距离
            diff_e = self.enemy_pos[idx] - self._goal_e[idx].unsqueeze(1)     # [K,E,3]
            enemy_goal_any[idx] = (diff_e.square().sum(dim=-1) < r2_goal).any(dim=1)

            # 2) “越线投影”：只对 alive 且“友/敌都仍有活体”的环境算
            friend_active = (~self.friend_frozen[idx])                  # [K,M]
            enemy_active  = self._enemy_active[idx]                     # [K,E]
            have_both = friend_active.any(dim=1) & enemy_active.any(dim=1)
            if have_both.any():
                k_idx = have_both.nonzero(as_tuple=False).squeeze(-1)   # 子索引
                # 缓存：质心/单位轴向
                centroid = self._enemy_centroid[idx][k_idx]             # [K2,3]
                gk = self._goal_e[idx][k_idx]                           # [K2,3]
                axis_hat = self._axis_hat[idx][k_idx]                   # [K2,3]

                # s = (p - g) · axis_hat   —— 点到轴的标量投影（以 g 为原点）
                sf = ((self.fr_pos[idx][k_idx]    - gk.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [K2,M]
                se = ((self.enemy_pos[idx][k_idx] - gk.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [K2,E]
                # print("sf:", sf)
                # print("se:", se)
                # 掩码搭配：
                # - 友机要取 min → 非活体置 +∞
                # - 敌机要取 max → 非活体置 -∞
                INF     = torch.tensor(float("inf"),     dtype=sf.dtype, device=sf.device)
                NEG_INF = torch.tensor(float("-inf"),    dtype=sf.dtype, device=sf.device)

                sf_masked_for_min = torch.where(friend_active[k_idx], sf, INF)      # [K2,M]
                se_masked_for_max = torch.where(enemy_active[k_idx],  se, NEG_INF)  # [K2,E]

                friend_min = sf_masked_for_min.min(dim=1).values    # [K2]  友机最小投影
                enemy_max  = se_masked_for_max.max(dim=1).values    # [K2]  敌机最大投影
                # print("friend_min:", friend_min)
                # print("enemy_max: ", enemy_max)
                # 你的需求：友机的最小位置 > 敌机的最大位置（留一点容差 tol）
                # 如果这是“越线/完全越过敌团”的判定：
                separated = friend_min > (enemy_max + tol)          # [K2], bool

                # 回填
                overshoot_any[idx[k_idx]] = separated

        #  汇总
        died = out_z_any | out_xy_any | nan_inf_any | success_all_enemies | enemy_goal_any | overshoot_any
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        #  降低 CPU 同步：把统计更新做成“可选且降频”
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
        if self._profile_print:
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
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()
        if not hasattr(self, "terrain"):
            self._setup_scene()
        if self._goal_e is None:
            self._rebuild_goal_e()
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # 初始化变量
        N = len(env_ids) # 需要重置的环境数量
        M = self.M # 友机数量
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
        self._newly_frozen_friend[env_ids] = False
        self._newly_frozen_enemy[env_ids]  = False

        # 友方并排生成（固定沿 X 方向）
        # spacing = getattr(self.cfg, "formation_spacing", 0.8)
        # idx = torch.arange(M, device=dev).float() - (M - 1) / 2.0
        # offsets_xy = torch.stack([idx * spacing, torch.zeros_like(idx)], dim=-1)  # [M,2]
        # offsets_xy = offsets_xy.unsqueeze(0).expand(N, M, 2)                      # [N,M,2]
        # fr0 = torch.empty(N, M, 3, device=dev)
        # fr0[..., :2] = origins[:, :2].unsqueeze(1) + offsets_xy
        # fr0[...,  2] = origins[:,  2].unsqueeze(1) + float(self.cfg.flight_altitude)
        # self.fr_pos[env_ids] = fr0
        # self.fr_vel_w[env_ids] = 0.0

        # 友方并排生成（固定沿 Y 方向）
        spacing = float(getattr(self.cfg, "formation_spacing", 0.8))
        idx = torch.arange(M, device=dev).float() - (M - 1) / 2.0
        offsets_xy = torch.stack([torch.zeros_like(idx), idx * spacing], dim=-1)  # [M,2]
        offsets_xy = offsets_xy.unsqueeze(0).expand(N, M, 2)                      # [N,M,2]
        fr0 = torch.empty(N, M, 3, device=dev)
        fr0[..., :2] = origins[:, :2].unsqueeze(1) + offsets_xy
        fr0[...,  2] = origins[:, 2].unsqueeze(1) + float(self.cfg.flight_altitude)  # [N,1] -> broadcast 到 [N,M]
        self.fr_pos[env_ids]  = fr0
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
        cnt            = e_mask.sum(dim=1).clamp_min(1.0)                     # [N,1]
        centroid       = sum_pos / cnt                                        # [N,3]
        c              = centroid.unsqueeze(1).expand(-1, self.M, 3)          # [N,M,3]
        dist0          = torch.linalg.norm(c - self.fr_pos[env_ids], dim=-1)  # [N,M]
        if not hasattr(self, "prev_dist_centroid") or self.prev_dist_centroid is None \
        or self.prev_dist_centroid.shape != (self.num_envs, self.M):
            self.prev_dist_centroid = torch.zeros(self.num_envs, self.M, device=self.device)
        self.prev_dist_centroid[env_ids] = dist0

        # 刷新缓存（重置后）
        self._refresh_enemy_cache()

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
        if self._profile_print:
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
        if self._profile_print:
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        eps = 1e-6
        N, M, E = self.num_envs, self.M, self.E

        # 全对全相对向量： enemy - friend  -> [N, M, E, 3]
        rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)                # [N,M,E,3]
        dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)      # [N,M,E,1]
        e_hat_all = rel_all / dist_all                                                  # [N,M,E,3]

        # 屏蔽“已冻结敌机”的方向（置零，不改变维度）
        if hasattr(self, "enemy_frozen") and self.enemy_frozen is not None:
            enemy_active = (~self.enemy_frozen).unsqueeze(1).unsqueeze(-1).float()      # [N,1,E,1]
            e_hat_all = e_hat_all * enemy_active

        # 展平每个友机的所有敌机方向 -> [N,M, 3E]
        e_hat_flat = e_hat_all.reshape(N, M, 3 * E)

        # 每个友机 6 + 3E 维
        obs_each = torch.cat([self.fr_pos, self.fr_vel_w, e_hat_flat], dim=-1)          # [N,M, 6+3E]
        # 串接 M 个友机 -> [N, M*(6+3E)]
        obs = obs_each.reshape(N, -1)

        # ==== 计时打印 ====
        if self._profile_print:
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _get_observations: {dt_ms:.3f} ms")

        return {"policy": obs, "odom": obs.clone()}

    # —————————————————— ↑↑↑↑↑主工作区↑↑↑↑↑ ——————————————————
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
