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
    # (x_m, y_m, z_m) -> (x_w, y_w, z_w)
    xm = vec_m[..., 0]
    ym = vec_m[..., 1]
    zm = vec_m[..., 2]
    return torch.stack([xm, -zm, ym], dim=-1)

def z_up_to_y_up(vec_w: torch.Tensor) -> torch.Tensor:
    # (x_w, y_w, z_w) -> (x_m, y_m, z_m)
    xw = vec_w[..., 0]
    yw = vec_w[..., 1]
    zw = vec_w[..., 2]
    return torch.stack([xw, zw, -yw], dim=-1)

@configclass
class FastInterceptionSwarmEnvCfg(DirectRLEnvCfg):
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # ---------- 数量控制 ----------
    swarm_size: int = 6                 # 便捷参数：同时设置友机/敌机数量
    friendly_size: int = 6             # 显式设置（可选）
    enemy_size: int = 6                # 显式设置（可选）

    # 敌机出生区域（圆盘）与最小间隔
    debug_vis_enemy = True
    enemy_height_min = 2.0
    enemy_height_max = 3.0
    enemy_speed = 0.5
    enemy_seek_origin = True
    enemy_target_alt = 3.0
    enemy_goal_radius = 0.5
    enemy_cluster_ring_radius: float = 15.0   # R：以 env 原点为圆心，在半径 R 的圆周上选簇中心
    enemy_cluster_radius: float = 3.0         # r：以簇中心为圆心的小圆半径
    enemy_min_separation: float = 0.5         # 敌机间最小 XY 间距（放不下会自适应稍微放宽）
    # hit_radius = 0.01
    hit_radius = 0.1

    # 友方控制/速度范围/位置间隔
    Vm_min = 1.1
    Vm_max = 1.3

    ny_max_g = 3.0
    nz_max_g = 3.0
    formation_spacing = 0.8
    flight_altitude = 0.2

    # 单机观测/动作维度（实际 env * M）
    single_observation_space = 9
    single_action_space = 5

    # 导引头相关参数
    gimbal_fov_h_deg: float = 10.0    # 水平视场角（度）
    gimbal_fov_v_deg: float = 12.0    # 垂直视场角（度）
    gimbal_range_deg: float = 30.0    # 云台最大偏离角（度）
    gimbal_rate_deg: float = 20.0     # 云台最大角速度（度/秒）
    gimbal_effective_range: float = 10  # 云台有效作用距离（米）

    # 占位（会在 __init__ 时按 M 覆盖）
    observation_space = 9
    state_space = 0
    action_space = 3
    clip_action = 1.0

    # 奖励相关
    centroid_approach_weight = 1.0
    hit_reward_weight: float = 2000.0        # 单对首次命中奖励
    w_gimbal_friend_block:float = 0.1
    w_gimbal_enemy_cover:float = 0.01

    # 频率
    episode_length_s = 30.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

    # === 云台视野可视化 ===
    gimbal_vis_enable: bool = False          # 是否绘制云台的视野范围
    gimbal_vis_max_envs: int = 1            # 限制前几个 env，避免点太多
    gimbal_vis_edge_step: float = 0.08      # 采样步距（越小越像“连续线”，但点更多）

    # === 投影与射线可视化 ===
    proj_vis_enable: bool = False         # 开关：是否显示投影与射线
    proj_max_envs: int = 1               # 最多可视化的前 K 个 env
    proj_ray_step: float = 0.2           # 射线虚线步距(米)
    proj_ray_size: tuple[float,float,float] = (0.08, 0.08, 0.08)  # 射线方块大小
    proj_friend_size: tuple[float,float,float] = (0.12, 0.12, 0.12)
    proj_enemy_size:  tuple[float,float,float] = (0.12, 0.12, 0.12)
    proj_centroid_size: tuple[float,float,float] = (0.16, 0.16, 0.16)

    # ==== TRAJ VIS ==== 友方轨迹可视化
    traj_vis_enable: bool = False            # 开关
    traj_vis_max_envs: int = 1              # 只画前几个 env
    traj_vis_len: int = 500                 # 每个友机最多保留多少个轨迹点（循环缓冲）
    traj_vis_every_n_steps: int = 2         # 每隔多少个物理步记录/刷新一次
    traj_marker_size: tuple[float,float,float] = (0.05, 0.05, 0.05)  # 面包屑小方块尺寸

    # for debug
    per_train_data_print: bool = False # reset中打印
    function_time_print: bool = False # 函数耗时打印

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
        # pos(3) + vel(3) + centroid_pos(3) + unit_to_all_enemies(3*E)
        single_obs_dim = 9 + 3 * E
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
        # self.episode_sums["hit_bonus"]      = torch.zeros(self.num_envs,       device=dev)    # 按 env 累计

        # 一次性奖励发放标记
        self._newly_frozen_friend = torch.zeros(N, self.M, dtype=torch.bool, device=dev)
        self._newly_frozen_enemy  = torch.zeros(N, self.E, dtype=torch.bool, device=dev)

        # —— 云台角（同一个云台服务奖励与obs gating） ——
        self._gimbal_yaw   = torch.zeros(N, self.M, device=dev, dtype=dtype)  # [-pi,pi)
        self._gimbal_pitch = torch.zeros(N, self.M, device=dev, dtype=dtype)  # 仰角
        self._gimbal_tgt_rel_yaw_cmd   = torch.zeros(N, self.M, device=dev)
        self._gimbal_tgt_rel_pitch_cmd = torch.zeros(N, self.M, device=dev)

        # ==== TRAJ VIS ====
        self._traj_buf  = torch.zeros(self.num_envs, self.M, int(self.cfg.traj_vis_len), 3,
                                      device=dev, dtype=dtype)  # [N,M,K,3]
        self._traj_len  = torch.zeros(self.num_envs, self.M, device=dev, dtype=torch.long) # [N,M]
        self._traj_markers: list[VisualizationMarkers] = []

        # 可视化器
        self.friendly_visualizer = None
        self.enemy_visualizer = None
        # —— 投影/射线可视化器 —— 
        self._fov_marker         = None
        self.centroid_marker = None
        self.ray_marker = None
        self.friend_proj_marker = None
        self.enemy_proj_marker = None
        self.set_debug_vis(self.cfg.debug_vis)
        self._enemy_centroid_init = torch.zeros(self.num_envs, 3, device=self.device, dtype=self.fr_pos.dtype)
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
        # q = q1 ⊗ q2,先应用 q2 的旋转，再应用 q1 的旋转
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
        # 以 Y 轴 为轴、旋转角 psi 的四元数
        half = 0.5 * psi
        return torch.stack([torch.cos(half), torch.zeros_like(psi), torch.sin(half), torch.zeros_like(psi)], dim=-1)

    def _qz(self, theta: torch.Tensor) -> torch.Tensor:
        # 以 Z 轴 为轴、旋转角 theta 的四元数
        half = 0.5 * theta
        return torch.stack([torch.cos(half), torch.zeros_like(theta), torch.zeros_like(theta), torch.sin(half)], dim=-1)

    def _qx_plus_90(self, *shape_prefix) -> torch.Tensor:
        # 围绕 X 轴 +90° 的固定旋转
        cx = math.sqrt(0.5)
        sx = cx
        base = torch.tensor([cx, sx, 0.0, 0.0], device=self.device, dtype=self.fr_pos.dtype)
        if len(shape_prefix) == 0:
            return base
        rep = int(torch.tensor(shape_prefix).prod().item())
        return base.repeat(rep, 1).reshape(*shape_prefix, 4)

    def _friendly_world_quats(self) -> torch.Tensor:
        q_m = self._quat_mul(self._qy(self.psi_v), self._qz(self.theta))         # [N,M,4] 先绕 Z 旋转 theta，再绕 Y 旋转 psi
        q_w = self._quat_mul(self._qx_plus_90(self.num_envs, self.M), q_m)       # [N,M,4] 把模型姿态整体再绕 X 轴旋转 +90°
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


    # def _spawn_enemy(self, env_ids: torch.Tensor):
    #     """
    #     并行生成敌机（固定最小间距 s_min，不放宽；必要时增大小圆半径）：
    #     1) 每个 env 在半径 R_big 的大圆上选一个“簇中心”（均匀随机）；
    #     2) 在以该中心为圆心、半径 r_env 的小圆内进行 Poisson 约束采样 E 个点；
    #     若放不下，则只增大 r_env（几何充足性：r_env >= 0.5*s_min*sqrt(E/eta)）。
    #     """
    #     if getattr(self.cfg, "function_time_print", False):
    #         self._cuda_sync_if_needed()
    #         t0 = time.perf_counter()

    #     dev = self.device
    #     env_ids = env_ids.to(dtype=torch.long, device=dev)
    #     N = env_ids.numel()
    #     if N == 0:
    #         return

    #     E = self.E
    #     R_big   = float(self.cfg.enemy_cluster_ring_radius)   # 大圆半径
    #     r_small = float(self.cfg.enemy_cluster_radius)        # 配置的小圆半径（下限）
    #     s_min   = float(self.cfg.enemy_min_separation)        # 固定最小间距（不放宽）
    #     hmin    = float(self.cfg.enemy_height_min)
    #     hmax    = float(self.cfg.enemy_height_max)

    #     eta = float(getattr(self.cfg, "enemy_poisson_eta", 0.7)) # 用于估算小圆能否容纳 E 个点

    #     # 读 env 原点
    #     origins_all = self.terrain.env_origins
    #     if origins_all.device != dev:
    #         origins_all = origins_all.to(dev)
    #     origins = origins_all[env_ids]      # [N, 3]

    #     # ---------- 1) 大圆上选簇中心 ----------
    #     two_pi = 2.0 * math.pi
    #     theta = two_pi * torch.rand(N, device=dev) # 生成 N 个随机角度
    #     centers = origins[:, :2] + R_big * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # [N,2] 得到大圆上的采样点xy坐标

    #     # ---------- 2) 小圆内 Poisson 约束采样（只增大半径，不放宽间距） ----------
    #     # 几何充足半径（下界）：r >= (s/2) * sqrt(E/eta)
    #     r_needed = 0.5 * s_min * math.sqrt(E / max(eta, 1e-6)) # 计算几何所需最小半径
    #     # 每个 env 的当前有效半径（起步为 max(r_small, r_needed) 再带 2% 裕度）
    #     r_env = torch.full((N,), max(r_small, r_needed * 1.02), device=dev)

    #     s2 = s_min * s_min
    #     pts = torch.zeros(N, E, 2, device=dev)                  # 结果 [N,E,2] 存储采样点的张量
    #     filled = torch.zeros(N, dtype=torch.long, device=dev)       # 各 env 已放入点数
    #     stagn  = torch.zeros(N, dtype=torch.long, device=dev)       # 连续停滞轮数（用于触发半径增长）
    #     BATCH = 128            # 每轮为每个环境生成128个候选点
    #     MAX_ROUNDS = 256       # 最大采样轮数
    #     GROW_FACTOR = 1.05     # 每次增长 5%
    #     STAGN_ROUNDS = 5       # 连续停滞 5 轮则增大半径
    #     ar_e = torch.arange(E, device=dev)  # [E]，生成 [0, 1, ..., E-1] 张量，用于后续掩码操作

    #     for _ in range(MAX_ROUNDS):
    #         if (filled >= E).all():
    #             break

    #         u = torch.rand(N, BATCH, device=dev)
    #         v = torch.rand(N, BATCH, device=dev)
    #         rr  = r_env.unsqueeze(1) * torch.sqrt(u.clamp_min(1e-12))
    #         ang = two_pi * v
    #         cand = centers[:, None, :] + torch.stack([rr * torch.cos(ang),
    #                                                 rr * torch.sin(ang)], dim=-1)  # [N,BATCH,2]

    #         diff_valid = (ar_e.unsqueeze(0) < filled.unsqueeze(1)).unsqueeze(1)     # [N,1,E]
    #         pts_eff = torch.where(diff_valid.unsqueeze(-1), pts[:, None, :, :], cand[:, :, None, :])
    #         diff = cand[:, :, None, :] - pts_eff
    #         sq   = (diff ** 2).sum(dim=-1).masked_fill(~diff_valid, float("inf"))
    #         min_sq, _ = sq.min(dim=-1)
    #         ok = min_sq >= s2

    #         idxs = torch.arange(BATCH, device=dev).unsqueeze(0).expand(N, -1)
    #         first_idx = torch.where(ok, idxs, torch.full_like(idxs, BATCH)).min(dim=1).values
    #         can_take = (first_idx < BATCH) & (filled < E)

    #         env_take = torch.nonzero(can_take, as_tuple=False).squeeze(1)
    #         if env_take.numel() > 0:
    #             pos = filled[env_take]
    #             pts[env_take, pos, :] = cand[env_take, first_idx[env_take], :]
    #             filled[env_take] += 1
    #             stagn[env_take] = 0

    #         stagn[~can_take] += 1
    #         grow_mask = stagn >= STAGN_ROUNDS
    #         if grow_mask.any():
    #             r_env[grow_mask] *= GROW_FACTOR
    #             stagn[grow_mask] = 0

    #     # 若极端情况下仍未满，再尝试若干轮“只增长半径”的补偿
    #     if (filled < E).any():
    #         EXTRA_GROW_STEPS = 8
    #         for _ in range(EXTRA_GROW_STEPS):
    #             need_mask = filled < E
    #             if not need_mask.any():
    #                 break
    #             r_env[need_mask] *= GROW_FACTOR

    #             u = torch.rand(N, BATCH, device=dev)
    #             v = torch.rand(N, BATCH, device=dev)
    #             rr  = r_env.unsqueeze(1) * torch.sqrt(u)
    #             ang = two_pi * v
    #             cand = centers[:, None, :] + torch.stack([rr * torch.cos(ang),
    #                                                     rr * torch.sin(ang)], dim=-1)

    #             diff  = cand[:, :, None, :] - pts[:, None, :, :]
    #             valid = (ar_e.unsqueeze(0) < filled.unsqueeze(1)).unsqueeze(1)
    #             sq    = (diff ** 2).sum(dim=-1).masked_fill(~valid, float("inf"))
    #             min_sq, _ = sq.min(dim=-1)
    #             ok = min_sq >= s2

    #             idxs = torch.arange(BATCH, device=dev).unsqueeze(0).expand(N, -1)
    #             first_idx = torch.where(ok, idxs, torch.full_like(idxs, BATCH)).min(dim=1).values
    #             can_take = (first_idx < BATCH) & (filled < E)

    #             env_take = torch.nonzero(can_take, as_tuple=False).squeeze(1)
    #             if env_take.numel() > 0:
    #                 pos = filled[env_take]
    #                 pts[env_take, pos, :] = cand[env_take, first_idx[env_take], :]
    #                 filled[env_take] += 1

    #         # 仍未满则报错（说明 r_env 已增大很多仍未成功，需检查参数）
    #         if (filled < E).any():
    #             not_full = int((filled < E).sum().item())
    #             raise RuntimeError(
    #                 f"Poisson sampling failed after radius growth for {not_full}/{N} envs. "
    #                 f"Consider increasing enemy_cluster_radius or reducing enemy_min_separation/E."
    #             )

    #     # ---------- 3) 写回到 self.enemy_pos ----------
    #     # XY
    #     pts = torch.nan_to_num(pts)
    #     self.enemy_pos[env_ids, :, 0:2] = pts
    #     # Z：按高度区间均匀随机
    #     z = origins[:, 2:3].unsqueeze(1) + (hmin + torch.rand(N, E, 1, device=dev) * (hmax - hmin))
    #     self.enemy_pos[env_ids, :, 2:3] = z                                     # [N,E,1]
    #     if getattr(self.cfg, "function_time_print", False):
    #         self._cuda_sync_if_needed()
    #         dt_ms = (time.perf_counter() - t0) * 1000.0
    #         print(f"[TIME] _spawn_enemy : {dt_ms:.3f} ms")

    def _spawn_enemy(self, env_ids: torch.Tensor):
        """
        三种典型“来袭”队形的随机生成（批量 env):
        - triangle:   等边三角阵（尖角朝前）
        - wedge:      人字形 / V 阵（两翼展开，尖角朝前）
        - diamond:    菱形 / Vic
        每个 env:随机选队形 + 在环上随机中心 + 阵型朝向对准目标 self._goal_e
        保证阵内最近邻间距 >= cfg.enemy_min_separation
        写入:self.enemy_pos[env_ids] -> [N,E,3]
        """
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        # --- 设备与索引准备 ---
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

        # 敌群中心：以各 env 原点为圆心的环（可抖动）
        R_center = float(getattr(self.cfg, "enemy_cluster_ring_radius", 8.0))
        center_jitter = float(getattr(self.cfg, "enemy_center_jitter", 0.0)) # 敌机中心随机抖动

        # -------- 3 种队形模板（局部坐标，以 +X 为“前向”）--------
        def tmpl_triangle(E, s):
            # 等边三角形栅格：第 r 行有 r+1 个点，行距 dy = s*sqrt(3)/2，列距 = s
            if E == 0:
                return torch.zeros(0, 2, device=dev)
            dy = s * math.sqrt(3.0) / 2.0
            xs, ys, cnt = [], [], 0
            # R 满足 1+2+…+R >= E
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

        def tmpl_wedge(E, s):
            # 人字形（V 阵）：(0,0) 为尖角，两翼沿 ±45° 展开，步长 s/√2
            if E == 0:
                return torch.zeros(0, 2, device=dev)
            pts = [(0.0, 0.0)]
            step = s / math.sqrt(2.0)
            k = 1
            while len(pts) < E:
                pts.append(( k * step,  k * step))  # 右上
                if len(pts) >= E: break
                pts.append(( k * step, -k * step))  # 右下
                k += 1
            return torch.tensor(pts, dtype=torch.float32, device=dev)

        def tmpl_diamond(E, s):
            # 菱形/Vic：前/左/右/后 4 点；其余向 -X 方向排尾随
            if E == 0:
                return torch.zeros(0, 2, device=dev)
            d = s / math.sqrt(2.0)
            base = [( d, 0.0), (0.0,  d), (0.0, -d), (-d, 0.0)]
            if E <= 4:
                return torch.tensor(base[:E], dtype=torch.float32, device=dev)
            pts = base[:]
            k = 1
            while len(pts) < E:
                pts.append((-d - k * s, 0.0))
                k += 1
            return torch.tensor(pts, dtype=torch.float32, device=dev)

        # 构建并中心化模板（几何中心在 (0,0)）
        templates = []
        for builder in (tmpl_triangle, tmpl_wedge, tmpl_diamond): # 返回该队形在局部坐标系下的平面坐标 xy，形状是 [E, 2]（E 架敌机，每架给一个 (x, y)）
            xy = builder(E, s_min) # 生成某个队形的原始坐标（通常原点附近，间距至少为 s_min）
            xy = xy - xy.mean(dim=0, keepdim=True) # 把几何中心挪到 (0, 0)
            templates.append(xy) # 把该队形的中心化坐标保存起来
        templates = torch.stack(templates, dim=0)  # [3,E,2] 把三个 [E, 2] 的队形模板堆叠成一个张量
        F = templates.shape[0]  # 3

        # -------- 给每个环境各自随机挑一种队形，并把队形中心放到一条圆环上的随机位置 --------
        f_idx = torch.randint(low=0, high=F, size=(N,), device=dev)     # [N]
        local_xy = templates[f_idx, :, :]                               # [N,E,2]

        # 只保留三角阵
        # templates = []
        # for builder in (tmpl_triangle,):   # ← 这里只留一个
        #     xy = builder(E, s_min)
        #     xy = xy - xy.mean(dim=0, keepdim=True)
        #     templates.append(xy)
        # templates = torch.stack(templates, dim=0)  # [1, E, 2]
        # F = templates.shape[0]  # 1

        # 后面这两行依然成立：f_idx∈[0,1)，全为0，相当于选到同一个模板
        f_idx = torch.randint(low=0, high=F, size=(N,), device=dev)  # 全0
        local_xy = templates[f_idx, :, :]  # [N, E, 2]


        theta = 2.0 * math.pi * torch.rand(N, device=dev)
        centers = torch.stack([
            origins[:, 0] + R_center * torch.cos(theta),
            origins[:, 1] + R_center * torch.sin(theta)
        ], dim=1)                                                       # [N,2]
        if center_jitter > 0.0:
            centers = centers + (torch.rand(N, 2, device=dev) - 0.5) * (2.0 * center_jitter)

        # -------- 将局部 +X 朝向对齐到 center->goal 的方向（面向目标前进）--------
        goal_xy = goal_e[:, :2]                                         # [N,2]
        head_vec = goal_xy - centers                                    # [N,2]
        head = head_vec / head_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # 单位向量
        c, s = head[:, 0], head[:, 1]                                   # cos, sin
        # 旋转矩阵：把局部(+X, +Y)旋到 (head_x, head_y) 的朝向平面
        Rm = torch.stack([torch.stack([c, -s], dim=-1),
                        torch.stack([s,  c], dim=-1)], dim=1)         # [N,2,2]
        rotated = torch.matmul(local_xy, Rm.transpose(1, 2))             # [N,E,2]

        xy = centers.unsqueeze(1) + rotated                              # [N,E,2]

        # -------- 高度 --------
        z = origins[:, 2:3].unsqueeze(1) \
            + (hmin + torch.rand(N, E, 1, device=dev) * (hmax - hmin))  # => [N,E,1]
        enemy_pos = torch.cat([xy, z], dim=-1)  # [N,E,3]

        self.enemy_pos[env_ids] = enemy_pos

        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _spawn_enemy (triangle/wedge/diamond): envs={N}, E={E}, s_min={s_min:.2f}, R={R_center:.2f} -> {dt:.2f} ms")

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

                # ==== TRAJ VIS ====
                if getattr(self.cfg, "traj_vis_enable", False):
                    self._ensure_traj_markers()
                    for mk in self._traj_markers:
                        mk.set_visibility(True)

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
            if self._traj_markers:
                for mk in self._traj_markers:
                    mk.set_visibility(False)

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
    # --------- ↓↓↓↓↓云台控制相关↓↓↓↓↓ ---------
    def _ensure_fov_marker(self):
        """创建/缓存一个用于“点阵线框”的小方块 marker(用已导入的 isaaclab.markers)"""
        if self._fov_marker is not None:
            return
        cfg = CUBOID_MARKER_CFG.copy()
        cfg.prim_path = "/Visuals/GimbalFOV"  # 独立path，避免撞别的marker
        cfg.markers["cuboid"].size = tuple(getattr(self.cfg, "gimbal_vis_edge_size", (0.05, 0.05, 0.05)))
        cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.95, 0.2))
        self._fov_marker = VisualizationMarkers(cfg)
        self._fov_marker.set_visibility(True)

    def _dir_from_yaw_pitch(self, yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        # z-up: (cos p cos y, cos p sin y, sin p)
        cp = torch.cos(pitch); sp = torch.sin(pitch)
        cy = torch.cos(yaw);   sy = torch.sin(yaw)
        return torch.stack([cp * cy, cp * sy, sp], dim=-1)

    @torch.no_grad()
    def _update_gimbal_fov_vis(self):
        """用同一个“小方块”marker 将四棱锥FOV画(线框（顶点→四角 + 远平面四边）"""
        if not getattr(self.cfg, "gimbal_vis_enable", False):
            return
        self._ensure_fov_marker()
        if self._fov_marker is None:
            return

        dev   = self.device
        Ndraw = int(min(self.num_envs, getattr(self.cfg, "gimbal_vis_max_envs", 4)))
        if Ndraw <= 0:
            return

        # FOV 半角 & 远平面长度（=云台有效距离，更符合“能拍到”的物理直觉）
        half_h = 0.5 * math.radians(float(self.cfg.gimbal_fov_h_deg))   # 10°→5°
        half_v = 0.5 * math.radians(float(self.cfg.gimbal_fov_v_deg))   # 12°→6°
        R      = float(getattr(self.cfg, "gimbal_effective_range", 40.0))
        step   = float(getattr(self.cfg, "gimbal_vis_edge_step", 0.08))

        all_pts = []

        for ei in range(Ndraw):
            active = (~self.friend_frozen[ei])           # [M]
            if not torch.any(active):
                continue

            # 顶点（友机位置）与云台角
            P = self.fr_pos[ei][active]                  # [S,3]
            Y = self._gimbal_yaw[ei][active]             # [S]
            T = self._gimbal_pitch[ei][active]           # [S]
            S = P.shape[0]
            if S == 0:
                continue

            # 四个角方向 (±half_h, ±half_v) → 远平面四角
            yaws   = torch.stack([Y - half_h, Y - half_h, Y + half_h, Y + half_h], dim=1)  # [S,4]
            pitchs = torch.stack([T - half_v, T + half_v, T - half_v, T + half_v], dim=1)  # [S,4]
            dirs4  = self._dir_from_yaw_pitch(yaws, pitchs)                                 # [S,4,3]
            corners = P[:, None, :] + R * dirs4                                             # [S,4,3]

            # 4 条侧边（apex→四角）
            for k in range(4):
                A = P
                B = corners[:, k, :]
                for s in range(S):
                    all_pts.append(self._build_ray_dots(A[s], B[s], step))

            # 远平面四条边（0-1-3-2-0）
            for (a, b) in ((0,1),(1,3),(3,2),(2,0)):
                A = corners[:, a, :]
                B = corners[:, b, :]
                for s in range(S):
                    all_pts.append(self._build_ray_dots(A[s], B[s], step))

        if all_pts:
            self._fov_marker.visualize(translations=torch.cat(all_pts, dim=0))

    def _wrap_pi(self, x: torch.Tensor) -> torch.Tensor:
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    def _body_yaw_pitch_from_pose(self):
        # 用姿态 (theta, psi_v) 在 y-up/body 坐标系构造一个“单位前向”向量
        th = self.theta
        ps = self.psi_v
        Vxm = torch.cos(th) * torch.cos(ps)
        Vym = torch.sin(th)
        Vzm = -torch.cos(th) * torch.sin(ps)
        v_m = torch.stack([Vxm, Vym, Vzm], dim=-1)          # [N,M,3] in y-up/body

        # 变换到 z-up 世界坐标（与你推进里一致）
        v_w = y_up_to_z_up(v_m)

        # 复用现有的“由向量求 yaw/pitch”
        yaw, pitch, _ = self._body_yaw_pitch_from_vel(v_w)
        return yaw, pitch

    def _body_yaw_pitch_from_vel(self, vel: torch.Tensor, eps=1e-9):
        vx, vy, vz = vel[..., 0], vel[..., 1], vel[..., 2]  # z-up
        sp_xy = torch.sqrt((vx * vx + vy * vy).clamp_min(eps))
        yaw   = torch.atan2(vy, vx)
        pitch = torch.atan2(vz, sp_xy)
        return yaw, pitch, sp_xy

    def _step_gimbals_to_cover_enemies(self):
        dt = float(self.physics_dt)
        rg = math.radians(float(self.cfg.gimbal_range_deg))      # 机械限位（相对机体）
        max_rate = math.radians(float(self.cfg.gimbal_rate_deg)) # 最大角速度（rad/s）
        max_step = max_rate * dt
        # dt = 0.005s,max_step = 0.1°/step,每秒20度
        # print(f"dt: {dt} s")
        # print(f"max_step: {max_step*180/math.pi} deg per step")
        # 基准机体朝向（避免速度几乎为 0 时不稳定）
        body_yaw_v, body_pitch_v, sp_xy = self._body_yaw_pitch_from_vel(self.fr_vel_w)
        body_yaw_p, body_pitch_p = self._body_yaw_pitch_from_pose()
        slow = (sp_xy < 1e-3)

        body_yaw   = body_yaw_p
        body_pitch = body_pitch_p

        # 当前“相对机体”的云台角
        rel_y = self._wrap_pi(self._gimbal_yaw - body_yaw)   # 当前相对机体的 yaw
        rel_p = self._gimbal_pitch - body_pitch              # 当前相对机体的 pitch

        # 目标“相对机体”的云台角（来自网络，已限制在 ±rg）
        tgt_rel_y = self._gimbal_tgt_rel_yaw_cmd
        tgt_rel_p = self._gimbal_tgt_rel_pitch_cmd

        # 误差（yaw 用环角差，pitch 用线性差）
        err_y = self._wrap_pi(tgt_rel_y - rel_y)
        err_p = tgt_rel_p - rel_p

        # 速率限制更新（slew-rate limit）
        step_y = torch.clamp(err_y, -max_step, +max_step)
        step_p = torch.clamp(err_p, -max_step, +max_step)

        new_rel_y = (rel_y + step_y).clamp(-rg, rg)
        new_rel_p = (rel_p + step_p).clamp(-rg, rg)

        # 写回“绝对角”（保持连续，不 wrap，以避免出现跨 ±pi 的跳变）
        self._gimbal_yaw   = body_yaw + new_rel_y
        self._gimbal_pitch = body_pitch + new_rel_p
        # print(f"gimbal_yaw: {(self._gimbal_yaw)*180/math.pi}")
        # print(f"gimbal_pitch: {(self._gimbal_pitch)*180/math.pi}")
        # === NaN guards（与原实现一致，便于排错）===
        bad_body = (~torch.isfinite(body_yaw)) | (~torch.isfinite(body_pitch))
        if bad_body.any():
            nidx, midx = bad_body.nonzero(as_tuple=False)[0].tolist()
            print(
                f"[NaN body_angles] env={nidx} agent={midx}",
                f"body_yaw={body_yaw[nidx, midx].item():.6g}",
                f"body_pitch={body_pitch[nidx, midx].item():.6g}",
            )
        bad_new = (~torch.isfinite(self._gimbal_yaw)) | (~torch.isfinite(self._gimbal_pitch))
        if bad_new.any():
            nidx, midx = bad_new.nonzero(as_tuple=False)[0].tolist()
            print(
                f"[NaN gimbal_new] env={nidx} agent={midx}",
                f"yaw={self._gimbal_yaw[nidx, midx].item():.6g}",
                f"pitch={self._gimbal_pitch[nidx, midx].item():.6g}",
            )

        # 可视化（保留原逻辑）
        if getattr(self.cfg, "gimbal_vis_enable", False):
            k = int(getattr(self.cfg, "gimbal_vis_stride", 1))
            if (k <= 1) or (int(self.progress_buf[0].item()) % k == 0):
                self._update_gimbal_fov_vis()

    @torch.no_grad()
    def _gimbal_enemy_visible_mask(self) -> torch.Tensor:
        """同一云台：敌机是否被拍到（含作用距离 + 质心距离门限）→ [N,M,E]"""
        N, M, E = self.num_envs, self.M, self.E
        if E == 0:
            return torch.zeros(N, M, 0, dtype=torch.bool, device=self.device)
        eps = 1e-9
        half_h = 0.5 * math.radians(float(self.cfg.gimbal_fov_h_deg)) # 角度值转为弧度制
        half_v = 0.5 * math.radians(float(self.cfg.gimbal_fov_v_deg))
        Rcam   = float(self.cfg.gimbal_effective_range)

        rel = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)     # 敌机相对于云台的相对位置向量 [N,M,E,3]
        dx, dy, dz = rel[...,0], rel[...,1], rel[...,2]
        az  = torch.atan2(dy, dx)                                 # 方位角（azimuth）敌机相对于云台的水平偏角
        horiz = torch.sqrt((dx*dx + dy*dy).clamp_min(eps))       # 水平距离
        el  = torch.atan2(dz, horiz)                             # 俯仰角（elevation）
        dist = torch.linalg.norm(rel, dim=-1)                    # 欧氏距离

        gy = self._gimbal_yaw.unsqueeze(-1); gp = self._gimbal_pitch.unsqueeze(-1)
        dyaw   = torch.abs(self._wrap_pi(az - gy))
        dpitch = torch.abs(el - gp)
        in_fov = (dyaw <= half_h) & (dpitch <= half_v)
        in_rng = (dist <= Rcam)

        alive_e = (~self.enemy_frozen).unsqueeze(1)                 # [N,1,E]

        # === NaN guard: FOV inputs (az/el/dist) ===
        bad_fov_in = (~torch.isfinite(az)) | (~torch.isfinite(el)) | (~torch.isfinite(dist))
        if bad_fov_in.any():
            nidx, midx, kidx = bad_fov_in.nonzero(as_tuple=False)[0].tolist()
            print(
                f"[NaN gimbal_fov_in] env={nidx} agent={midx} enemy={kidx}",
                f"az={az[nidx, midx, kidx].item():.6g}",
                f"el={el[nidx, midx, kidx].item():.6g}",
                f"dist={dist[nidx, midx, kidx].item():.6g}",
            )

        gy = self._gimbal_yaw.unsqueeze(-1); gp = self._gimbal_pitch.unsqueeze(-1)

        # === NaN guard: gimbal angles in FOV ===
        bad_g_angles = (~torch.isfinite(gy)) | (~torch.isfinite(gp))
        if bad_g_angles.any():
            nidx, midx, kidx = bad_g_angles.nonzero(as_tuple=False)[0].tolist()
            print(
                f"[NaN gimbal_angles] env={nidx} agent={midx} enemy={kidx}",
                f"gy={gy[nidx, midx, kidx].item():.6g}",
                f"gp={gp[nidx, midx, kidx].item():.6g}",
            )

        return in_fov & in_rng & alive_e

    @torch.no_grad()
    def _gimbal_friend_visible_mask(self) -> torch.Tensor:
        """同一云台：友机是否被拍到（含作用距离；不受质心门限）→ [N,M,M] (i 看 j)
           i≠j、i与j都存活、j在i的视场角内、j在i的有效距离内"""
        N, M = self.num_envs, self.M
        if M <= 1:
            return torch.zeros(N, M, M, dtype=torch.bool, device=self.device)
        eps = 1e-9
        half_h = 0.5 * math.radians(float(self.cfg.gimbal_fov_h_deg))
        half_v = 0.5 * math.radians(float(self.cfg.gimbal_fov_v_deg))
        Rcam   = float(self.cfg.gimbal_effective_range)

        rel = self.fr_pos.unsqueeze(2) - self.fr_pos.unsqueeze(1)        # [N,M,M,3] i<-j
        dx, dy, dz = rel[...,0], rel[...,1], rel[...,2]
        az  = torch.atan2(dy, dx)
        horiz = torch.sqrt((dx*dx + dy*dy).clamp_min(eps))
        el  = torch.atan2(dz, horiz)
        dist = torch.linalg.norm(rel, dim=-1)

        gy = self._gimbal_yaw.unsqueeze(2).expand_as(az)
        gp = self._gimbal_pitch.unsqueeze(2).expand_as(el)
        dyaw   = torch.abs(self._wrap_pi(az - gy))
        dpitch = torch.abs(el - gp)
        in_fov = (dyaw <= half_h) & (dpitch <= half_v)
        in_rng = (dist <= Rcam)

        eye = torch.eye(M, dtype=torch.bool, device=self.device).unsqueeze(0).expand(N,-1,-1) # 排除自身
        alive = (~self.friend_frozen)
        return in_fov & in_rng & (~eye) & alive.unsqueeze(2) & alive.unsqueeze(1)

    # -------- ↑↑↑↑↑云台视角可视化相关↑↑↑↑↑ --------
    # --------- ↑↑↑↑↑云台控制相关↑↑↑↑↑ ---------
    # --------- ↓↓↓↓↓轨迹可视化区↓↓↓↓↓ ---------
    # ==== TRAJ VIS ====
    def _ensure_traj_markers(self):
        if self._traj_markers:
            return
        # 给不同友机配几种区分度高的颜色（循环使用）
        palette = [
            (0.15, 0.50, 1.00),  # 蓝
            (1.00, 0.30, 0.30),  # 红
            (0.20, 0.90, 0.60),  # 绿
            (0.95, 0.90, 0.20),  # 黄
            (0.60, 0.40, 1.00),  # 紫
            (1.00, 0.60, 0.20),  # 橙
            (0.20, 1.00, 1.00),  # 青
            (1.00, 0.40, 0.80),  # 粉
        ]
        for i in range(self.M):
            cfg = CUBOID_MARKER_CFG.copy()
            cfg.prim_path = f"/Visuals/Traj/Friend{i}"
            cfg.markers["cuboid"].size = tuple(getattr(self.cfg, "traj_marker_size", (0.05, 0.05, 0.05)))
            cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=palette[i % len(palette)]
            )
            mk = VisualizationMarkers(cfg)
            mk.set_visibility(True)
            self._traj_markers.append(mk)

    def _traj_reset(self, env_ids: torch.Tensor):
        # 重置这些 env 的轨迹缓存
        self._traj_len[env_ids] = 0
        self._traj_buf[env_ids] = 0.0

    def _update_traj_vis(self):
        """记录并绘制友方轨迹（仅前 K 个 env；按步抽样）。"""
        if not getattr(self.cfg, "traj_vis_enable", False):
            return
        # 步抽样
        stride = int(getattr(self.cfg, "traj_vis_every_n_steps", 1))
        if stride > 1 and ((self._sim_step_counter - 1) % stride != 0):
            return

        N_draw = int(min(self.num_envs, getattr(self.cfg, "traj_vis_max_envs", 4)))
        if N_draw <= 0:
            return
        self._ensure_traj_markers()

        K = int(getattr(self.cfg, "traj_vis_len", 200))

        # 1) 先把当前点写入循环缓冲
        for ei in range(N_draw):
            for mi in range(self.M):
                w = int(self._traj_len[ei, mi].item() % K)
                self._traj_buf[ei, mi, w, :] = self.fr_pos[ei, mi]
                self._traj_len[ei, mi] += 1

        # 2) 为每个友机收集其在前 N_draw 个 env 的历史点并下发 marker
        for mi in range(self.M):
            pts_list = []
            for ei in range(N_draw):
                n = int(self._traj_len[ei, mi].item())
                if n == 0:
                    continue
                L = min(n, K)
                buf = self._traj_buf[ei, mi, :K, :]  # [K,3]
                if n <= K:
                    pts = buf[:L, :]
                else:
                    # 循环缓冲展开为时间顺序（oldest -> newest）
                    start = n % K
                    pts = torch.cat([buf[start:K, :], buf[0:start, :]], dim=0)
                pts_list.append(pts)
            if pts_list:
                self._traj_markers[mi].visualize(translations=torch.cat(pts_list, dim=0))

    # --------- ↑↑↑↑↑轨迹可视化区↑↑↑↑↑ ---------
    # —————————————————— ↑↑↑↑↑工具区↑↑↑↑↑ ——————————————————
    # —————————————————— ↓↓↓↓↓主工作区↓↓↓↓↓ ——————————————————
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        控制输入预处理：
        - 支持 3 种形状的动作： [N, 3*M] / [N, M, 3] / [N, 3](广播到 M)
        - 对已冻结的配对(friend_frozen=True)屏蔽动作，不再对其施加控制
        - 将规范化动作映射到物理量:ny/nz ∈ [-1,1] -> g 值;throttle ∈ [0,1] -> Vm ∈ [Vm_min, Vm_max]
        """
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        if actions is None:
            return
        N = self.num_envs
        M = self.M

        # --- 统一动作形状为 [N, M, 5] ---
        if actions.dim() == 2:
            if actions.shape[1] == 5 * M:             # [N, 5*M]
                act = actions.view(N, M, 5)
            elif actions.shape[1] == 5:               # [N, 5] -> 广播到每个友机
                act = actions.view(N, 1, 5).expand(N, M, 5)
            else:
                raise RuntimeError(
                    f"Action shape mismatch. Got {tuple(actions.shape)}, "
                    f"expected [N,{5*M}] or [N,5]."
                )
        elif actions.dim() == 3:
            if actions.shape[1:] == (M, 5):           # [N, M, 5]
                act = actions
            elif actions.shape[1:] == (1, 5):         # [N, 1, 5] -> 广播到 M
                act = actions.expand(N, M, 5)
            else:
                raise RuntimeError(
                    f"Action shape mismatch. Got {tuple(actions.shape)}, "
                    f"expected [N,{M},5] or [N,1,5]."
                )
        else:
            raise RuntimeError(
                f"Action shape mismatch. Got dim={actions.dim()}, "
                f"expected 2D or 3D with last dim = 5."
            )
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

        # === 云台：第4/5维 → “目标相对角的增量”并积分为 cmd ===
        rg = math.radians(float(self.cfg.gimbal_range_deg))
        dt = float(self.physics_dt)
        tgt_rate = math.radians(float(getattr(self.cfg, "gimbal_tgt_rate_deg",
                                            self.cfg.gimbal_rate_deg)))   # 目标角每秒最大变化率（建议 ≤ gimbal_rate_deg）
        max_tgt_step = tgt_rate * dt
        deadband = math.radians(float(getattr(self.cfg, "gimbal_deadband_deg", 0.0))) # 目标角增量死区（度），0.3~1.0 可抑制细抖

        a_y = act[..., 3].clamp(-1.0, 1.0)
        a_p = act[..., 4].clamp(-1.0, 1.0)

        # 本步“目标角增量”（可选死区抑制抖动）
        d_y = a_y * max_tgt_step
        d_p = a_p * max_tgt_step
        if deadband > 0.0:
            d_y = torch.where(torch.abs(d_y) < deadband, torch.zeros_like(d_y), d_y)
            d_p = torch.where(torch.abs(d_p) < deadband, torch.zeros_like(d_p), d_p)

        # 若有冻结掩码，阻止冻结体更新目标（可选）
        if hasattr(self, "friend_frozen") and self.friend_frozen is not None:
            mask = (~self.friend_frozen).float()   # [N,M]
            d_y = d_y * mask
            d_p = d_p * mask

        # 积分到“目标相对角 cmd”，并夹到机械范围内
        self._gimbal_tgt_rel_yaw_cmd   = (self._gimbal_tgt_rel_yaw_cmd   + d_y).clamp(-rg, rg)
        self._gimbal_tgt_rel_pitch_cmd = (self._gimbal_tgt_rel_pitch_cmd + d_p).clamp(-rg, rg)

        # （可选，仅保留占位意义，便于调试对比）
        self._gimbal_cmd_yaw   = a_y
        self._gimbal_cmd_pitch = a_p

        # 打印时间
        if getattr(self.cfg, "function_time_print", False):
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
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        dt = float(self.physics_dt)
        # # ✅ 只在每个 RL 步的第一个子步清空“本步新冻结”缓冲
        # print("Sim step counter:", self._sim_step_counter)
        # print("Decimation:", self.cfg.decimation)
        # print("is_first_substep:", ((self._sim_step_counter - 1) % self.cfg.decimation) == 0)
        # print("Newly frozen enemy before:", self._newly_frozen_enemy)
        is_first_substep = ((self._sim_step_counter - 1) % self.cfg.decimation) == 0
        if is_first_substep:
            self._newly_frozen_friend[:] = False
            self._newly_frozen_enemy[:]  = False
        # print("Newly frozen enemy after:", self._newly_frozen_enemy)
        N, M, E = self.num_envs, self.M, self.E
        r = float(self.cfg.hit_radius)  # 1.0m

        # ---------- 0) 缓存步首状态,0.5)要判断是否命中 ----------
        fr_pos0 = self.fr_pos.clone()        # [N,M,3]
        en_pos0 = self.enemy_pos.clone()     # [N,E,3]
        fz0 = self.friend_frozen.clone()     # [N,M]
        ez0 = self.enemy_frozen.clone()      # [N,E]

        # ---------- 0.5) 步首命中：用 fr_pos0/en_pos0 判定 ----------
        # 仅考虑“未冻结的友/敌”对
        active_pair0 = (~fz0).unsqueeze(2) & (~ez0).unsqueeze(1)  # [N,M,E] 可以理解为一个M*E的矩阵，如果友机i与敌机j未冻结，则第i行第j列为True，否则为False
        if active_pair0.any():
            diff0 = fr_pos0.unsqueeze(2) - en_pos0.unsqueeze(1)   # [N,M,E,3]
            dist0 = torch.linalg.norm(diff0, dim=-1)              # [N,M,E]
            hit_pair0 = (dist0 <= r) & active_pair0               # [N,M,E]友机i、敌机j如果满足“未冻结且距离<=r”，则第i行第j列为True，否则为False

            # 命中的友/敌（任一配对命中即算）
            fr_hit0 = hit_pair0.any(dim=2)  # [N,M]对每个友机 i，只要它在任意一个敌机 j 上命中过（该行里有一个 True），fr_hit0[n, i] 就是 True
            en_hit0 = hit_pair0.any(dim=1)  # [N,E]对每个敌机 j，只要它被任意一个友机 i 命中过（该列里有一个 True），en_hit0[n, j] 就是 True

            # 本步新冻结标记（用于奖励）
            newly_fr = (~fz0) & fr_hit0 # [N, M]
            newly_en = (~ez0) & en_hit0 # [N, E]
            self._newly_frozen_friend |= newly_fr # 左右两边只要有一个是true则左边被赋值为true，布尔张量
            self._newly_frozen_enemy  |= newly_en

            # 敌机捕获点：取步首位置
            if newly_en.any():
                self.enemy_capture_pos[newly_en] = en_pos0[newly_en] # 在 PyTorch 里，用一个二维的布尔掩码去索引一个三维张量时，这个掩码会作用在前两个维度上

            # 友机捕获点：在其命中集合里选“最近的敌机（欧氏距离）”的步首位置。用“敌机索引”去取敌机的位置，再赋值给“这个友机”的捕获点；
            if newly_fr.any():
                INF = torch.tensor(float("inf"), device=self.device, dtype=dist0.dtype) # 创建无穷大值，用于屏蔽非命中对的距离
                dist_masked0 = torch.where(hit_pair0, dist0, INF)      # [N,M,E] 将非命中对的距离置为无穷大，保留命中对的实际距离
                j_star0 = dist_masked0.argmin(dim=2)                   # [N,M] 沿敌方维度找到每个友方物体命中的最近敌方物体的索引，找到最小的dist0，每个友机对应的“最近命中敌机的下标”
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

        # 计算完 fr_vel_w_step 之后
        badv = ~torch.isfinite(fr_vel_w_step).all(dim=-1)  # [N,M]
        if badv.any():
            nidx, midx = badv.nonzero(as_tuple=False)[0].tolist()
            print(f"[NaN vel] env={nidx} agent={midx}",
                f"Vm={self.Vm[nidx, midx].item():.6g}",
                f"ny={self._ny[nidx, midx].item():.6g}",
                f"nz={self._nz[nidx, midx].item():.6g}",
                f"theta={self.theta[nidx, midx].item():.6g}",
                f"psi={self.psi_v[nidx, midx].item():.6g}",
                f"fr_vel_w_step={fr_vel_w_step[nidx, midx]}")

        # ---------- 2) 敌机速度（冻结=0） ----------
        if self.cfg.enemy_seek_origin:
            # 刚体式队形平移：所有敌机同速同向（质心→目标）
            v_dir = (-self._axis_hat).unsqueeze(1).expand(-1, self.E, -1)     # [N,E,3]
            enemy_vel_step = v_dir * float(self.cfg.enemy_speed)              # [N,E,3]
        else:
            enemy_vel_step = self.enemy_vel
        enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)

        # ---------- 3) 推进到步末 ----------
        fr_pos1 = fr_pos0 + fr_vel_w_step * dt   # [N,M,3]
        en_pos1 = en_pos0 + enemy_vel_step * dt  # [N,E,3]


        # 写回前检查 fr_pos1 / en_pos1
        badp = ~torch.isfinite(fr_pos1).all(dim=-1)  # [N,M]
        if badp.any():
            nidx, midx = badp.nonzero(as_tuple=False)[0].tolist()
            print(f"[NaN pos] env={nidx} agent={midx}",
                f"fr_pos0={fr_pos0[nidx, midx]}",
                f"fr_vel_w_step={fr_vel_w_step[nidx, midx]}",
                f"dt={dt}")
            # 应急：避免扩散
            fr_pos1[nidx, midx] = fr_pos0[nidx, midx]

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

        # ---------- 5) 推进云台角 ----------
        self._step_gimbals_to_cover_enemies()

        # ---------- 6) 刷新缓存 + 可视化 ----------
        self._refresh_enemy_cache()

        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

        self._update_projection_debug_vis() # 投影可视化
        self._update_traj_vis()
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _apply_action (pre-step 1m check): {dt_ms:.3f} ms")

    def _get_rewards(self) -> torch.Tensor:
        """
        奖励：
        - centroid_approach: 鼓励友机靠近“当前存活敌机”的质心（距离减小为正）。仅计未冻结友机
        - hit_bonus        : 若任一友机与任一“存活敌机”的距离 <= hit_radius(1m)
                            则该敌机本回合记为“命中一次”（一次性发放，每敌机只发一次）
        - 云台拍摄到友机惩罚
        - 云台拍摄到敌机奖励
        """
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        N, M = self.num_envs, self.M

        # --- 配置权重 ---
        centroid_w  = float(getattr(self.cfg, "centroid_approach_weight", 1.0))
        hit_w       = float(getattr(self.cfg, "hit_reward_weight", 1000.0))

        # --- 活跃掩码 / 质心（用缓存） ---
        friend_active     = (~self.friend_frozen)                   # [N,M]
        enemy_active_any  = self._enemy_active_any                  # [N]
        friend_active_f   = friend_active.float()
        centroid          = self._enemy_centroid                    # [N,3]

        # --- 友机到质心的距离减小 ---
        c = centroid.unsqueeze(1).expand(N, M, 3)                  # [N,M,3]
        diff = c - self.fr_pos                                     # [N,M,3]
        dist_now = torch.linalg.norm(diff, dim=-1)                 # [N,M]

        if (not hasattr(self, "prev_dist_centroid")) or (self.prev_dist_centroid is None) \
        or (self.prev_dist_centroid.shape != dist_now.shape):
            self.prev_dist_centroid = dist_now.detach().clone()

        dist_now_safe = torch.where(enemy_active_any.unsqueeze(1), dist_now, self.prev_dist_centroid)
        d_delta_signed = self.prev_dist_centroid - dist_now_safe
        centroid_each = d_delta_signed * friend_active_f            # [N,M]
        base_each = centroid_w * centroid_each
        reward = base_each.sum(dim=1)                               # [N]

        # 保存分项
        self.closing_reward = base_each.sum(dim=1)                  # [N]

        # --- 命中奖励 ---
        new_hits_mask = self._newly_frozen_enemy                    # [N,E]
        hit_bonus = new_hits_mask.float().sum(dim=1) * hit_w        # [N]
        reward = reward + hit_bonus
        self.hit_reward = hit_bonus                                 # [N]
        self._newly_frozen_enemy[:]  = False
        self._newly_frozen_friend[:] = False

        # --- 云台看到友方惩罚 ---
        vis_ff = self._gimbal_friend_visible_mask().float()         # [N,M,M]
        pair_ff = torch.maximum(vis_ff, vis_ff.transpose(1,2))
        tri = torch.triu(torch.ones(self.M, self.M, dtype=torch.bool, device=self.device), diagonal=1).unsqueeze(0)
        pen_friend = (pair_ff * tri).sum(dim=(1,2))                 # [N]
        reward = reward - float(self.cfg.w_gimbal_friend_block) * pen_friend
        self.pen_friend_reward = - float(self.cfg.w_gimbal_friend_block) * pen_friend

        # --- 云台拍到敌机奖励 ---
        vis_fe = self._gimbal_enemy_visible_mask()                  # [N, M, E]
        enemy_count_each = vis_fe.sum(dim=-1).float()               # [N, M]
        alive_f = (~self.friend_frozen).float()
        enemy_count_each = enemy_count_each * alive_f
        env_cover = enemy_count_each.sum(dim=1)                     # [N]
        cover_reward = float(self.cfg.w_gimbal_enemy_cover) * env_cover
        reward = reward + cover_reward
        self.enemy_cover_reward = cover_reward

        # --- 统计累计 ---
        self.episode_sums.setdefault("centroid_approach", torch.zeros_like(centroid_each))
        self.episode_sums.setdefault("hit_bonus",         torch.zeros(self.num_envs, device=self.device, dtype=reward.dtype))
        self.episode_sums.setdefault("gimbal_friend_block", torch.zeros(self.num_envs, device=self.device, dtype=reward.dtype))
        self.episode_sums.setdefault("gimbal_enemy_cover",  torch.zeros(self.num_envs, device=self.device, dtype=reward.dtype))
        self.episode_sums["gimbal_friend_block"] += pen_friend
        self.episode_sums["centroid_approach"] += centroid_each
        self.episode_sums["hit_bonus"]         += hit_bonus
        self.episode_sums["gimbal_enemy_cover"]  += env_cover

        self.prev_dist_centroid = dist_now_safe

        if getattr(self.cfg, "function_time_print", False):
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
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        BIG = 1e9
        tol = float(getattr(self.cfg, "overshoot_tol", 0.4))
        r2_goal = float(self.cfg.enemy_goal_radius) ** 2          # 目标半径平方
        xy_max2 = 15.0 ** 2                                       # 越界半径平方

        if self._goal_e is None:
            self._rebuild_goal_e()

        # 1. 全部被拦截
        success_all_enemies = self.enemy_frozen.all(dim=1)        # [N]

        # 2. 友机 Z 越界
        z = self.fr_pos[..., 2]
        out_z_any = ((z < 0.0) | (z > 5.0)).any(dim=1)            # [N]（示例高度限制，可按需调整）

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

                # 掩码搭配：
                # - 友机要取 min → 非活体置 +∞
                # - 敌机要取 max → 非活体置 -∞
                INF     = torch.tensor(float("inf"),     dtype=sf.dtype, device=sf.device)
                NEG_INF = torch.tensor(float("-inf"),    dtype=sf.dtype, device=sf.device)

                sf_masked_for_min = torch.where(friend_active[k_idx], sf, INF)      # [K2,M]
                se_masked_for_max = torch.where(enemy_active[k_idx],  se, NEG_INF)  # [K2,E]

                friend_min = sf_masked_for_min.min(dim=1).values    # [K2]  友机最小投影
                enemy_max  = se_masked_for_max.max(dim=1).values    # [K2]  敌机最大投影

                # 你的需求：友机的最小位置 > 敌机的最大位置（留一点容差 tol）
                # 如果这是“越线/完全越过敌团”的判定：
                separated = friend_min > (enemy_max + tol)          # [K2], bool

                # 回填
                overshoot_any[idx[k_idx]] = separated
        #  汇总
        died = out_z_any | out_xy_any | nan_inf_any | success_all_enemies | enemy_goal_any | overshoot_any
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # print("episode_length_buf : ", self.episode_length_buf)
        # print("max_episode_length : ", self.max_episode_length)
        # print("out_z_any : ", out_z_any)
        # print("out_xy_any : ", out_xy_any)
        # print("nan_inf_any : ", nan_inf_any)
        # print("success_all_enemies : ", success_all_enemies)
        # print("enemy_goal_any : ", enemy_goal_any)
        # print("overshoot_any : ", overshoot_any)
        # print("time_out : ", time_out)
        #  降低 CPU 同步：把统计更新做成“可选且降频”
        log_every = int(getattr(self.cfg, "log_termination_every", 1))  # 0 表示关闭
        if log_every and (int(self.episode_length_buf.max().item()) % log_every == 0):
            term = self.extras.setdefault("termination", {})
            # 只做一次同步，把七个统计合在一起搬到 CPU
            stats = torch.stack([
                success_all_enemies.sum(),
                self.enemy_frozen.sum(),
                out_z_any.sum(),
                out_xy_any.sum(),
                nan_inf_any.sum(),
                enemy_goal_any.sum(),
                overshoot_any.sum(),
                time_out.sum(),
            ]).to("cpu")
            term.update({
                "success_envs":      int(stats[0]),
                "hit_total_enemies": int(stats[1]),
                "out_of_z_any":      int(stats[2]),
                "out_of_xy_any":     int(stats[3]),
                "nan_inf_any":       int(stats[4]),
                "enemy_goal_any":    int(stats[5]),
                "overshoot_any":     int(stats[6]),
                "time_out":          int(stats[7]),
            })
        # ==== 计时打印 ====
        if getattr(self.cfg, "function_time_print", False):
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
            8) 云台
        """
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()
        if not hasattr(self, "terrain"):
            self._setup_scene()
        if self._goal_e is None:
            self._rebuild_goal_e()
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if getattr(self.cfg, "per_train_data_print", False):
            # === 打印上一次 episode 的终止原因 ===
            if hasattr(self, "extras") and "termination" in self.extras:
                term = self.extras["termination"]
                print("\n--- Episode Termination Summary ---")
                for k, v in term.items():
                    print(f"{k:<20}: {v}")
                print("-----------------------------------")

            # === 在 episode 结束打印上一次累计的分项奖励 ===
            if hasattr(self, "episode_sums") and len(self.episode_sums) > 0:
                print("\n--- Episode Reward Summary ---")
                for k, v in self.episode_sums.items():
                    if v is not None and torch.is_tensor(v):
                        mean_val = v.mean().item() if v.numel() > 0 else 0.0
                        print(f"{k:<25}: {mean_val:8.3f}")
                # 小心 key 不存在
                total = 0.0
                for key in ["centroid_approach", "hit_bonus", "gimbal_friend_block", "gimbal_enemy_cover"]:
                    if key in self.episode_sums:
                        val = self.episode_sums[key]
                        total += val.sum().item() if key != "gimbal_friend_block" else -val.sum().item()
                print(f"{'TOTAL':<25}: {total:8.3f}")
                print("-----------------------------------\n")

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

        # 敌机在圆盘内随机出生（带最小间隔约束）
        self._spawn_enemy(env_ids)

        # 敌机初速度（环向寻标）
        phi = torch.rand(N, device=dev) * 2.0 * math.pi
        spd = float(self.cfg.enemy_speed)
        self.enemy_vel[env_ids, :, 0] = spd * torch.cos(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 1] = spd * torch.sin(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 2] = 0.0

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
        # spacing = float(getattr(self.cfg, "formation_spacing", 0.8))
        # idx = torch.arange(M, device=dev).float() - (M - 1) / 2.0
        # offsets_xy = torch.stack([torch.zeros_like(idx), idx * spacing], dim=-1)  # [M,2]
        # offsets_xy = offsets_xy.unsqueeze(0).expand(N, M, 2)                      # [N,M,2]
        # fr0 = torch.empty(N, M, 3, device=dev)
        # fr0[..., :2] = origins[:, :2].unsqueeze(1) + offsets_xy
        # fr0[...,  2] = origins[:, 2].unsqueeze(1) + float(self.cfg.flight_altitude)  # [N,1] -> broadcast 到 [N,M]
        # self.fr_pos[env_ids]  = fr0
        # self.fr_vel_w[env_ids] = 0.0

        # # 友方初始速度（自动面向团中心）/姿态
        # self.Vm[env_ids] = 0.0
        # en_pos = self.enemy_pos[env_ids]                # [N,E,3]
        # centroid = en_pos.mean(dim=1)                   # [N,3]

        # # 每个友机指向质心的相对向量（世界系 z-up）
        # rel_w = centroid.unsqueeze(1) - self.fr_pos[env_ids]   # [N,M,3]

        # # 转到机体使用的 y-up 表达，并单位化
        # rel_m = z_up_to_y_up(rel_w)
        # rel_m = rel_m / rel_m.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        # # 由方向向量解初始姿态
        # sin_th = rel_m[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        # theta0 = torch.asin(sin_th)
        # psi0   = torch.atan2(-rel_m[..., 2], rel_m[..., 0])
        # self.theta[env_ids] = theta0
        # self.psi_v[env_ids] = psi0
        # self._ny[env_ids] = 0.0
        # self._nz[env_ids] = 0.0

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

        # ---------------友方出生朝向---------------
        # 友方出生即面朝团中心一字排开
        eps = 1e-6
        spacing = float(getattr(self.cfg, "formation_spacing", 0.8))
        backoff = float(getattr(self.cfg, "formation_backoff", 0.0))

        # 仅用 XY 平面决定横队与朝向
        axis_hat_xy = self._axis_hat[env_ids, :2]                              # [N,2]
        # 面向质心的方向（原点 -> 质心）：-axis_hat
        face_xy = -axis_hat_xy
        face_norm = torch.linalg.norm(face_xy, dim=-1, keepdim=True).clamp_min(eps)
        f_hat = face_xy / face_norm                                            # [N,2] 归一化朝向（面向质心）

        # 横向排队方向 = 朝向的左法向量（旋转 90°）
        # r_hat = rot90ccw(f_hat) = [-fy, fx]
        r_hat = torch.stack([-f_hat[..., 1], f_hat[..., 0]], dim=-1)          # [N,2]

        # 以各 env 的 origin 作为队列中心（也可以换成你已有的出生中心）
        # origins: [N,3]（若你变量名不同，这里替换成你的“出生中心”）
        row_center = origins[:, :2]

        # 整条队列沿“远离质心”的方向后退 backoff
        if backoff > 0.0:
            row_center = row_center - backoff * f_hat

        # 对称编号：..., -2,-1,0,1,2,...
        idx = torch.arange(self.M, device=self.device).float() - (self.M - 1) / 2.0  # [M]
        idx = idx.view(1, self.M, 1)                                                   # [1,M,1]
        r_hat_exp = r_hat.unsqueeze(1)                                                 # [N,1,2]
        offsets_xy = idx * spacing * r_hat_exp                                         # [N,M,2]

        # 计算友机初始位置
        fr0 = torch.empty(len(env_ids), self.M, 3, device=self.device, dtype=self.fr_pos.dtype)
        fr0[..., :2] = row_center.unsqueeze(1) + offsets_xy                            # [N,M,2]
        fr0[...,  2] = origins[:,  2].unsqueeze(1) + float(self.cfg.flight_altitude)

        self.fr_pos[env_ids]  = fr0
        self.fr_vel_w[env_ids]= 0.0
        self.Vm[env_ids] = 0.0

        # 每架友机的初始航向：指向“敌团质心”，这里的psi0是机体系y-up表达即yaw
        # 友机指向质心（z-up 世界系）
        d = self._enemy_centroid[env_ids].unsqueeze(1) - self.fr_pos[env_ids]   # [N,M,3]

        # yaw（z-up）：atan2(y, x)，无需取负；再 wrap 到 [-pi, pi)
        psi0 = torch.atan2(d[..., 1], d[..., 0])
        psi0 = ((psi0 + math.pi) % (2.0 * math.pi)) - math.pi
        self.psi_v[env_ids] = psi0

        # pitch（y-up）：sin(theta) = z_w，所以用 z 分量
        d_m = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        sin_th = d_m[..., 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta0 = torch.asin(sin_th)
        self.theta[env_ids] = theta0

        # 初始化纵向过载等
        self._ny[env_ids] = 0.0
        self._nz[env_ids] = 0.0

        # ---------------友方出生朝向---------------

        # 云台重置
        self._gimbal_yaw[env_ids]   = psi0
        self._gimbal_pitch[env_ids] = theta0
        self._gimbal_tgt_rel_yaw_cmd[env_ids]   = 0.0
        self._gimbal_tgt_rel_pitch_cmd[env_ids] = 0.0

        # ==== TRAJ VIS ====
        self._traj_reset(env_ids)

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
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _reset_idx: {dt_ms:.3f} ms")

        bad = ~torch.isfinite(self.fr_pos[env_ids]).all(dim=-1)  # [N,M]
        if bad.any():
            nidx = env_ids[bad.any(dim=1).nonzero(as_tuple=False)[0,0]].item()
            midx = bad[nidx == env_ids].nonzero(as_tuple=False)[0,1].item()
            print(f"[NaN after reset] env={nidx} agent={midx} fr_pos={self.fr_pos[nidx, midx]}")

    def _get_observations(self) -> dict:
        """
        集中式观测（修改后）：
        对每个友机，拼接：
            [ fr_pos(3) | fr_vel_w(3) | enemy_centroid_pos(3) | e_hat_to_all_enemies(3*E, gated) ]
        其中：
        - enemy_centroid_pos(3) 为“当前存活敌机”的质心（世界系 z-up),对同一 env 下的每个友机相同；
        - e_hat_to_all_enemies 仍是基于“敌机 - 友机”的单位向量，但：
            * 对“已冻结敌机”置零（维度不变）；
            * 对“距离敌团中心 > cfg.e_hat_enable_radius 的友机”整段置零（这就是半径门控/gating)。
        最终把 M 个友机的观测串接： [N, M * (9 + 3E)]。
        """
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        eps = 1e-6
        N, M, E = self.num_envs, self.M, self.E

        # ------- 1) 基础量：友机位置/速度 -------
        fr_pos = self.fr_pos                  # [N,M,3]
        fr_vel = self.fr_vel_w                # [N,M,3]

        # ------- 2) 指向敌机的单位向量 -------
        if E > 0:
            # 全对全相对向量： enemy - friend  -> [N, M, E, 3]
            rel_all = self.enemy_pos.unsqueeze(1) - fr_pos.unsqueeze(2)        # [N,M,E,3]
            dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)  # [N,M,E,1]
            e_hat_all = rel_all / dist_all                                     # [N,M,E,3]

            # 屏蔽“已冻结敌机”的方向（置零，不改变维度）
            if hasattr(self, "enemy_frozen") and self.enemy_frozen is not None:
                enemy_active = (~self.enemy_frozen).unsqueeze(1).unsqueeze(-1).float()  # [N,1,E,1]
                e_hat_all = e_hat_all * enemy_active

            vis_fe = self._gimbal_enemy_visible_mask()  # True 表示第 j 个友机能看到第 k 个敌机
            e_hat_all = e_hat_all * vis_fe.unsqueeze(-1).float()
            # ------- 展平并拼接 -------
            e_hat_flat = e_hat_all.reshape(N, M, 3 * E)           # [N,M,3E]

        # ------- 3) 敌群质心（世界系）-------
        # 使用在 _apply_action / reset 中已维护的缓存 self._enemy_centroid:[N,3]
        centroid = self._enemy_centroid                                       # [N,3]
        centroid_per_friend = centroid.unsqueeze(1).expand(N, M, 3)           # [N,M,3]

        # 每个友机 9 + 3E 维
        obs_each = torch.cat([fr_pos, fr_vel, centroid_per_friend, e_hat_flat], dim=-1)  # [N,M, 9+3E]
        # 串接 M 个友机 -> [N, M*(9+3E)]
        obs = obs_each.reshape(N, -1)

        # ==== 计时打印 ====
        if getattr(self.cfg, "function_time_print", False):
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
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:Loitering_Munition_interception_swarm_skrl_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.Loitering_Munition_interception_swarm_rsl_rl_ppo_cfg:FASTInterceptSwarmPPORunnerCfg",
    },
)
