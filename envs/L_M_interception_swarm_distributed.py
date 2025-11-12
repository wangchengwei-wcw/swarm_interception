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
try:
    from isaaclab.markers import SPHERE_MARKER_CFG
    HAS_SPHERE_MARKER = True
except Exception:
    HAS_SPHERE_MARKER = False

from isaaclab.markers import GIMBAL_RAY_MARKER_CFG
from isaaclab.markers import Loitering_Munition_MARKER_CFG


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
    swarm_size: int = 6                 # 便捷参数：同时设置友机/敌机数量
    friendly_size: int = 6
    enemy_size: int = 6

    # 敌机出生区域（圆盘）与最小间隔
    debug_vis_enemy = True
    enemy_height_min = 3.0
    enemy_height_max = 10.0
    enemy_speed = 5.0
    enemy_seek_origin = True
    enemy_target_alt = 3.0
    enemy_goal_radius = 0.5
    enemy_cluster_ring_radius: float = 100.0  # 敌机的生成距离
    enemy_cluster_radius: float = 20.0        # 敌机团的半径(固定队形中未使用)
    enemy_min_separation: float = 5.0         # 敌机间最小水平间隔
    enemy_vertical_separation: float = 5.0    # 长方体队形敌机间最小垂直间隔
    enemy_center_jitter: float = 0.0        # 敌机团中心位置随机抖动幅度
    hit_radius = 0.3

    # 友方控制/速度范围/位置间隔
    Vm_min = 11
    Vm_max = 13
    ny_max_g = 3.0
    nz_max_g = 3.0
    formation_spacing = 2.0
    flight_altitude = 0.2

    # —— 单 agent 观测/动作维（用于 MARL 的 per-agent 空间）——
    single_observation_space: int = 9     # 将在 __post_init__ 基于 E 自动覆盖为 6 + 3E
    single_action_space: int = 5          # (ny, nz, throttle, gimbal_yaw_cmd, gimbal_pitch_cmd)

    # —— Multi-agent 所需的字典空间（在 __post_init__ 填充）——
    possible_agents: list[str] | None = None
    action_spaces: dict[str, int] | None = None
    observation_spaces: dict[str, int] | None = None

    # 奖励相关
    centroid_approach_weight = 0.5
    hit_reward_weight: float = 1500.0
    w_gimbal_friend_block: float = 100.0
    w_gimbal_enemy_cover:  float = 0.01
    intercept_alignment_weight: float = 0.04
    vel_to_centroid_weight: float = 0.01
    # —— vel→centroid 距离衰减
    vc_decay_alpha: float = 10.0   # 衰减尺度(米)：越大衰减越慢，远处权重更高
    vc_decay_power: float = 1.5    # 形状：=1 为双曲线；>1 让近处更快衰减
    vc_zero_inside: float = 10.0    # 近距离屏蔽半径(米)：d <= R0 时奖励≈0，避免干扰命中
    # 目标分散
    assign_diversity_weight: float = 0.25 # 惩罚权重 
    assign_beta: float = 6.0              # 软指向的“锐度”越大越“像 argmax”，小一些更平滑
    assign_angle_margin_deg: float = 10.0 # 认为“方向太像”的角阈(度)
    assign_prox_sigma: float = 5.0        # 空间接近权重的尺度(米)，彼此离得近才算“抢同一目标”；远处同向不算扎堆。

    # —— 云台 / FOV & 生效距离 ——
    gimbal_fov_h_deg: float = 10.0      # 水平总 FOV（度）
    gimbal_fov_v_deg: float = 12.0      # 垂直总 FOV（度）
    gimbal_range_deg: float = 30.0      # 相对机体限位 ±30°
    gimbal_rate_deg:  float = 20.0      # 角速度 20°/s
    gimbal_effective_range: float = 100.0  # 云台“有效拍摄距离”（米）

    # 频率
    episode_length_s = 40.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

    # for debug
    gimbal_vis_enable: bool = False          # 云台视野可视化开关
    traj_vis_enable: bool = False            # 轨迹可视化开关
    per_train_data_print: bool = True       # reset中打印
    function_time_print: bool = False        # 函数耗时打印
    gimbal_face_centroid: bool = True        # True: 自动指向敌团质心；False: 由策略的第4/5维控制
    gimbal_axis_vis_enable: bool = False       # 可视化云台光轴
    proj_vis_enable: bool = False            # 投影与射线可视化开关

    # —— 云台可视化（小方块点阵线框） ——
    gimbal_vis_max_envs: int = 1            # 只画前K个env，控性能
    gimbal_vis_edge_step: float = 0.12      # 边上点的采样间距（越小越像连续线）
    gimbal_vis_edge_size: tuple[float, float, float] = (0.05, 0.05, 0.05)  # 小方块尺寸

    # === 投影与射线可视化 ===
    proj_max_envs: int = 1
    proj_ray_step: float = 0.2
    proj_ray_size: tuple[float,float,float] = (0.08, 0.08, 0.08)
    proj_friend_size: tuple[float,float,float] = (0.12, 0.12, 0.12)
    proj_enemy_size:  tuple[float,float,float] = (0.12, 0.12, 0.12)
    proj_centroid_size: tuple[float,float,float] = (0.16, 0.16, 0.16)

    # ==== TRAJ VIS ==== 友方轨迹可视化
    traj_vis_max_envs: int = 1              # 只画前几个 env
    traj_vis_len: int = 500                 # 每个友机最多保留多少个轨迹点（循环缓冲）
    traj_vis_every_n_steps: int = 2         # 每隔多少个物理步记录/刷新一次
    traj_marker_size: tuple[float,float,float] = (0.05, 0.05, 0.05)  # 面包屑小方块尺寸

    # === Action Mask（连续动作的分量屏蔽）===
    action_mask_enable: bool = True          # 开/关
    mask_yaw_err_deg: float = 40.0           # |yaw误差| > 此阈值时，屏蔽“错误号”的 nz
    mask_yaw_err_hard_deg: float = 110.0     # |yaw误差| > 此阈值时，直接把 throttle 打 0
    mask_outer_band_ratio: float = 1.2       # 外包圈半径 = 敌机到质心的最大半径 * 该比例
    mask_outer_stop: bool = True             # 在外包圈且径向外飘时，是否将油门打 0

    # === 敌机球体可视化（以拦截半径为尺寸） ===
    enemy_render_as_sphere: bool = True
    enemy_sphere_color: tuple[float, float, float] = (1.0, 0.3, 0.3)  # 红色

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

    # —— 关键：类级别一定要是“可序列化”的 Gym Space（占位即可）——
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space      = spaces.Box(low=-1.0,   high=1.0,   shape=(1,), dtype=np.float32)
    state_space       = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    clip_action       = 1.0

    # 多智能体的 spaces：类级别给“空字典”，让序列化器能安全遍历（得到 {}）
    possible_agents: list[str] = []
    action_spaces: dict[str, gym.Space] = {}
    observation_spaces: dict[str, gym.Space] = {}

    def __post_init__(self):
        M = self.friendly_size if getattr(self, "friendly_size", None) is not None else self.swarm_size
        E = self.enemy_size    if getattr(self, "enemy_size", None)    is not None else self.swarm_size

        # agent 名单
        self.possible_agents = [f"drone_{i}" for i in range(int(M))]

        # 单智能体维度（与原集中式每机位 obs 结构一致）
        single_obs_dim = 6 * int(M) + 3 * int(E)
        single_act_dim = 3                  # (ny, nz, throttle)

        # 可选：在 cfg 层先放一个“数字”，真正的 gym.Space 在 Env.__init__ 里构造
        self.single_observation_space = single_obs_dim
        self.single_action_space = single_act_dim

        # 也可在 cfg 层放 per-agent 的“维度字典”（同样只是数字）
        self.observation_spaces = {ag: single_obs_dim for ag in self.possible_agents}
        self.action_spaces      = {ag: single_act_dim for ag in self.possible_agents}

class FastInterceptionSwarmMARLEnv(DirectMARLEnv):
    """多智能体（分布式）版拦截环境：按 agent 字典进行 obs/action/reward/done 交互。"""
    cfg: FastInterceptionSwarmMARLCfg
    _is_closed = True

    def __init__(self, cfg: FastInterceptionSwarmMARLCfg, render_mode: str | None = None, **kwargs):
        # ------------------ 维度与空间 ------------------
        M = cfg.friendly_size if cfg.friendly_size is not None else cfg.swarm_size
        E = cfg.enemy_size    if cfg.enemy_size    is not None else cfg.swarm_size
        act_dim = int(cfg.single_action_space)                 # 已有的动作维
        # single_obs_dim = 9 + 3 * int(E)                        # 位置3+速度3+中心3+逐敌3E
        single_obs_dim = 6 * int(M) + 3 * int(E)

        # 兼容：有些下游脚本把这个字段当“维度”用
        cfg.single_observation_dim = single_obs_dim
        cfg.single_observation_space = single_obs_dim  # <- 保留旧别名（int），不与 gym.Space 混淆

        # —— Multi-agent spaces（挂到 cfg，供 base/shell wrappers 读取）——
        agents = [f"drone_{i}" for i in range(M)]
        cfg.possible_agents = agents

        ma_act_space  = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,),        dtype=np.float32)
        ma_obs_space  = spaces.Box(low=-np.inf, high=np.inf, shape=(single_obs_dim,), dtype=np.float32)

        cfg.action_spaces      = {a: ma_act_space for a in agents}
        cfg.observation_spaces = {a: ma_obs_space for a in agents}

        # 集中式 state（例如 MAPPO），按需启用
        cfg.state_space = spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(M * single_obs_dim,), dtype=np.float32)

        # 占位的“单智能体”Space（部分工具链会读取）
        cfg.action_space      = ma_act_space
        cfg.observation_space = ma_obs_space

        # 让父类完成 IsaacLab 的设备/并行环境初始化
        super().__init__(cfg, render_mode, **kwargs)
        self._is_closed = False

        # ------------------ 基本属性与空间引用 ------------------
        self.is_multi_agent      = True
        self.possible_agents     = list(cfg.possible_agents)           # list[str]
        self.action_spaces       = cfg.action_spaces                   # dict[str, Space]
        self.observation_spaces  = cfg.observation_spaces              # dict[str, Space]
        self.single_action_space = cfg.action_space                    # Space
        self.single_observation_space = cfg.observation_space          # Space
        self._profile_print = bool(getattr(self.cfg, "profile_print", False))
        # 兼容：部分包装器读取这两个别名
        self.action_space      = self.single_action_space
        self.observation_space = self.single_observation_space
        if getattr(cfg, "state_space", None) is not None:
            self.state_space = cfg.state_space

        # ------------------ 尺寸/设备/类型 ------------------
        self.M = int(M)
        self.E = int(E)
        N      = self.num_envs
        dev    = self.device
        dtype  = torch.float32

        # ------------------ 云台角（同一云台） ------------------
        self._gimbal_yaw   = torch.zeros(N, self.M, device=dev, dtype=dtype)
        self._gimbal_pitch = torch.zeros(N, self.M, device=dev, dtype=dtype)

        # ------------------ 友/敌状态与动力学 ------------------
        self.fr_pos   = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)
        self.fr_vel_w = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)

        self.enemy_pos = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)
        self.enemy_vel = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)

        self.g0    = 9.81
        self.theta = torch.zeros(N, self.M, device=dev, dtype=dtype)
        self.psi_v = torch.zeros(N, self.M, device=dev, dtype=dtype)
        self.Vm    = torch.zeros(N, self.M, device=dev, dtype=dtype)
        self._ny   = torch.zeros(N, self.M, device=dev, dtype=dtype)
        self._nz   = torch.zeros(N, self.M, device=dev, dtype=dtype)

        # ------------------ 冻结/命中缓存 ------------------
        self.friend_frozen       = torch.zeros(N, self.M, device=dev, dtype=torch.bool)
        self.enemy_frozen        = torch.zeros(N, self.E, device=dev, dtype=torch.bool)
        self.friend_capture_pos  = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)
        self.enemy_capture_pos   = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)

        # ------------------ 统计与一次性事件 ------------------
        self.episode_sums = {}
        self._newly_frozen_friend = torch.zeros(N, self.M, dtype=torch.bool, device=dev)
        self._newly_frozen_enemy  = torch.zeros(N, self.E, dtype=torch.bool, device=dev)

        # ==== TRAJ VIS ====
        self._traj_buf  = torch.zeros(self.num_envs, self.M, int(self.cfg.traj_vis_len), 3,
                                      device=dev, dtype=dtype)  # [N,M,K,3]
        self._traj_len  = torch.zeros(self.num_envs, self.M, device=dev, dtype=torch.long) # [N,M]
        self._traj_markers: list[VisualizationMarkers] = []

        # —— 云台角（同一个云台服务奖励与obs gating） ——
        self._gimbal_yaw   = torch.zeros(N, self.M, device=dev, dtype=dtype)  # [-pi,pi)
        self._gimbal_pitch = torch.zeros(N, self.M, device=dev, dtype=dtype)  # 仰角
        self._gimbal_tgt_rel_yaw_cmd   = torch.zeros(N, self.M, device=dev)
        self._gimbal_tgt_rel_pitch_cmd = torch.zeros(N, self.M, device=dev)

        # ------------------ 可视化与调试 ------------------
        self.friendly_visualizer = None
        self.enemy_visualizer    = None
        self.centroid_marker     = None
        self.ray_marker          = None
        self.friend_proj_marker  = None
        self.enemy_proj_marker   = None
        self._fov_marker         = None
        self._traj_markers = []  # per-friend trajectory markers
        self.set_debug_vis(self.cfg.debug_vis)

        # ------------------ 敌团缓存（每步更新） ------------------
        self._enemy_centroid_init = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._enemy_centroid      = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._enemy_active        = torch.zeros(N, self.E, device=dev, dtype=torch.bool)
        self._enemy_active_any    = torch.zeros(N, device=dev, dtype=torch.bool)
        self._goal_e              = None
        self._axis_hat            = torch.zeros(N, 3, device=dev, dtype=dtype)

    # —————————————————— ↓↓↓↓↓工具/可视化区↓↓↓↓↓ ——————————————————
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

    def _spawn_enemy_random(self, env_ids: torch.Tensor):
        """在指定环境中，使用泊松盘采样生成敌机位置。"""
        if getattr(self.cfg, "function_time_print", False):
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

        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _spawn_enemy_random : {dt_ms:.3f} ms")

    def _spawn_enemy(self, env_ids: torch.Tensor):
        """
        四种来袭队形（批量 env,完全并行）：
        0: v_wedge_2d  1: rect_2d  2: rect_3d  3: cube_3d
        每个 env 随机挑一种，放在以 env 原点为圆心的环上；全部使用 E=self.E 架敌机。
        写入: self.enemy_pos[env_ids] -> [N,E,3]
        """
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        # ---- 基本量 ----
        # 确保后续所有新建张量都在同一设备与 dtype 上，避免 CPU/GPU 混放
        dev   = self.fr_pos.device
        dtype = self.fr_pos.dtype
        env_ids = env_ids.to(dtype=torch.long, device=dev)

        origins_all = self.terrain.env_origins
        if origins_all.device != dev:
            origins_all = origins_all.to(dev)
        origins = origins_all[env_ids]  # [N,3]

        if self._goal_e is None:
            self._rebuild_goal_e()
        goal_e = self._goal_e[env_ids]  # [N,3]

        N, E = env_ids.shape[0], int(self.E)
        s_min = float(self.cfg.enemy_min_separation)
        sz_v  = float(getattr(self.cfg, "enemy_vertical_separation", s_min))
        hmin  = float(self.cfg.enemy_height_min)
        hmax  = float(self.cfg.enemy_height_max)
        R_center = float(getattr(self.cfg, "enemy_cluster_ring_radius", 8.0))
        center_jitter = float(getattr(self.cfg, "enemy_center_jitter", 0.0))

        # ---- 工具：中心化、网格 ----
        def _centerize(xyz: torch.Tensor) -> torch.Tensor:
            # 将输入的点云张量中心化，通过减去点的坐标均值，使点云的质心移到原点 (0, 0, 0)
            return xyz - xyz.mean(dim=-2, keepdim=True)

        def _rect2d_dims(E: int, aspect_w: float = 2.0) -> tuple[int, int]:
            # 计算二维网格的行数和列数，满足近似的宽高比 aspect_w（宽/高），并确保总点数 rows * cols >= E
            cols = max(1, int(math.ceil(math.sqrt(E * max(1e-3, aspect_w)))))
            rows = int(math.ceil(E / cols))
            return rows, cols

        def _grid2d(rows: int, cols: int, s: float) -> torch.Tensor:
            # 生成一个 rows 行 cols 列的二维规则网格点云，点间距为 s，z 坐标为 0，并中心化
            xs = torch.arange(cols, dtype=dtype, device=dev) # 列索引
            ys = torch.arange(rows, dtype=dtype, device=dev) # 行索引
            X, Y = torch.meshgrid(xs, ys, indexing="xy")  # 使用 torch.meshgrid 生成二维网格坐标，indexing="xy" 表示按矩阵索引（x 对应列，y 对应行）
            X = X.t().reshape(-1) # X.t() 和 Y.t() 转置张量，变成 [cols, rows]
            Y = Y.t().reshape(-1)
            xyz = torch.stack([X * s, Y * s, torch.zeros_like(X)], dim=-1)  # [Emax,3]
            return _centerize(xyz)

        def _grid3d(rows: int, cols: int, layers: int, sx: float, sy: float, sz_: float) -> torch.Tensor:
            # 返回 [Emax,3]
            xs = torch.arange(cols,   dtype=dtype, device=dev)
            ys = torch.arange(rows,   dtype=dtype, device=dev)
            zs = torch.arange(layers, dtype=dtype, device=dev)
            X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="xy")
            X = X.permute(1, 0, 2).reshape(-1)
            Y = Y.permute(1, 0, 2).reshape(-1)
            Z = Z.permute(1, 0, 2).reshape(-1)
            xyz = torch.stack([X * sx, Y * sy, Z * sz_], dim=-1)  # [Emax,3]
            return _centerize(xyz)

        def _best_rc(cap_layer: int, aspect_xy: float = 2.0) -> tuple[int, int]:
            """找 r, c 使 r*c >= cap_layer，且 c/r ~ aspect_xy，优先最小面积冗余，其次形状接近。"""
            aspect_xy = max(1e-6, float(aspect_xy))
            best = None
            best_rc = (1, cap_layer)
            for r in range(1, cap_layer + 1):
                c = math.ceil(cap_layer / r)
                area_over = r * c - cap_layer          # 冗余格子越少越好
                aspect_err = abs((c / r) - aspect_xy)  # 形状越接近越好
                score = (area_over, aspect_err)
                if best is None or score < best:
                    best = score
                    best_rc = (r, c)
            return best_rc
        # ---- 四种模板（局部坐标，+X 为前向；每个模板恰好 E 个点）----
        # 0) V 字形（平面，尖角朝 +X）
        def _tmpl_v_wedge_2d(E: int, s: float) -> torch.Tensor:
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)
            step = s / math.sqrt(2.0)
            if E == 1:
                pts = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=dev)
                return pts
            K = (E - 1) // 2                      # 完整的成对层数，//表示两个数相除向下取整5//2=2.
            ks = torch.arange(1, K + 1, dtype=dtype, device=dev) # 从 1 到 K 的张量，表示层的索引
            # K 对 (±)
            up   = torch.stack([-ks * step,  ks * step, torch.zeros_like(ks)], dim=-1) # 生成上半部分的点，坐标为 (k*step, k*step, 0)，沿 y=x 方向（45 度线）
            down = torch.stack([-ks * step, -ks * step, torch.zeros_like(ks)], dim=-1) # 生成下半部分的点
            pts = torch.cat([torch.zeros(1, 3, dtype=dtype, device=dev), up, down], dim=0)  # [1+2K,3] 合并中心点和对称点
            # 若还有 1 架余数（E 为偶数），补在右上
            if (E - 1) % 2 == 1:
                extra_k = torch.tensor([(K + 1) * step], dtype=dtype, device=dev)
                extra   = torch.stack([-extra_k, extra_k, torch.zeros_like(extra_k)], dim=-1)  # [1,3]
                pts = torch.cat([pts, extra], dim=0)
            return _centerize(pts[:E, :])

        # 1) 平面长方形（长边沿 +X）
        def _tmpl_rect_2d(E: int, s: float, aspect: float = 2.0) -> torch.Tensor:
            r, c = _rect2d_dims(E, aspect)
            xyz = _grid2d(r, c, s)[:E, :]
            return xyz

        # 2) 立体长方体（xy 近似矩形，z 多层）
        def _tmpl_rect_3d(E: int, s: float, sz_: float, aspect_xy: float = 2.0) -> torch.Tensor:
            # 固定两层
            L = 2
            # 单层需要的容量
            cap_layer = max(1, math.ceil(E / L))
            # 计算每层行列数（两层相同）
            r, c = _best_rc(cap_layer, aspect_xy)
            # 若需要知道每层实际填充数量：
            # n0 = min(r * c, math.ceil(E / 2))
            # n1 = E - n0
            xyz = _grid3d(r, c, L, s, s, sz_)[:E, :]
            return xyz

        # 3) 立体正方体（尽量 n≈E^(1/3)）
        # def _tmpl_cube_3d(E: int, s: float, sz_: float) -> torch.Tensor:
        #     n = int(math.ceil(E ** (1.0 / 3.0)))                # 计算三维网格的边长 n
        #     xyz = _grid3d(n, n, n, s, s, sz_)[:E, :]
        #     return xyz

        def _tmpl_rect_3d_reverse(E: int, s: float, sz_: float, aspect_xy: float = 2.0) -> torch.Tensor:
            # 固定两层
            L = 2
            # 单层需要的容量
            cap_layer = max(1, math.ceil(E / L))
            # 计算每层行列数（两层相同）
            r, c = _best_rc(cap_layer, aspect_xy)
            # 若需要知道每层实际填充数量：
            # n0 = min(r * c, math.ceil(E / 2))
            # n1 = E - n0
            xyz = _grid3d(c, r, L, s, s, sz_)[:E, :]
            return xyz

        # 组装为 [F,E,3]，一次性并行给所有 env 复用
        templates = torch.stack([
            _tmpl_v_wedge_2d(E, s_min),                 # 0
            _tmpl_rect_2d(E, s_min, aspect=2.0),        # 1
            _tmpl_rect_3d(E, s_min, sz_v,  aspect_xy=2.0),  # 2
            # _tmpl_cube_3d(E, s_min, s_min)               # 3
            _tmpl_rect_3d_reverse(E, s_min, sz_v,  aspect_xy=2.0)
        ], dim=0)  # [4,E,3]

        # ---- 每个 env 随机挑一种队形（并行索引）----
        f_idx = torch.randint(low=0, high=templates.shape[0], size=(N,), device=dev)  # [N]
        local_xyz = templates[f_idx, :, :]  # [N,E,3]  <- 并行 gather，无 per-env 循环

        # ---- 计算环上中心（并行）----
        theta = 2.0 * math.pi * torch.rand(N, device=dev, dtype=dtype)
        centers = torch.stack([
            origins[:, 0] + R_center * torch.cos(theta),
            origins[:, 1] + R_center * torch.sin(theta)
        ], dim=1)  # [N,2]
        if center_jitter > 0.0:
            centers = centers + (torch.rand(N, 2, device=dev, dtype=dtype) - 0.5) * (2.0 * center_jitter)

        # ---- 将局部 +X 旋到 center->goal 的方向（只旋 XY；全部并行）----
        head_vec = (goal_e[:, :2] - centers)                           # [N,2]
        head = head_vec / head_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        c, s = head[:, 0], head[:, 1]                                   # cos, sin
        # 旋转矩阵 R = [[c,-s],[s,c]]，批量乘 local_xyz[..., :2]
        Rm = torch.stack([
            torch.stack([c, -s], dim=-1),   # [N,2]
            torch.stack([s,  c], dim=-1)
        ], dim=1)  # [N,2,2]
        xy_rot = torch.matmul(local_xyz[:, :, :2], Rm.transpose(1, 2))  # [N,E,2]
        xy = centers.unsqueeze(1) + xy_rot                               # [N,E,2]

        # ---- 高度（并行）：为每个 env 采样基线 z0，再叠加局部 z，并夹到 [hmin,hmax] ----
        z0 = hmin + torch.rand(N, 1, 1, device=dev, dtype=dtype) * max(1e-6, (hmax - hmin))
        local_z = local_xyz[:, :, 2:3]                    # [N,E,1]（平面队形时为 0）
        z_rel   = (z0 + local_z).clamp(hmin, hmax)        # 相对 origins[:,2]
        z_abs   = origins[:, 2:3].unsqueeze(1) + z_rel    # 绝对高度 [N,E,1]

        enemy_pos = torch.cat([xy, z_abs], dim=-1)        # [N,E,3]
        self.enemy_pos[env_ids] = enemy_pos               # 写回（一次性）

        # 可选：记录队形索引用于统计/可视化
        try:
            if not hasattr(self, "_enemy_formation_idx"):
                self._enemy_formation_idx = torch.full((self.num_envs,), -1, device=dev, dtype=torch.long)
            self._enemy_formation_idx[env_ids] = f_idx
        except Exception:
            pass

        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _spawn_enemy (V/rect2d/rect3d/cube3d): envs={N}, E={E}, s_min={s_min:.2f}, R={R_center:.2f} -> {dt_ms:.2f} ms")

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
                # from isaaclab.markers import VisualizationMarkers, Loitering_Munition_MARKER_CFG
                # f_cfg = Loitering_Munition_MARKER_CFG.copy()
                # f_cfg.prim_path = "/Visuals/FriendlyModel"  # 每类 marker 建议单独命名路径
                # # 如果你想额外调节缩放或材质，这里也可以覆盖：
                # f_cfg.markers["mymodel"].scale = (10.5, 10.5, 10.5)
                # # 创建 USD 模型 marker
                # self.friendly_visualizer = VisualizationMarkers(f_cfg)
                # self.friendly_visualizer.set_visibility(True)
                if HAS_AXIS_MARKER and AXIS_MARKER_CFG is not None:
                    f_cfg = AXIS_MARKER_CFG.copy()
                    f_cfg.prim_path = "/Visuals/FriendlyAxis"
                    f_cfg.markers["frame"].scale = (1, 1, 1)
                    self.friendly_visualizer = VisualizationMarkers(f_cfg)
                else:
                    f_cfg = CUBOID_MARKER_CFG.copy()
                    f_cfg.markers["cuboid"].size = (0.18, 0.18, 0.18)
                    f_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 1.0))
                    f_cfg.prim_path = "/Visuals/Friendly"
                    self.friendly_visualizer = VisualizationMarkers(f_cfg)
                self.friendly_visualizer.set_visibility(True)

            if self.cfg.debug_vis_enemy and self.enemy_visualizer is None:
                if getattr(self.cfg, "enemy_render_as_sphere", False) and HAS_SPHERE_MARKER:
                    # —— 敌机球体可视化（半径 = hit_radius）——
                    e_cfg = SPHERE_MARKER_CFG.copy()
                    # 颜色
                    e_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(getattr(self.cfg, "enemy_sphere_color", (1.0, 0.3, 0.3)))
                    )
                    # 半径 = 拦截半径
                    if hasattr(e_cfg.markers["sphere"], "radius"):
                        e_cfg.markers["sphere"].radius = float(self.cfg.hit_radius)
                    else:
                        # 个别发行版不暴露 radius，就用整体缩放把单位球(半径0.5)放大到 hit_radius
                        s = 2.0 * float(self.cfg.hit_radius)
                        if hasattr(e_cfg.markers["sphere"], "scale"):
                            e_cfg.markers["sphere"].scale = (s, s, s)
                    e_cfg.prim_path = "/Visuals/Enemy"
                    self.enemy_visualizer = VisualizationMarkers(e_cfg)
                    self.enemy_visualizer.set_visibility(True)
                    self._enemy_vis_kind = "sphere"
                else:
                    # —— 后备：仍用原来的方块 —— 
                    e_cfg = CUBOID_MARKER_CFG.copy()
                    e_cfg.markers["cuboid"].size = (1.5, 1.5, 1.5)
                    e_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                    e_cfg.prim_path = "/Visuals/Enemy"
                    self.enemy_visualizer = VisualizationMarkers(e_cfg)
                    self.enemy_visualizer.set_visibility(True)
                    self._enemy_vis_kind = "cuboid"

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
        # if self.friendly_visualizer is not None:
        #     pos = self.fr_pos.reshape(-1, 3)  # [N*M,3]
        #     quat = torch.zeros(pos.shape[0], 4, device=self.device)
        #     quat[:, 0] = 1.0  # 单位四元数 (w=1,x=0,y=0,z=0)
        #     scale = torch.ones_like(pos)
        #     self.friendly_visualizer.visualize(translations=pos, orientations=quat, scales=scale)
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

    def _ensure_fov_marker(self):
        """创建/缓存一个用于“点阵线框”的小方块 marker(用已导入的 isaaclab.markers)"""
        if self._fov_marker is not None:
            return
        cfg = CUBOID_MARKER_CFG.copy()
        cfg.prim_path = "/Visuals/GimbalFOV"  # 独立path，避免撞别的marker
        cfg.markers["cuboid"].size = tuple(getattr(self.cfg, "gimbal_vis_edge_size", (0.02, 0.02, 0.02)))
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
        """用圆柱射线绘制云台四棱锥 FOV 线框(4 条侧边 + 远平面矩形）"""
        if not getattr(self.cfg, "gimbal_vis_enable", False):
            return

        # === 确保 marker ===
        if not hasattr(self, "_gimbal_fov_ray_marker") or self._gimbal_fov_ray_marker is None:
            from isaaclab.markers import VisualizationMarkers
            self._gimbal_fov_ray_marker = VisualizationMarkers(GIMBAL_RAY_MARKER_CFG)
            self._gimbal_fov_ray_marker.set_visibility(True)

        def _quat_wxyz_from_z_to_dir(d: torch.Tensor) -> torch.Tensor:
            """把局部 z 轴旋到方向 d(世界系单位向量)，返回 (w,x,y,z)"""
            d = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-8)
            z = torch.tensor([0.0, 0.0, 1.0], device=d.device, dtype=d.dtype).expand_as(d)
            v = torch.cross(z, d)
            c = (z * d).sum(dim=-1, keepdim=True)
            eps = 1e-8
            q = torch.zeros(d.shape[0], 4, device=d.device, dtype=d.dtype)
            mask_mid = ((c > -1 + eps) & (c < 1 - eps)).squeeze(-1)
            if mask_mid.any():
                s = torch.sqrt((1.0 + c[mask_mid]) * 2.0)
                q[mask_mid, 0] = 0.5 * s.squeeze(-1)
                q[mask_mid, 1:4] = v[mask_mid] / s
            mask_aligned = (c >= 1 - eps).squeeze(-1)
            if mask_aligned.any():
                q[mask_aligned, 0] = 1.0
            mask_opposite = (c <= -1 + eps).squeeze(-1)
            if mask_opposite.any():
                q[mask_opposite, 2] = 1.0
            return q

        dev = self.device
        Ndraw = int(min(self.num_envs, getattr(self.cfg, "gimbal_vis_max_envs", 4)))
        if Ndraw <= 0:
            return

        # === 参数 ===
        half_h = 0.5 * math.radians(float(self.cfg.gimbal_fov_h_deg))   # 水平半角
        half_v = 0.5 * math.radians(float(self.cfg.gimbal_fov_v_deg))   # 垂直半角
        R      = float(getattr(self.cfg, "gimbal_effective_range", 40.0))

        all_mid = []
        all_quat = []
        all_scale = []

        for ei in range(Ndraw):
            active = (~self.friend_frozen[ei])
            if not torch.any(active):
                continue

            # 顶点与角度
            P = self.fr_pos[ei][active]          # [S,3]
            Y = self._gimbal_yaw[ei][active]     # [S]
            T = self._gimbal_pitch[ei][active]   # [S]
            S = P.shape[0]
            if S == 0:
                continue

            # 四个角方向 (±half_h, ±half_v) → 远平面四角
            yaws   = torch.stack([Y - half_h, Y - half_h, Y + half_h, Y + half_h], dim=1)  # [S,4]
            pitchs = torch.stack([T - half_v, T + half_v, T - half_v, T + half_v], dim=1)  # [S,4]
            dirs4  = self._dir_from_yaw_pitch(yaws, pitchs)                                 # [S,4,3]
            corners = P[:, None, :] + R * dirs4                                             # [S,4,3]

            # 可选：中心轴线
            if getattr(self.cfg, "gimbal_axis_vis_enable", False):
                dir_c = self._dir_from_yaw_pitch(Y, T)
                tip_c = P + R * dir_c
                mids = P + 0.5 * (tip_c - P)
                dirs = tip_c - P
                quat_wxyz = _quat_wxyz_from_z_to_dir(dirs / dirs.norm(dim=-1, keepdim=True))
                scale = torch.tensor([1.0, 1.0, dirs.norm(dim=-1)], device=dev, dtype=P.dtype).repeat(1, 1)
                all_mid.append(mids)
                all_quat.append(quat_wxyz)
                all_scale.append(scale)

            # 四条侧边（apex→四角）
            for k in range(4):
                A = P
                B = corners[:, k, :]
                dirs = B - A
                lens = torch.linalg.norm(dirs, dim=-1, keepdim=True)
                mids = A + 0.5 * dirs
                dirs_norm = dirs / lens.clamp_min(1e-8)
                quat_wxyz = _quat_wxyz_from_z_to_dir(dirs_norm)
                scale = torch.tensor([1.0, 1.0, 1.0], device=dev, dtype=P.dtype).repeat(S, 1)
                scale[:, 2] = lens.squeeze(-1)
                all_mid.append(mids)
                all_quat.append(quat_wxyz)
                all_scale.append(scale)

            # 远平面四条边（0-1-3-2-0）
            for (a, b) in ((0,1),(1,3),(3,2),(2,0)):
                A = corners[:, a, :]
                B = corners[:, b, :]
                dirs = B - A
                lens = torch.linalg.norm(dirs, dim=-1, keepdim=True)
                mids = A + 0.5 * dirs
                dirs_norm = dirs / lens.clamp_min(1e-8)
                quat_wxyz = _quat_wxyz_from_z_to_dir(dirs_norm)
                scale = torch.tensor([1.0, 1.0, 1.0], device=dev, dtype=P.dtype).repeat(S, 1)
                scale[:, 2] = lens.squeeze(-1)
                all_mid.append(mids)
                all_quat.append(quat_wxyz)
                all_scale.append(scale)

        # === 一次 visualize ===
        if not all_mid:
            return

        pos   = torch.cat(all_mid, dim=0)
        quat  = torch.cat(all_quat, dim=0)
        scale = torch.cat(all_scale, dim=0)

        self._gimbal_fov_ray_marker.visualize(
            translations=pos,
            orientations=quat,
            scales=scale
        )

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

        if getattr(self.cfg, "gimbal_face_centroid", False):
            # 让导引头指向敌方团质心（世界系 z-up）
            d = self._enemy_centroid.unsqueeze(1) - self.fr_pos               # [N,M,3]
            eps = 1e-9
            yaw_des   = torch.atan2(d[..., 1], d[..., 0])                     # z-up yaw
            horiz     = torch.sqrt((d[..., 0]**2 + d[..., 1]**2).clamp_min(eps))
            pitch_des = torch.atan2(d[..., 2], horiz)                          # z-up pitch

            # 目标“相对机体”角（并夹在机械范围内）
            tgt_rel_y = self._wrap_pi(yaw_des - body_yaw).clamp(-rg, rg)
            tgt_rel_p = (pitch_des - body_pitch).clamp(-rg, rg)

            # （可选）清零策略缓冲，避免误解
            self._gimbal_tgt_rel_yaw_cmd.zero_()
            self._gimbal_tgt_rel_pitch_cmd.zero_()
        else:
            # 保留原先“由策略设定的目标相对角”
            tgt_rel_y = self._gimbal_tgt_rel_yaw_cmd.clamp(-rg, rg)
            tgt_rel_p = self._gimbal_tgt_rel_pitch_cmd.clamp(-rg, rg)

        # 误差（yaw 用环角差，pitch 用线性差）
        err_y = self._wrap_pi(tgt_rel_y - rel_y)
        err_p = tgt_rel_p - rel_p

        # 速率限制更新（slew-rate limit）
        step_y = torch.clamp(err_y, -max_step, +max_step)
        step_p = torch.clamp(err_p, -max_step, +max_step)

        new_rel_y = (rel_y + step_y).clamp(-rg, rg)
        new_rel_p = (rel_p + step_p).clamp(-rg, rg)

        # 写回“绝对角”（保持连续，不 wrap，以避免出现跨 ±pi 的跳变）
        # self._gimbal_yaw   = body_yaw + new_rel_y
        # self._gimbal_pitch = body_pitch + new_rel_p
        self._gimbal_yaw   = body_yaw
        self._gimbal_pitch = body_pitch
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

    def _compute_action_mask(self, act: torch.Tensor) -> torch.Tensor:
        """返回与 act 同形状 [N, M, 5] 的乘法掩码(0/1)，只屏蔽明显“越走越糟”的分量。"""
        if not getattr(self.cfg, "action_mask_enable", False) or self.E == 0:
            return torch.ones_like(act)

        dev = self.device
        N, M, _ = act.shape
        mask = torch.ones_like(act, device=dev)

        # ---------- 计算航向误差（机体yaw vs 指向质心） ----------
        body_yaw, _ = self._body_yaw_pitch_from_pose()                      # [N,M]
        d = self._enemy_centroid.unsqueeze(1) - self.fr_pos                 # [N,M,3]
        yaw_des = torch.atan2(d[..., 1], d[..., 0])                         # [N,M]
        yaw_err = self._wrap_pi(yaw_des - body_yaw)                         # [-pi, pi)

        nz_cmd = act[..., 1]                                                # [N,M] 连续动作第2维是 nz

        # 规则 A：若 |yaw误差| 较大，只允许“朝正确方向转”的 nz（错误号的 nz 屏蔽为 0）
        # psi_rate = -g0 * nz / (Vm * cos_th)，因此：想让 yaw 增大(误差>0)，应 nz<0；想让 yaw 减小(误差<0)，应 nz>0
        yaw_thr = math.radians(float(getattr(self.cfg, "mask_yaw_err_deg", 45.0)))
        wrong_left  = (yaw_err > +yaw_thr) & (nz_cmd > 0)   # 该转右，结果在转左 → 屏蔽
        wrong_right = (yaw_err < -yaw_thr) & (nz_cmd < 0)   # 该转左，结果在转右 → 屏蔽
        mask[..., 1] = torch.where(wrong_left | wrong_right, torch.zeros_like(mask[..., 1]), mask[..., 1])

        # 规则 B：若偏得离谱（>hard阈值），将油门直接打 0（先别跑，先把头摆正）
        hard_thr = math.radians(float(getattr(self.cfg, "mask_yaw_err_hard_deg", 110.0)))
        reversed_heading = torch.abs(yaw_err) > hard_thr
        mask[..., 2] = torch.where(reversed_heading, torch.zeros_like(mask[..., 2]), mask[..., 2])

        # ---------- 外包圈判定（抑制“外环绕圈”） ----------
        # 敌群外包圈半径（按 XY 平面）：r_env_max = max_k ||enemy_k - centroid||；外包圈 = ratio * r_env_max
        enemy_active = (~self.enemy_frozen)                                 # [N,E]
        if enemy_active.any():
            centroid_xy = self._enemy_centroid[:, :2]                       # [N,2]
            enemy_xy = self.enemy_pos[:, :, :2]                             # [N,E,2]
            r_enemy = torch.linalg.norm(enemy_xy - centroid_xy.unsqueeze(1), dim=-1)  # [N,E]
            r_enemy = torch.where(enemy_active, r_enemy, torch.zeros_like(r_enemy))
            r_env_max = r_enemy.max(dim=1).values                           # [N]
        else:
            r_env_max = torch.zeros(self.num_envs, device=dev)

        ratio = float(getattr(self.cfg, "mask_outer_band_ratio", 1.2))
        r_band = r_env_max * ratio                                          # [N]

        fr_xy = self.fr_pos[:, :, :2]                                       # [N,M,2]
        r_friend = torch.linalg.norm(fr_xy - centroid_xy.unsqueeze(1), dim=-1)  # [N,M]

        # 径向“外飘”判据：v · r > 0
        v_xy = self.fr_vel_w[:, :, :2]                                      # [N,M,2]
        r_vec = fr_xy - centroid_xy.unsqueeze(1)                             # [N,M,2]
        outward = (v_xy * r_vec).sum(dim=-1) > 0.0                          # [N,M]
        outer   = r_friend > r_band.unsqueeze(1)                             # [N,M]
        outer_and_outward = outer & outward

        if bool(getattr(self.cfg, "mask_outer_stop", True)):
            # 在外且还在往外漂：油门直接打 0（把“外圈大饼”切断）
            mask[..., 2] = torch.where(outer_and_outward, torch.zeros_like(mask[..., 2]), mask[..., 2])

        # # ---------- New: 预测下一步并避免友机进入本机 FOV ----------
        # # 若关闭或 M<=1 则跳过
        # if bool(getattr(self.cfg, "action_mask_enable", False)) and self.M > 1:
        #     # 基于 act 预测下一步位置 (近似使用单步动力学，相同于 _apply_action)
        #     dt = float(getattr(self.cfg, "sim").dt) if hasattr(self.cfg, "sim") else float(self.physics_dt)
        #     # 取 act -> ny, nz, throttle（和 _pre_physics_step 的映射一致）
        #     act_ny = act[..., 0].clamp(-1.0, 1.0) * float(self.cfg.ny_max_g)   # [N,M]
        #     act_nz = act[..., 1].clamp(-1.0, 1.0) * float(self.cfg.nz_max_g)   # [N,M]
        #     act_thr = ((act[..., 2].clamp(-1.0, 1.0) + 1.0) * 0.5)              # [N,M] in [0,1]
        #     Vm_pred = float(self.cfg.Vm_min) + act_thr * (float(self.cfg.Vm_max) - float(self.cfg.Vm_min))  # [N,M]

        #     # 当前姿态/vm 用于计算速率（与 _apply_action 保持一致）
        #     cos_th_now = torch.cos(self.theta).clamp_min(1e-6)
        #     Vm_eff = Vm_pred
        #     Vm_eps = Vm_eff.clamp_min(1e-6)

        #     theta_rate = self.g0 * (act_ny - cos_th_now) / Vm_eps
        #     psi_rate   = - self.g0 * act_nz / (Vm_eps * cos_th_now)
        #     # 限幅与冻结处理（保持与 _apply_action 一致）
        #     THETA_RATE_LIMIT = 1.0
        #     PSI_RATE_LIMIT   = 1.0
        #     theta_rate = torch.clamp(theta_rate, -THETA_RATE_LIMIT, THETA_RATE_LIMIT)
        #     psi_rate   = torch.clamp(psi_rate,   -PSI_RATE_LIMIT,   PSI_RATE_LIMIT)
        #     # 若 agent 已冻结，则不移动
        #     if hasattr(self, "friend_frozen") and self.friend_frozen is not None:
        #         mask_alive = (~self.friend_frozen).float()
        #         theta_rate = theta_rate * mask_alive
        #         psi_rate   = psi_rate   * mask_alive
        #         Vm_eff     = Vm_eff * mask_alive

        #     theta_next = self.theta + theta_rate * dt
        #     psi_next   = (self.psi_v + psi_rate * dt + math.pi) % (2.0 * math.pi) - math.pi

        #     # 速度 -> 世界系（复用 y_up_to_z_up）
        #     sin_th, cos_th = torch.sin(theta_next), torch.cos(theta_next)
        #     sin_ps, cos_ps = torch.sin(psi_next), torch.cos(psi_next)
        #     Vxm = Vm_eff * cos_th * cos_ps
        #     Vym = Vm_eff * sin_th
        #     Vzm = -Vm_eff * cos_th * sin_ps
        #     V_m = torch.stack([Vxm, Vym, Vzm], dim=-1)   # in y-up/body
        #     fr_vel_w_next = y_up_to_z_up(V_m)            # [N,M,3]
        #     fr_pos_next = self.fr_pos + fr_vel_w_next * dt  # [N,M,3]

        #     # 计算下一步是否有 friend 会进入 i 的 FOV（使用当前云台角 gy/gp 作为基准）
        #     # rel: i<-j (注意方向)  -> we want j relative to i: rel_ij = pos_j - pos_i
        #     rel_next = fr_pos_next.unsqueeze(2) - fr_pos_next.unsqueeze(1)   # [N, M, M, 3] i<-j
        #     dx, dy, dz = rel_next[...,0], rel_next[...,1], rel_next[...,2]
        #     eps = 1e-9
        #     az  = torch.atan2(dy, dx)                            # [N,M,M]
        #     horiz = torch.sqrt((dx*dx + dy*dy).clamp_min(eps))
        #     el  = torch.atan2(dz, horiz)
        #     dist = torch.linalg.norm(rel_next, dim=-1)           # [N,M,M]

        #     gy = self._gimbal_yaw.unsqueeze(2).expand_as(az)     # [N,M,M]
        #     gp = self._gimbal_pitch.unsqueeze(2).expand_as(el)
        #     dyaw   = torch.abs(self._wrap_pi(az - gy))
        #     dpitch = torch.abs(el - gp)
        #     half_h = 0.5 * math.radians(float(self.cfg.gimbal_fov_h_deg))
        #     half_v = 0.5 * math.radians(float(self.cfg.gimbal_fov_v_deg))
        #     Rcam   = float(self.cfg.gimbal_effective_range)

        #     in_fov_next = (dyaw <= half_h) & (dpitch <= half_v) & (dist <= Rcam)

        #     # 排除自身（对角位）
        #     eye = torch.eye(self.M, dtype=torch.bool, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
        #     in_fov_next = in_fov_next & (~eye)

        #     # 若任一 j (j != i) 在 i 的下一步 FOV 中，则认为该 i 的动作会造成视野内友机
        #     any_friend_in_fov_next = in_fov_next.any(dim=-1).float()   # [N,M]

        #     # 屏蔽策略：若 any_friend_in_fov_next，为保守起见把油门/throttle 屏蔽（可替换为屏蔽 nz/ny 或屏蔽云台动作）
        #     mask[..., 2] = torch.where(any_friend_in_fov_next > 0.5, torch.zeros_like(mask[..., 2]), mask[..., 2])
        #     # 可选：同时屏蔽云台目标指令（动作维 3/4 是云台——如果你的 policy 输出云台命令需要屏蔽）
        #     mask[..., 3] = torch.where(any_friend_in_fov_next > 0.5, torch.zeros_like(mask[..., 3]), mask[..., 3])
        #     mask[..., 4] = torch.where(any_friend_in_fov_next > 0.5, torch.zeros_like(mask[..., 4]), mask[..., 4])


        return mask

    # —————————————————— ↑↑↑ 工具/可视化区 ↑↑↑ ——————————————————

    # ============================ MARL 交互实现 ============================

    def _pre_physics_step(self, actions: dict[str, torch.Tensor] | torch.Tensor | None) -> None:
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed(); t0 = time.perf_counter()
        if actions is None:
            return

        N, M = self.num_envs, self.M
        ACT_DIM = 3
        act = torch.zeros(N, M, ACT_DIM, device=self.device, dtype=self.fr_pos.dtype)

        if isinstance(actions, dict):
            for i, agent in enumerate(self.possible_agents):
                a = actions.get(agent)
                if a is not None:
                    a = a.to(self.device)
                    cols = min(a.shape[-1], ACT_DIM)    # 允许 3 或 5
                    act[:, i, :cols] = a[..., :cols]
        elif torch.is_tensor(actions):
            a = actions.to(self.device)
            if a.ndim == 3 and a.shape[1] == M:
                cols = min(a.shape[-1], ACT_DIM)        # 允许 [N,M,3] 或 [N,M,5]
                act[..., :cols] = a[..., :cols]
            elif a.ndim == 2 and a.shape[-1] in (3, 5): # 允许 [N,3] 或 [N,5]
                cols = min(a.shape[-1], ACT_DIM)
                act[:, None, :cols] = a[:, None, :cols].repeat(1, M, 1)
            else:
                raise ValueError(f"Unsupported actions shape: {a.shape}")

        # 屏蔽冻结
        if hasattr(self, "friend_frozen") and self.friend_frozen is not None:
            act = act * (~self.friend_frozen).unsqueeze(-1).float()

        # Apply centralized action mask
        act = act * self._compute_action_mask(act)
        # 如果有 NaN，就替换为 0，如果网络全部都输出nan那么将nan转换为0没有任何意义
        # act = torch.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
        # 规范化与映射（ny,nz ∈ [-1,1]； throttle ∈ [-1,1] → [0,1]）
        ny = act[..., 0].clamp(-1.0, 1.0)
        nz = act[..., 1].clamp(-1.0, 1.0)
        throttle = ((act[..., 2].clamp(-1.0, 1.0) + 1.0) * 0.5)

        self._ny = ny * self.cfg.ny_max_g
        self._nz = nz * self.cfg.nz_max_g
        self.Vm  = self.cfg.Vm_min + throttle * (self.cfg.Vm_max - self.cfg.Vm_min)
        
        # # === 云台：第4/5维 → “目标相对角的增量”并积分为 cmd ===
        # rg = math.radians(float(self.cfg.gimbal_range_deg))
        # dt = float(self.physics_dt)
        # tgt_rate = math.radians(float(getattr(self.cfg, "gimbal_tgt_rate_deg",
        #                                     self.cfg.gimbal_rate_deg)))   # 目标角每秒最大变化率（建议 ≤ gimbal_rate_deg）
        # max_tgt_step = tgt_rate * dt
        # deadband = math.radians(float(getattr(self.cfg, "gimbal_deadband_deg", 0.0))) # 目标角增量死区（度），0.3~1.0 可抑制细抖

        # a_y = act[..., 3].clamp(-1.0, 1.0)
        # a_p = act[..., 4].clamp(-1.0, 1.0)

        # # 本步“目标角增量”（可选死区抑制抖动）
        # d_y = a_y * max_tgt_step
        # d_p = a_p * max_tgt_step
        # if deadband > 0.0:
        #     d_y = torch.where(torch.abs(d_y) < deadband, torch.zeros_like(d_y), d_y)
        #     d_p = torch.where(torch.abs(d_p) < deadband, torch.zeros_like(d_p), d_p)

        # # 若有冻结掩码，阻止冻结体更新目标（可选）
        # if hasattr(self, "friend_frozen") and self.friend_frozen is not None:
        #     mask = (~self.friend_frozen).float()   # [N,M]
        #     d_y = d_y * mask
        #     d_p = d_p * mask

        # # 积分到“目标相对角 cmd”，并夹到机械范围内
        # self._gimbal_tgt_rel_yaw_cmd   = (self._gimbal_tgt_rel_yaw_cmd   + d_y).clamp(-rg, rg)
        # self._gimbal_tgt_rel_pitch_cmd = (self._gimbal_tgt_rel_pitch_cmd + d_p).clamp(-rg, rg)

        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            print(f"[TIME] _pre_physics_step: {(time.perf_counter()-t0)*1000:.3f} ms")

    def _apply_action(self):
        """
        与原版一致：步首命中→冻结→推进→写回→缓存更新/可视化
        """
        if getattr(self.cfg, "function_time_print", False):
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
        # print("Vm_eps:",Vm_eps)
        # print(" self.g0:",self.g0)
        # print("(self._ny:",self._ny)
        # print("cos_th_now:",cos_th_now)
        # print("(self._nz:",self._nz)
        # print("Vm_eff:",Vm_eff)
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
        # print("V_m:",V_m)
        # print("Vxm:",Vxm)
        # print("Vym:",Vym)
        # print("Vzm:",Vzm)
        # print("self.theta:",self.theta)
        # print("self.psi_v:",self.psi_v)
        # print("theta_rate:",theta_rate)
        # print("psi_rate:",psi_rate)
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
            v_dir = (-self._axis_hat).unsqueeze(1).expand(-1, self.E, -1)    # [N,E,3]
            enemy_vel_step = v_dir * float(self.cfg.enemy_speed)             # [N,E,3]
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

        self._step_gimbals_to_cover_enemies()
        # 刷新缓存 + 可视化
        self._refresh_enemy_cache()
        self._update_traj_vis()

        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

        self._update_projection_debug_vis()

        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _apply_action (pre-step 1m check): {dt_ms:.3f} ms")

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """
        为每个 agent 返回 reward（字典：agent -> [N]）。
        命中奖励：按“本步新冻结的敌机”计数，均分给 M 个友机（保持与集中式一致的策略）。
        """
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        N, M = self.num_envs, self.M
        dev = self.device
        dtype = self.fr_pos.dtype

        # --- 权重 ---
        centroid_w  = float(getattr(self.cfg, "centroid_approach_weight", 1.0))
        hit_w       = float(getattr(self.cfg, "hit_reward_weight", 1000.0))
        w_fb        = float(getattr(self.cfg, "w_gimbal_friend_block", 0.1))
        w_ec        = float(getattr(self.cfg, "w_gimbal_enemy_cover", 0.1))
        w_int = float(getattr(self.cfg, "intercept_alignment_weight", 1.0))
        w_vc = float(getattr(self.cfg, "vel_to_centroid_weight", 0.0))
        R0    = float(getattr(self.cfg, "vc_zero_inside", 10.0))   # 近距离屏蔽半径（<=R0 基本不给向心奖励）
        alpha = float(getattr(self.cfg, "vc_decay_alpha", 10.0))  # 衰减尺度（越大，远处权重越慢降）
        p     = float(getattr(self.cfg, "vc_decay_power", 1.5))   # 形状（>1 近处更快衰减）
        # —— 目标分散（ID-free）——
        w_assign = float(getattr(self.cfg, "assign_diversity_weight", 0.05))   # 惩罚权重
        beta_soft = float(getattr(self.cfg, "assign_beta", 6.0))               # 软指向的“锐度”
        ang_margin_deg = float(getattr(self.cfg, "assign_angle_margin_deg", 15.0))  # 认为“方向太像”的角阈(度)
        prox_sigma = float(getattr(self.cfg, "assign_prox_sigma", 5.0))        # 空间接近权重的尺度(米)

        # --- 活跃掩码 / 质心 ---
        friend_active    = (~self.friend_frozen)                     # [N,M] bool
        enemy_active_any = self._enemy_active_any                    # [N]   bool
        centroid         = self._enemy_centroid                      # [N,3]

        # --- 质心接近增量（距离减小为正）---
        c = centroid.unsqueeze(1).expand(N, M, 3)                    # [N,M,3]
        diff = c - self.fr_pos                                       # [N,M,3]
        dist_now = torch.linalg.norm(diff, dim=-1)                   # [N,M]

        if (not hasattr(self, "prev_dist_centroid")) or (self.prev_dist_centroid is None) \
        or (self.prev_dist_centroid.shape != dist_now.shape):
            self.prev_dist_centroid = dist_now.detach().clone()

        # 当敌机全灭时，保持距离不变，避免最后一步大幅负增量
        dist_now_safe = torch.where(enemy_active_any.unsqueeze(1), dist_now, self.prev_dist_centroid)
        d_delta_signed = self.prev_dist_centroid - dist_now_safe     # [N,M]
        centroid_each = d_delta_signed * friend_active.float()       # 只计未冻结友机 [N,M]

        # --- 命中奖励（本步新冻结敌机数量 * hit_w，均分给 M 个友机）---
        new_hits_mask   = self._newly_frozen_enemy                   # [N,E] bool
        hit_bonus_env   = new_hits_mask.float().sum(dim=1) * hit_w   # [N]
        per_agent_hit   = (hit_bonus_env / max(M, 1)).unsqueeze(1).expand(-1, M)  # [N,M]

        # --- 云台相关（可见性）---
        if getattr(self.cfg, "gimbal_face_centroid", False):
            # 关闭云台奖励/惩罚时，设为 0
            # pen_friend_each = torch.zeros((N, M), device=dev, dtype=dtype)  # [N,M]
            enemy_count_each = torch.zeros((N, M), device=dev, dtype=dtype) # [N,M]
            # 用于 episode 统计（env-level）
            # pair_friend_pen_env = torch.zeros(N, device=dev, dtype=dtype)   # [N]
            cover_count_env     = torch.zeros(N, device=dev, dtype=dtype)   # [N]

            # 友->友：第 j 个云台能否看到第 k 个友机
            vis_ff = self._gimbal_friend_visible_mask().float()             # [N,M,M]
            # 每个 agent 看到了多少友机（有向计数 i->k）
            pen_friend_each = vis_ff.sum(dim=2)                              # [N,M]

            # env-level：用上三角去重（i<j 计一次）
            pair_ff = torch.maximum(vis_ff, vis_ff.transpose(1, 2))          # [N,M,M]
            tri = torch.triu(torch.ones(self.M, self.M, dtype=torch.bool, device=dev), diagonal=1).unsqueeze(0)
            pair_friend_pen_env = (pair_ff * tri).sum(dim=(1, 2)).to(dtype)  # [N] 未加权
        else:
            # 友->友：第 j 个云台能否看到第 k 个友机
            vis_ff = self._gimbal_friend_visible_mask().float()             # [N,M,M]
            # 每个 agent 看到了多少友机（有向计数 i->k）
            pen_friend_each = vis_ff.sum(dim=2)                              # [N,M]

            # env-level：用上三角去重（i<j 计一次）
            pair_ff = torch.maximum(vis_ff, vis_ff.transpose(1, 2))          # [N,M,M]
            tri = torch.triu(torch.ones(self.M, self.M, dtype=torch.bool, device=dev), diagonal=1).unsqueeze(0)
            pair_friend_pen_env = (pair_ff * tri).sum(dim=(1, 2)).to(dtype)  # [N] 未加权

            # 友->敌：第 j 个云台能否看到第 k 个敌机
            vis_fe = self._gimbal_enemy_visible_mask()                       # [N,M,E] bool
            enemy_count_each = vis_fe.sum(dim=-1).float()                    # [N,M]
            # 冻结友机不计
            enemy_count_each = enemy_count_each * friend_active.float()
            # env-level 聚合（求和）
            cover_count_env = enemy_count_each.sum(dim=1)                    # [N]

        # === New: 拦截对齐奖励（仅用 e_hat，不暴露敌机位置给策略） ===
        # 逻辑：把友机速度在“任一被云台看到的敌机方向单位向量 e_hat_k”上的投影取最大，
        # 仅正向（靠近）计奖；不可见或冻结敌机的 e_hat 置 0。
        intercept_reward_each = torch.zeros((N, M), device=self.device, dtype=self.fr_pos.dtype)

        if self.E > 0:
            # [N,M,E]：第 j 个云台能否看到第 k 个敌机
            vis_fe = self._gimbal_enemy_visible_mask()  # bool
            if hasattr(self, "enemy_frozen") and self.enemy_frozen is not None:
                vis_fe = vis_fe & (~self.enemy_frozen).unsqueeze(1)

            # e_hat_all: [N,M,E,3]（与 obs 构造一致，但这里只在 env 内部计算用于 reward）
            rel_all  = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)               # [N,M,E,3]
            dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(1e-9)     # [N,M,E,1]
            e_hat_all = rel_all / dist_all                                                  # [N,M,E,3]
            e_hat_all = e_hat_all * vis_fe.unsqueeze(-1).float()                          # 不可见置 0
            # if hasattr(self, "enemy_frozen") and self.enemy_frozen is not None:
            #     mask_alive = (~self.enemy_frozen).unsqueeze(1).unsqueeze(-1)   # [N,1,E,1], bool
            #     e_hat_all = e_hat_all * mask_alive.to(e_hat_all.dtype)

            # 速度在各 e_hat 上的标量投影：proj = v · e_hat
            v = self.fr_vel_w.unsqueeze(2)                                                  # [N,M,1,3]
            proj = (v * e_hat_all).sum(dim=-1)                                              # [N,M,E]

            # 每个友机选择“当前最对齐/最靠近”的那个敌机方向，友机速度在“与敌机连线方向”上的最大正投影量（代表该友机目前最有拦截意图的敌机）
            proj_max, _ = proj.max(dim=-1)                                                  # [N,M]
            intercept_each = torch.clamp(proj_max, min=0.0)                                 # 仅正向计奖
            intercept_reward_each = w_int * intercept_each                                  # [N,M]

            # === ID-free：软目标方向 + 同向靠近惩罚（抑制扎堆） ===
            # 1) 基于“对齐的正投影”做带遮罩的 softmax 得到权重（不依赖编号）
            proj_pos = torch.clamp(proj, min=0.0)                                          # [N,M,E]
            logits = beta_soft * proj_pos + (~vis_fe) * (-1e6)                             # mask 无可见时 ≈ -inf
            w_soft = torch.softmax(logits, dim=-1)                                         # [N,M,E]
            has_any = vis_fe.any(dim=-1, keepdim=True)                                     # [N,M,1]
            w_soft = w_soft * has_any.float()                                              # 全不可见时 -> 全零

            # 2) 软目标方向（连续向量，不是编号）
            u_soft = (w_soft.unsqueeze(-1) * e_hat_all).sum(dim=-2)                        # [N,M,3]
            u_norm = u_soft / u_soft.norm(dim=-1, keepdim=True).clamp_min(1e-6)           # 归一化；全零保持零

            # 3) 仅对“彼此很接近且指向很相似”的友机对施加惩罚（余弦铰链）
            cos = (u_norm.unsqueeze(2) * u_norm.unsqueeze(1)).sum(dim=-1)                  # [N,M,M]
            cos0 = math.cos(math.radians(ang_margin_deg))
            pen_pair = torch.clamp(cos - cos0, min=0.0)                                    # 角差小 -> 惩罚大

            # 空间接近加权（离得越近越算扎堆）
            ppos = self.fr_pos                                                             # [N,M,3]
            dij  = torch.linalg.norm(ppos.unsqueeze(2) - ppos.unsqueeze(1), dim=-1)        # [N,M,M]
            w_prox = torch.exp(- (dij * dij) / (2.0 * max(prox_sigma, 1e-6) ** 2))         # [N,M,M]

            # 去掉自身、自/他冻结
            eye = torch.eye(M, dtype=torch.bool, device=dev).unsqueeze(0)                  # [1,M,M]
            pairmask = (~eye) & friend_active.unsqueeze(2) & friend_active.unsqueeze(1)    # [N,M,M]

            # 每个友机的“扎堆惩罚”（对其它友机求和）
            assign_pen_each = (pen_pair * w_prox * pairmask.float()).sum(dim=2)            # [N,M]


        # --- 速度朝向质心的对齐（正向投影，单位 m/s）---
        # 单位轴：从友机指向“当前存活敌机质心”的单位向量 ê_c
        e_hat_c = torch.where(
            dist_now_safe.unsqueeze(-1) > 1e-6,
            diff / dist_now_safe.unsqueeze(-1),
            torch.zeros_like(diff)
        )  # [N,M,3]

        # 速度在该轴上的标量投影（只取正向，表示“朝质心的速度分量”）
        v = self.fr_vel_w                                                # [N,M,3]
        # v_proj_c = (v * e_hat_c).sum(dim=-1)                             # [N,M] (m/s)
        # vel_to_centroid_each_raw = torch.clamp(v_proj_c, min=0.0)        # 仅正向
        # vel_to_centroid_each     = vel_to_centroid_each_raw * friend_active.float()
        v_proj_c = (v * e_hat_c).sum(dim=-1)                             # [N,M] (m/s)

        # 距离衰减 gate：远处≈1，近处→0（强→弱）
        d_eff = (dist_now_safe - R0).clamp(min=0.0)                      # [N,M]
        gate  = ((d_eff / (d_eff + alpha)).clamp(0.0, 1.0)) ** p         # [N,M]

        # “raw”记为未乘权重但已衰减后的量，便于日志直观看贡献
        vel_to_centroid_each_raw = gate * torch.clamp(v_proj_c, min=0.0) # [N,M]
        vel_to_centroid_each     = vel_to_centroid_each_raw * friend_active.float()


        # --- 合成 per-agent reward ---
        # 基础：质心接近 + 均分命中
        r_each = centroid_w * centroid_each + per_agent_hit                  # [N,M]
        # 云台项（按 agent 各自的观测计）
        r_each = r_each - w_fb * pen_friend_each + w_ec * enemy_count_each   # [N,M]
        # 拦截对齐
        r_each = r_each + intercept_reward_each
        # 速度朝质心对齐
        r_each = r_each + w_vc * vel_to_centroid_each
        # 目标分散（ID-free）
        r_each = r_each - w_assign * assign_pen_each
        # --- 写出字典 ---
        rewards = {agent: r_each[:, i] for i, agent in enumerate(self.possible_agents)}

        # --- 统计（episode_sums，env-level）---
        self.episode_sums.setdefault("centroid_approach",     torch.zeros(N, device=dev, dtype=dtype))
        self.episode_sums.setdefault("hit_bonus",             torch.zeros(N, device=dev, dtype=dtype))
        self.episode_sums.setdefault("gimbal_friend_block",   torch.zeros(N, device=dev, dtype=dtype))
        self.episode_sums.setdefault("gimbal_enemy_cover",    torch.zeros(N, device=dev, dtype=dtype))
        self.episode_sums.setdefault("intercept_alignment",   torch.zeros(N, device=dev, dtype=dtype))
        self.episode_sums.setdefault("vel_to_centroid",       torch.zeros(N, device=dev, dtype=dtype))
        self.episode_sums.setdefault("assign_diversity_pen", torch.zeros(N, device=dev, dtype=dtype))
        self.episode_sums["assign_diversity_pen"] += assign_pen_each.sum(dim=1)

        # 将 per-agent 的质心靠近求和后记一条 env-level 统计（与集中式语义一致）
        self.episode_sums["centroid_approach"] += centroid_each.sum(dim=1)   # [N]
        self.episode_sums["hit_bonus"]         += hit_bonus_env              # [N]
        # 这两项统计按 env-level 加权累计（仅用于日志）
        self.episode_sums["gimbal_friend_block"] -= w_fb * pair_friend_pen_env
        self.episode_sums["gimbal_enemy_cover"]  += w_ec * cover_count_env
        self.episode_sums["intercept_alignment"] += intercept_each.sum(dim=1)
        self.episode_sums["vel_to_centroid"]     += vel_to_centroid_each_raw.sum(dim=1)

        # --- 状态缓存/一次性标志 ---
        self.prev_dist_centroid = dist_now_safe
        self._newly_frozen_enemy[:]  = False
        self._newly_frozen_friend[:] = False

        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _get_rewards: {dt_ms:.3f} ms")

        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        为每个 agent 返回统一的 done / time_out 掩码（与 swarm_vel_env 的风格一致）。
        额外：在 self.extras['termination'] 中记录本步各终止原因的计数（仅统计本步结束的 env）。
        """
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        tol = float(getattr(self.cfg, "overshoot_tol", 3.0))
        r2_goal = float(self.cfg.enemy_goal_radius) ** 2
        xy_max2 = 150.0 ** 2

        if self._goal_e is None:
            self._rebuild_goal_e()

        N         = self.num_envs
        device    = self.device

        # ---------- 基本终止判据（逐 env） ----------
        success_all_enemies = self.enemy_frozen.all(dim=1)                        # [N] 敌全灭（友军成功）
        z = self.fr_pos[..., 2]
        out_z_any = ((z < 0.0) | (z > 10.5)).any(dim=1)                            # [N] Z 越界
        origin_xy = self.terrain.env_origins[:, :2].unsqueeze(1)
        dxy = self.fr_pos[..., :2] - origin_xy
        out_xy_any = (dxy.square().sum(dim=-1) > xy_max2).any(dim=1)              # [N] XY 越界
        nan_inf_any = ~torch.isfinite(self.fr_pos).all(dim=(1, 2))                # [N] NaN/Inf

        # 场上冻结数目不一致
        fr_frozen_sum = self.friend_frozen.sum(dim=1)    # [N]
        enemy_frozen_sum = self.enemy_frozen.sum(dim=1)
        freeze_mismatch_any = (fr_frozen_sum != enemy_frozen_sum)  # [N]

        enemy_goal_any = torch.zeros(N, dtype=torch.bool, device=device)          # [N]
        overshoot_any  = torch.zeros(N, dtype=torch.bool, device=device)          # [N]

        # 仅对尚未触发终止/越界/NaN 的 env 继续判“敌达终点”和“超越”
        alive_mask = ~(success_all_enemies | out_z_any | out_xy_any | nan_inf_any)
        if alive_mask.any():
            idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)

            # 敌机到达目标（任一敌进入目标圆）
            diff_e = self.enemy_pos[idx] - self._goal_e[idx].unsqueeze(1)
            enemy_goal_any[idx] = (diff_e.square().sum(dim=-1) < r2_goal).any(dim=1)

            # “超越”判定（沿 axis_hat 投影友/敌，若 min(friend) > max(enemy)+tol 则认为友已整体越过敌）
            friend_active = (~self.friend_frozen[idx])                            # [n,M]
            enemy_active  = self._enemy_active[idx]                               # [n,E]
            have_both = friend_active.any(dim=1) & enemy_active.any(dim=1)
            if have_both.any():
                k_idx    = have_both.nonzero(as_tuple=False).squeeze(-1)
                gk       = self._goal_e[idx][k_idx]                               # [k,3]
                axis_hat = self._axis_hat[idx][k_idx]                             # [k,3]
                sf = ((self.fr_pos[idx][k_idx]    - gk.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [k,M]
                se = ((self.enemy_pos[idx][k_idx] - gk.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [k,E]

                INF     = torch.tensor(float("inf"),  dtype=sf.dtype, device=sf.device)
                NEG_INF = torch.tensor(float("-inf"), dtype=sf.dtype, device=sf.device)
                sf_masked_for_min = torch.where(friend_active[k_idx], sf, INF)
                se_masked_for_max = torch.where(enemy_active[k_idx],  se, NEG_INF)

                friend_min = sf_masked_for_min.min(dim=1).values                  # [k]
                enemy_max  = se_masked_for_max.max(dim=1).values                  # [k]
                separated  = friend_min > (enemy_max + tol)
                overshoot_any[idx[k_idx]] = separated

        died     = out_z_any | out_xy_any | nan_inf_any | success_all_enemies | enemy_goal_any | overshoot_any | freeze_mismatch_any  # [N]
        time_out = self.episode_length_buf >= self.max_episode_length - 1                                         # [N]

        dones  = {agent: died     for agent in self.possible_agents}
        truncs = {agent: time_out for agent in self.possible_agents}

        # ---------- 终止原因统计（仅统计“本步结束”的 env；按优先级做互斥分类） ----------
        ended = died | time_out                             # 本步确实结束的 env
        if ended.any():
            remaining = ended.clone()

            # 先把“纯超时”分出来（非 died 且 time_out）
            timeout_mask = remaining & (~died) & time_out
            cnt_timeout = int(timeout_mask.sum().item())
            remaining = remaining & (~timeout_mask)

            # 再对 died 的 env 做互斥分类（优先级避免重复计数）
            def take(mask: torch.Tensor) -> int:
                nonlocal remaining
                m = (remaining & died & mask)
                c = int(m.sum().item())
                remaining = remaining & (~m)
                return c

            cnt_nan_inf        = take(nan_inf_any)
            cnt_freeze_mismatch= take(freeze_mismatch_any)
            cnt_oob_xy         = take(out_xy_any)
            cnt_oob_z          = take(out_z_any)
            cnt_overshoot      = take(overshoot_any)
            cnt_enemygoal      = take(enemy_goal_any)
            cnt_success        = take(success_all_enemies)

            # 剩余兜底（理论应为 0）
            cnt_other = int(remaining.sum().item())

            # 写入 extras，供 reset() 打印
            if not hasattr(self, "extras") or self.extras is None:
                self.extras = {}
            self.extras["termination"] = {
                "done_total"         : int(ended.sum().item()),
                "timeout"            : cnt_timeout,
                "nan_or_inf"         : cnt_nan_inf,
                "freeze_mismatch"    : cnt_freeze_mismatch,
                "out_of_bounds_xy"   : cnt_oob_xy,
                "out_of_bounds_z"    : cnt_oob_z,
                "overshoot"          : cnt_overshoot,
                "enemy_goal"         : cnt_enemygoal,
                "success_all_enemies": cnt_success,
                "other"              : cnt_other
            }

        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _get_dones: {dt_ms:.3f} ms")

        return dones, truncs

    def _reset_idx(self, env_ids: torch.Tensor | None):
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
            if hasattr(self, "extras") and isinstance(self.extras, dict) and "termination" in self.extras:
                term = self.extras["termination"]
                print("\n--- Episode Termination Summary ---")
                for k, v in term.items():
                    print(f"{k:<20}: {v}")
                print("-----------------------------------")

            # === 在 episode 结束打印上一次累计的分项奖励 ===
            if hasattr(self, "episode_sums") and isinstance(self.episode_sums, dict) and len(self.episode_sums) > 0:
                N = self.num_envs
                dev = self.device
                dtype = self.fr_pos.dtype

                # 取权重
                centroid_w = float(getattr(self.cfg, "centroid_approach_weight", 1.0))
                w_int      = float(getattr(self.cfg, "intercept_alignment_weight", 10.0))
                w_vc       = float(getattr(self.cfg, "vel_to_centroid_weight", 1.0))
                w_assign   = float(getattr(self.cfg, "assign_diversity_weight", 0.0))

                # 取各项（可能不存在时用 0 向量）
                zero = lambda: torch.zeros(N, device=dev, dtype=dtype)

                centroid_raw = self.episode_sums.get("centroid_approach", zero())       # 未加权 [N]
                hit_bonus    = self.episode_sums.get("hit_bonus", zero())               # 已加权 [N]
                gfb          = self.episode_sums.get("gimbal_friend_block", zero())     # 已加权(通常为负) [N]
                gec          = self.episode_sums.get("gimbal_enemy_cover",  zero())     # 已加权(为正)   [N]

                # ★ 拦截对齐（两种口径兼容）
                intercept_raw    = self.episode_sums.get("intercept_alignment", zero())        # 未加权 [N]
                has_int_reward   = ("intercept_alignment_reward" in self.episode_sums)
                intercept_reward = self.episode_sums.get("intercept_alignment_reward", zero()) # 已加权 [N]

                # ★ 速度→质心（已做距离衰减的 raw）
                vel_to_centroid_raw = self.episode_sums.get("vel_to_centroid", zero())         # 未加权 [N]

                # ★ 新增：ID-free 方向多样性惩罚（抑制扎堆）
                assign_pen_raw = self.episode_sums.get("assign_diversity_pen", zero())         # 未加权 [N]

                # 与当前 reward 对应的“加权后分项”
                centroid_contrib  = centroid_w * centroid_raw
                intercept_contrib = intercept_reward if has_int_reward else (w_int * intercept_raw)
                vel_to_centroid_contrib = w_vc * vel_to_centroid_raw
                assign_diversity_contrib= - w_assign * assign_pen_raw

                # 合计（把所有项都加上）
                total_env = (
                    centroid_contrib + intercept_contrib + vel_to_centroid_contrib + assign_diversity_contrib +
                    hit_bonus + gfb + gec
                )

                print("\n--- Episode Reward Summary (env-level, weighted) ---")
                print(f"{'centroid_approach(w)':<25}: {centroid_contrib.mean().item():8.3f}")
                print(f"{'intercept_alignment(w)':<25}: {intercept_contrib.mean().item():8.3f}")
                print(f"{'vel_to_centroid(w)':<25}: {vel_to_centroid_contrib.mean().item():8.3f}")
                print(f"{'assign_diversity_pen(w)':<25}: {assign_diversity_contrib.mean().item():8.3f}")
                print(f"{'hit_bonus':<25}: {hit_bonus.mean().item():8.3f}")
                print(f"{'gimbal_friend_block':<25}: {gfb.mean().item():8.3f}")
                print(f"{'gimbal_enemy_cover':<25}: {gec.mean().item():8.3f}")
                print(f"{'ENV_MEAN_TOTAL':<25}: {total_env.mean().item():8.3f}")
                print("-----------------------------------\n")

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

        # # 友方并排沿 Y 生成
        # spacing = float(getattr(self.cfg, "formation_spacing", 0.8))
        # idx = torch.arange(M, device=dev).float() - (M - 1) / 2.0
        # offsets_xy = torch.stack([torch.zeros_like(idx), idx * spacing], dim=-1)
        # offsets_xy = offsets_xy.unsqueeze(0).expand(N, M, 2)
        # fr0 = torch.empty(N, M, 3, device=dev)
        # fr0[..., :2] = origins[:, :2].unsqueeze(1) + offsets_xy
        # fr0[...,  2] = origins[:, 2].unsqueeze(1) + float(self.cfg.flight_altitude)
        # self.fr_pos[env_ids]  = fr0
        # self.fr_vel_w[env_ids] = 0.0

        # 敌机生成
        self._spawn_enemy(env_ids)
        # self._spawn_enemy_random(env_ids)

        # 敌机初速度（环向）
        phi = torch.rand(N, device=dev) * 2.0 * math.pi
        spd = float(self.cfg.enemy_speed)
        self.enemy_vel[env_ids, :, 0] = spd * torch.cos(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 1] = spd * torch.sin(phi).unsqueeze(1)
        self.enemy_vel[env_ids, :, 2] = 0.0

        # === 刷新敌团缓存（保证 _axis_hat / _enemy_centroid 与本轮出生一致）===
        self._refresh_enemy_cache()

        # ---------------友方出生朝向---------------
        # 友方出生即面朝团中心一字排开（沿“朝向的左法向量”展开）
        eps = 1e-6
        spacing = float(getattr(self.cfg, "formation_spacing", 0.8))
        backoff = float(getattr(self.cfg, "formation_backoff", 0.0))

        # 仅用 XY 平面决定横队与朝向：axis_hat 指向“原点->质心”，所以面向质心用 -axis_hat
        axis_hat_xy = self._axis_hat[env_ids, :2]                     # [N,2]
        face_xy     = -axis_hat_xy
        face_norm   = torch.linalg.norm(face_xy, dim=-1, keepdim=True).clamp_min(eps)
        f_hat       = face_xy / face_norm                             # [N,2] 面向质心单位向量

        # 横向排队方向 = 朝向的左法向量 r_hat = rot90ccw(f_hat) = [-fy, fx]
        r_hat = torch.stack([-f_hat[..., 1], f_hat[..., 0]], dim=-1)  # [N,2]

        # 以各 env 的 origin 为队列中心，可沿“远离质心”的方向回撤 backoff
        row_center = origins[:, :2]
        if backoff > 0.0:
            row_center = row_center - backoff * f_hat

        # 对称编号 …, -2, -1, 0, 1, 2, …
        idx = torch.arange(self.M, device=self.device).float() - (self.M - 1) / 2.0  # [M]
        offsets_xy = idx.view(1, self.M, 1) * spacing * r_hat.unsqueeze(1)          # [N,M,2]

        # 友机初始位置
        fr0 = torch.empty(N, self.M, 3, device=dev, dtype=self.fr_pos.dtype)
        fr0[..., :2] = row_center.unsqueeze(1) + offsets_xy
        fr0[...,  2] = origins[:, 2].unsqueeze(1) + float(self.cfg.flight_altitude)
        self.fr_pos[env_ids]   = fr0
        self.fr_vel_w[env_ids] = 0.0
        self.Vm[env_ids]       = 0.0

        # 每架友机的初始航向/俯仰：直接指向“敌团质心”
        d = self._enemy_centroid[env_ids].unsqueeze(1) - self.fr_pos[env_ids]  # [N,M,3] (z-up)
        # yaw（z-up）
        psi0 = torch.atan2(d[..., 1], d[..., 0])
        psi0 = ((psi0 + math.pi) % (2.0 * math.pi)) - math.pi
        self.psi_v[env_ids] = psi0
        # pitch（y-up）：sin(theta) = z_w
        d_m = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        sin_th = d_m[..., 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta0 = torch.asin(sin_th)
        self.theta[env_ids] = theta0

        # 初始化纵向过载等
        self._ny[env_ids] = 0.0
        self._nz[env_ids] = 0.0
        # ---------------友方出生朝向---------------

        # 云台重置（与机体保持一致指向质心；若你有机械限位需求，可在此处 clamp）
        self._gimbal_yaw[env_ids]   = psi0
        self._gimbal_pitch[env_ids] = theta0
        if hasattr(self, "_gimbal_tgt_rel_yaw_cmd"):
            self._gimbal_tgt_rel_yaw_cmd[env_ids]   = 0.0
        if hasattr(self, "_gimbal_tgt_rel_pitch_cmd"):
            self._gimbal_tgt_rel_pitch_cmd[env_ids] = 0.0

        # ==== TRAJ VIS ====
        self._traj_reset(env_ids)

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

        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

        if getattr(self.cfg, "function_time_print", False):
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
        per-agent 观测（每个 agent 一行）：
        [ self_abs_pos(3),
        sorted_other_abs_pos_flat(3*(M-1)),
        self_abs_vel(3),
        sorted_other_abs_vel_flat(3*(M-1)),
        e_hat_to_visible_enemies_flat(3E),
        centroid_u_hat(3) ]

        - 所有友机位置和速度均为世界系绝对量；
        - 除自己外的友机按与自己距离升序排列。
        """
        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            t0 = time.perf_counter()

        N, M, E = self.num_envs, self.M, self.E
        dev, dtype = self.device, self.fr_pos.dtype
        eps = 1e-9

        # ====================== 友机绝对位置/速度排序 ======================
        # fr_pos, fr_vel_w: [N, M, 3]
        pos_i = self.fr_pos.unsqueeze(2).expand(N, M, M, 3)
        pos_j = self.fr_pos.unsqueeze(1).expand(N, M, M, 3)
        dist_ij = torch.linalg.norm(pos_j - pos_i, dim=-1)   # [N,M,M] 友机i到友机j的欧氏距离
        dist_ij += torch.eye(M, device=dev, dtype=dtype).unsqueeze(0) * 1e6  # 自身距离设极大

        # 获取每个友机视角下的“其他友机”排序索引
        sorted_idx = dist_ij.argsort(dim=-1)[:, :, :M-1]  # [N,M,M-1]，sorted_idx[n, i, k] = 第 i 架友机的第 k 近的友机的编号 j

        # torch.gather(input, dim, index, out=None)，从 input 中按 index 提取对应位置的值
        other_pos_sorted = torch.gather(
            self.fr_pos.unsqueeze(1).expand(N, M, M, 3),
            2, sorted_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # [N,M,M-1,3]

        other_vel_sorted = torch.gather(
            self.fr_vel_w.unsqueeze(1).expand(N, M, M, 3),
            2, sorted_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # [N,M,M-1,3]

        # 拼接：自己的绝对位置/速度 + 排序后的其他友机绝对位置/速度
        self_pos = self.fr_pos.unsqueeze(2)  # [N,M,1,3]
        self_vel = self.fr_vel_w.unsqueeze(2)  # [N,M,1,3]
        all_pos_sorted = torch.cat([self_pos, other_pos_sorted], dim=2).reshape(N, M, 3 * M)  # [N,M,3M]
        all_vel_sorted = torch.cat([self_vel, other_vel_sorted], dim=2).reshape(N, M, 3 * M)  # [N,M,3M]

        # ====================== 敌机方向（单位向量） ======================
        if E > 0:
            vis_fe = self._gimbal_enemy_visible_mask()  # [N,M,E]
            if hasattr(self, "enemy_frozen") and self.enemy_frozen is not None:
                vis_fe = vis_fe & (~self.enemy_frozen).unsqueeze(1)

            rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)  # [N,M,E,3]
            dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)
            e_hat_all = (rel_all / dist_all) * vis_fe.unsqueeze(-1).float()
            e_hat_flat = e_hat_all.reshape(N, M, 3 * E)
        else:
            e_hat_flat = torch.zeros((N, M, 0), device=dev, dtype=dtype)

        # # ====================== 敌群质心方向 ======================
        # if E > 0:
        #     centroid_env = self._enemy_centroid              # [N,3]
        #     any_active   = self._enemy_active_any            # [N] bool

        #     rel_c = centroid_env.unsqueeze(1) - self.fr_pos  # [N,M,3]
        #     d_c   = torch.linalg.norm(rel_c, dim=-1, keepdim=True).clamp_min(eps)
        #     u_c   = rel_c / d_c                              # [N,M,3]

        #     # 如果这一局已经没有活敌了，就把 u_c 清零
        #     u_c = u_c * any_active.view(N, 1, 1).float()
        # else:
        #     u_c = torch.zeros((N, M, 3), device=dev, dtype=dtype)

        # ====================== 拼接总观测 ======================
        obs_each = torch.cat([all_pos_sorted, all_vel_sorted, e_hat_flat], dim=-1)  # [N,M,6M+3E+3]
        obs_dict = {ag: obs_each[:, i, :] for i, ag in enumerate(self.possible_agents)}

        if getattr(self.cfg, "function_time_print", False):
            self._cuda_sync_if_needed()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] _get_observations: {dt_ms:.3f} ms")

        return obs_dict

    def _get_states(self) -> torch.Tensor:
        """
        提供集中式状态（供 MAPPO 等使用）：简单拼接所有友机的 [pos, vel] 与敌机方向。
        也可以直接复用集中式观测：这里用每个友机的 obs 串接。
        """
        obs = self._get_observations()
        states = torch.cat([obs[ag] for ag in self.possible_agents], dim=-1)  # [N, M*single_obs_dim]

        return states

# ---------------- Gym 注册 ----------------
from config import agents

gym.register(
    id="FAST-Intercept-Swarm-Distributed",
    entry_point=FastInterceptionSwarmMARLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FastInterceptionSwarmMARLCfg,
        # "skrl_mappo_cfg_entry_point": f"{agents.__name__}:L_M_interception_swarm_skrl_mappo_cfg.yaml",
        # "skrl_ppo_cfg_entry_point": f"{agents.__name__}:L_M_interception_swarm_skrl_mappo_cfg.yaml",
        "skrl_ippo_cfg_entry_point":  f"{agents.__name__}:L_M_interception_swarm_ippo.yaml",
    },
)
