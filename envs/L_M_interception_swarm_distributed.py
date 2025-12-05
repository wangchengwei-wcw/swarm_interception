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

@configclass
class FastInterceptionSwarmMARLCfg(DirectMARLEnvCfg):
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # ---------- 数量控制 ----------
    swarm_size: int = 6                 # 便捷参数：同时设置友机/敌机数量
    # friendly_size: int = math.floor(30 * 1.25)
    friendly_size: int = 30
    enemy_size: int = 30

    # 敌机出生区域（圆盘）与最小间隔
    debug_vis_enemy = True
    enemy_height_min = 10.0
    enemy_height_max = 10.0
    enemy_speed = 5.0
    enemy_target_alt = 10.0
    enemy_goal_radius = 1.0
    enemy_cluster_ring_radius: float = 100.0  # 敌机的生成距离
    enemy_cluster_radius: float = 20.0        # 敌机团的半径(固定队形中未使用)
    enemy_min_separation: float = 5.0         # 敌机间最小水平间隔
    enemy_vertical_separation: float = 5.0    # 立体队形敌机间最小垂直间隔
    enemy_center_jitter: float = 0.0          # 敌机团中心位置随机抖动幅度
    hit_radius = 1.0

    # 友方控制/速度范围/位置间隔
    Vm_min = 11.0
    Vm_max = 13.0
    ny_max_g = 3.0
    nz_max_g = 3.0
    formation_spacing = 2.0
    flight_altitude = 5.0

    # 奖励相关权重配置
    centroid_approach_weight: float = 0.02
    vel_to_centroid_weight: float = 0.0
    hit_reward_weight: float = 10.0
    all_kill_weight: float = 10.0
    w_target_align: float = 0.0008
    leak_penalty_weight: float = 0.01
    leak_margin: float = 1.0
    friend_too_low_penalty_weight: float = 0.001
    friend_too_high_penalty_weight: float = 0.001
    enemy_reach_goal_penalty_weight: float = 10.0
    w_gimbal_friend_block: float = 0.3
    w_gimbal_enemy_cover: float = 0.0
    vc_zero_inside: float = 15.0


    # 友机队形参数
    agents_per_row: int     = 10       # 每排数量 (建议 10)
    lat_spacing: float      = 5.0      # 横向间隔 (同一排飞机间距)
    row_spacing: float      = 5.0      # 纵向间隔 (排与排之间距，需>4m以避开水平FOV)
    row_height_diff: float  = 3.0      # 高度阶梯 (后排比前排高)

    # -------- 分波发射配置（按友机编号切波）--------
    friend_wave_enable: bool =  False               # 分拨发射开关
    friend_wave_launch_dist: float = 110.0          # 前一波离原点超过多少米发射下一波

    # —— 云台 / FOV & 生效距离 ——
    gimbal_fov_h_deg: float = 10.0      # 水平总 FOV（度）
    gimbal_fov_v_deg: float = 12.0      # 垂直总 FOV（度）
    gimbal_range_deg: float = 30.0      # 相对机体限位 ±30°
    gimbal_rate_deg:  float = 20.0      # 角速度 20°/s
    gimbal_effective_range: float = 100.0  # 云台“有效拍摄距离”（米）

    # 频率
    episode_length_s = 50.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

    # for debug
    gimbal_vis_enable: bool = False          # 云台视野可视化开关
    traj_vis_enable: bool = False            # 轨迹可视化开关
    per_train_data_print: bool = False       # reset中打印
    gimbal_axis_vis_enable: bool = False     # 可视化云台光轴

    # —— 云台可视化（小方块点阵线框） ——
    gimbal_vis_max_envs: int = 1            # 只画前K个env，控性能

    # ==== TRAJ VIS ==== 友方轨迹可视化
    traj_vis_max_envs: int = 1              # 只画前几个 env
    traj_vis_len: int = 500                 # 每个友机最多保留多少个轨迹点（循环缓冲）
    traj_vis_every_n_steps: int = 2         # 每隔多少个物理步记录/刷新一次
    traj_marker_size: tuple[float,float,float] = (0.05, 0.05, 0.05)  # 面包屑小方块尺寸

    # —— 单 agent 观测/动作维（用于 MARL 的 per-agent 空间）——
    single_observation_space: int = 9     # 将在 __post_init__ 基于 E 自动覆盖为 6 + 3E
    single_action_space: int = 3          # (ny, nz, throttle)

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
        # single_obs_dim = 7 * int(M) + 3 * int(E) + 11
        single_obs_dim = 7 * int(M) + 15
        single_act_dim = 3                  # (ny, nz, throttle)

        self.single_observation_space = single_obs_dim
        self.single_action_space = single_act_dim
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
        # single_obs_dim = 7 * int(M) + 3 * int(E) + 11
        single_obs_dim = 7 * int(M) + 15
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

        self._enemy_exists_mask = torch.ones(N, self.E, device=dev, dtype=torch.bool)         # 哪些敌机槽位“真正存在”（用于变编队数量）
        self._enemy_active_count = torch.full((N,), self.E, device=dev, dtype=torch.long)     # 每个 env 当前真实敌机数量（用于匹配友机出动数量）

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

        # ------------------ 可视化与调试 ------------------
        self.friendly_visualizer = None
        self.enemy_visualizer    = None
        self.centroid_marker     = None
        self.ray_marker          = None
        self._fov_marker         = None
        self._traj_markers = []  # per-friend trajectory markers
        self.set_debug_vis(self.cfg.debug_vis)

        # ------------------- 奖励 ----------------------
        self.prev_dist_centroid = torch.zeros(N, M, device=self.device, dtype=torch.float32)

        # ------------------ 敌团缓存（每步更新） ------------------
        self._enemy_centroid_init = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._enemy_centroid      = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._enemy_active        = torch.zeros(N, self.E, device=dev, dtype=torch.bool)
        self._enemy_active_any    = torch.zeros(N, device=dev, dtype=torch.bool)
        self._goal_e              = None
        self._axis_hat            = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._axis_hat_xy         = torch.zeros(N, 2, device=dev, dtype=dtype)
        self.enemy_goal_height    = torch.zeros(N, 1, device=dev, dtype=dtype)

        # ------------------ 分波发射：按编号预分配波次 ------------------
        self.friend_wave_enable = bool(getattr(self.cfg, "friend_wave_enable", False))
        self.friend_wave_size   = int(getattr(self.cfg, "agents_per_row", self.M))

        # 波次数 = ceil(M / wave_size)
        self.friend_wave_count = int(math.ceil(self.M / self.friend_wave_size))

        # 每个友机固定所属波次（0,1,2,...）
        self.friend_wave_index = (torch.arange(self.M, device=dev, dtype=torch.long) // self.friend_wave_size)  # [M]

        # 每个env当前发射到第几波（0-based）
        self.friend_wave_stage = torch.zeros(self.num_envs, device=dev, dtype=torch.long)

        # 分波发射用的缓存：原点 xy（懒初始化）
        self._origins_xy = None

        # ---- agent id one-hot feature ----
        self._agent_id_onehot = torch.eye(self.M, device=dev, dtype=torch.float32).unsqueeze(0)

        # 记录每个环境当前的有效作战单位数量（初始化为满员）
        self._current_active_count = torch.full((self.num_envs,), self.M, device=dev, dtype=torch.long)

    # —————————————————— ↓↓↓↓↓工具/可视化区↓↓↓↓↓ ——————————————————
    def _friendly_world_quats_zup(self) -> torch.Tensor:
        """由zup系下欧拉角的pitch是抬头正,低头负。而转换到四元数中pitch是抬头负、低头正。从print来看直接从速度反算yaw、pitch和直接拿yaw、pitch数值好像是一样的""" 
        # vel = self.fr_vel_w  # [N, M, 3]  Z-up 世界速度
        # vx, vy, vz = vel[..., 0], vel[..., 1], vel[..., 2]  # z-up
        # sp_xy = torch.sqrt((vx * vx + vy * vy).clamp_min(1e-9))

        # # ==== Yaw（偏航角，绕 Z 轴）====
        # yaw   = torch.atan2(vy, vx)
        # # ==== Pitch（俯仰角，绕 Y 轴）====
        # pitch = -torch.atan2(vz, sp_xy)
        # # ==== Roll = 0（我们不控制横滚）====
        # roll = torch.zeros_like(yaw)
        # print("yaw1:",yaw, " pitch1:", pitch)
        # print("yaw2:",self.psi_v, " pitch2:", -self.theta)

        yaw   = self.psi_v
        pitch = -self.theta
        roll = torch.zeros_like(yaw)

        # ==== 欧拉角 → 四元数（Z-Y-X 顺序，即 yaw → pitch → roll）====
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        quats = torch.stack([w, x, y, z], dim=-1)  # [N, M, 4]
        quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True).clamp_min(1e-8)

        return quats

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
        # 只统计“存在且未冻结”的敌机
        exists = self._enemy_exists_mask                     # [N,E]
        enemy_active = exists & (~self.enemy_frozen)         # [N,E]
        e_mask = enemy_active.unsqueeze(-1).float()          # [N,E,1]

        # 质心：如果某个 env 没有任何 active 敌机，就用 clamp_min 防止除零
        sum_pos = (self.enemy_pos * e_mask).sum(dim=1)       # [N,3]
        cnt     = e_mask.sum(dim=1).clamp_min(1.0)           # [N,1]
        centroid = sum_pos / cnt                             # [N,3]

        self._enemy_centroid   = centroid
        self._enemy_active     = enemy_active
        self._enemy_active_any = enemy_active.any(dim=1)

        axis = centroid - self._goal_e
        norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._axis_hat = axis / norm                # 敌方目标点指向敌团质心的单位向量

        axis_xy = centroid[:, :2] - self._goal_e[:, :2]
        norm_xy = axis_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._axis_hat_xy = axis_xy / norm_xy

    def _spawn_enemy(self, env_ids: torch.Tensor):
        # ---- 基本量 ----
        dev   = self.fr_pos.device
        dtype = self.fr_pos.dtype
        env_ids = env_ids.to(dtype=torch.long, device=dev)
        N = env_ids.shape[0]
        E_max = int(self.E)  # 最大敌机数
        E_min = 12

        origins_all = self.terrain.env_origins
        if origins_all.device != dev:
            origins_all = origins_all.to(dev)
        origins = origins_all[env_ids]  # [N,3]

        if self._goal_e is None:
            self._rebuild_goal_e()
        goal_e = self._goal_e[env_ids]  # [N,3]

        s_min = float(self.cfg.enemy_min_separation)
        sz_v  = float(getattr(self.cfg, "enemy_vertical_separation", s_min))
        hmin  = float(self.cfg.enemy_height_min)
        hmax  = float(self.cfg.enemy_height_max)
        R_center = float(getattr(self.cfg, "enemy_cluster_ring_radius", 8.0))
        center_jitter = float(getattr(self.cfg, "enemy_center_jitter", 0.0))

        # ==================================================================
        #  内部工具函数定义 (保持原逻辑，确保自包含)
        # ==================================================================
        def _centerize(xyz: torch.Tensor) -> torch.Tensor:
            return xyz - xyz.mean(dim=-2, keepdim=True)

        def _rect2d_dims(E: int, aspect_w: float = 2.0) -> tuple[int, int]:
            cols = max(1, int(math.ceil(math.sqrt(E * max(1e-3, aspect_w)))))
            rows = int(math.ceil(E / cols))
            return rows, cols

        def _grid2d(rows: int, cols: int, s: float) -> torch.Tensor:
            xs = torch.arange(cols, dtype=dtype, device=dev)
            ys = torch.arange(rows, dtype=dtype, device=dev)
            X, Y = torch.meshgrid(xs, ys, indexing="xy")
            X = X.t().reshape(-1)
            Y = Y.t().reshape(-1)
            xyz = torch.stack([X * s, Y * s, torch.zeros_like(X)], dim=-1)
            return _centerize(xyz)

        def _grid3d(rows: int, cols: int, layers: int, sx: float, sy: float, sz_: float) -> torch.Tensor:
            xs = torch.arange(cols,   dtype=dtype, device=dev)
            ys = torch.arange(rows,   dtype=dtype, device=dev)
            zs = torch.arange(layers, dtype=dtype, device=dev)
            X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="xy")
            X = X.permute(1, 0, 2).reshape(-1)
            Y = Y.permute(1, 0, 2).reshape(-1)
            Z = Z.permute(1, 0, 2).reshape(-1)
            xyz = torch.stack([X * sx, Y * sy, Z * sz_], dim=-1)
            return _centerize(xyz)

        def _best_rc(cap_layer: int, aspect_xy: float = 2.0) -> tuple[int, int]:
            aspect_xy = max(1e-6, float(aspect_xy))
            best = None
            best_rc = (1, cap_layer)
            for r in range(1, cap_layer + 1):
                c = math.ceil(cap_layer / r)
                area_over = r * c - cap_layer
                aspect_err = abs((c / r) - aspect_xy)
                score = (area_over, aspect_err)
                if best is None or score < best:
                    best = score
                    best_rc = (r, c)
            return best_rc

        # ---- 模板生成函数 ----
        def _tmpl_v_wedge_2d(E: int, s: float) -> torch.Tensor:
            if E <= 0: return torch.zeros(0, 3, dtype=dtype, device=dev)
            step = s / math.sqrt(2.0)
            if E == 1: return torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=dev)
            K = (E - 1) // 2
            ks = torch.arange(1, K + 1, dtype=dtype, device=dev)
            up   = torch.stack([ks * step,  ks * step, torch.zeros_like(ks)], dim=-1)
            down = torch.stack([ks * step, -ks * step, torch.zeros_like(ks)], dim=-1)
            pts = torch.cat([torch.zeros(1, 3, dtype=dtype, device=dev), up, down], dim=0)
            if (E - 1) % 2 == 1:
                extra_k = torch.tensor([(K + 1) * step], dtype=dtype, device=dev)
                extra   = torch.stack([extra_k, extra_k, torch.zeros_like(extra_k)], dim=-1)
                pts = torch.cat([pts, extra], dim=0)
            return _centerize(pts[:E, :])

        def _tmpl_rect_2d(E: int, s: float, aspect: float = 2.0) -> torch.Tensor:
            r, c = _rect2d_dims(E, aspect)
            xyz = _grid2d(r, c, s)[:E, :]
            return xyz

        def _tmpl_square_2d(E: int, s: float) -> torch.Tensor:
            return _tmpl_rect_2d(E, s, aspect=1.0)

        def _tmpl_rect_3d(E: int, s: float, sz_: float, aspect_xy: float = 2.0) -> torch.Tensor:
            L = 2
            cap_layer = max(1, math.ceil(E / L))
            r, c = _best_rc(cap_layer, aspect_xy)
            xyz = _grid3d(r, c, L, s, s, sz_)[:E, :]
            return xyz

        def _tmpl_cube_3d(E: int, s: float, sz_: float) -> torch.Tensor:
            if E <= 0: return torch.zeros(0, 3, dtype=dtype, device=dev)
            n = max(1, int(round(E ** (1.0 / 3.0))))
            while (n + 1) ** 3 <= E: n += 1
            while n ** 3 > E: n -= 1
            base_count = n ** 3
            xs = torch.arange(n, dtype=dtype, device=dev)
            ys = torch.arange(n, dtype=dtype, device=dev)
            zs = torch.arange(n, dtype=dtype, device=dev)
            X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
            Xf, Yf, Zf = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
            base_xyz = torch.stack([Xf * s, Yf * s, Zf * sz_], dim=-1)
            pts = base_xyz
            rem = E - base_count
            if rem > 0:
                z_mid = n // 2
                idx = torch.arange(rem, dtype=torch.long, device=dev)
                col_idx = idx // n
                row_idx = idx % n
                x_extra = (n + col_idx).to(dtype) * s
                y_extra = row_idx.to(dtype) * s
                z_extra = torch.full((rem,), float(z_mid), dtype=dtype, device=dev) * sz_
                extra_xyz = torch.stack([x_extra, y_extra, z_extra], dim=-1)
                pts = torch.cat([pts, extra_xyz], dim=0)
            return _centerize(pts[:E])

        def _tmpl_rect_3d_reverse(E: int, s: float, sz_: float, aspect_xy: float = 2.0) -> torch.Tensor:
            L = 2
            cap_layer = max(1, math.ceil(E / L))
            r, c = _best_rc(cap_layer, aspect_xy)
            xyz = _grid3d(c, r, L, s, s, sz_)[:E, :]
            return xyz

        # 1. 为每个环境随机分配一个模板 ID (0~5)
        # 0:V, 1:Rect, 2:Square, 3:Rect3D, 4:Cube, 5:Rect3DRev
        template_ids = torch.randint(low=0, high=6, size=(N,), device=dev)

        # 准备容器
        local_pos_buffer = torch.zeros(N, E_max, 3, device=dev, dtype=dtype)
        active_counts_buffer = torch.zeros(N, device=dev, dtype=torch.long)

        # 辅助：根据几何约束获取合法数量
        def get_valid_counts(tmpl_id, min_n, max_n):
            valid = []
            
            # V字: 奇数
            if tmpl_id == 0: 
                for x in range(min_n, max_n + 1):
                    if x % 2 == 1: valid.append(x)
            
            # Rect 2D: 必须填满完美矩形 (rows * cols == x)
            elif tmpl_id == 1:
                for x in range(min_n, max_n + 1):
                    cols = max(1, int(math.ceil(math.sqrt(x * 2.0)))) # aspect=2.0
                    rows = int(math.ceil(x / cols))
                    if rows * cols == x:
                        valid.append(x)

            # Square 2D: 完全平方数
            elif tmpl_id == 2: 
                for x in range(1, 10): 
                    sq = x * x
                    if sq >= min_n and sq <= max_n:
                        valid.append(sq)

            # Rect 3D / Reverse: 偶数且单层为完美矩形
            elif tmpl_id in [3, 5]:
                for x in range(min_n, max_n + 1):
                    if x % 2 != 0: continue
                    cap = x // 2
                    r, c = _best_rc(cap, aspect_xy=2.0)
                    if r * c == cap:
                        valid.append(x)

            # Cube 3D: 完全立方数
            elif tmpl_id == 4: 
                for x in range(1, 6):
                    cb = x ** 3
                    if cb >= min_n and cb <= max_n:
                        valid.append(cb)
            
            else: 
                valid = list(range(min_n, max_n + 1))
            
            if not valid: valid = [max_n]
            return valid

        # 2. 按模板类型分组处理
        unique_t_ids = torch.unique(template_ids)

        for t_id in unique_t_ids:
            t_id_val = int(t_id.item())

            # 找到当前模板的所有环境索引
            env_mask = (template_ids == t_id)
            indices = torch.nonzero(env_mask, as_tuple=False).squeeze(-1) # [n_group]
            n_group = indices.shape[0]

            # a. 获取该模板的合法数量
            valid_list = get_valid_counts(t_id_val, E_min, E_max)
            valid_tensor = torch.tensor(valid_list, device=dev, dtype=torch.long)

            # b. 随机选择数量
            rand_idx = torch.randint(0, len(valid_list), (n_group,), device=dev)
            chosen_counts = valid_tensor[rand_idx] # [n_group]

            active_counts_buffer[indices] = chosen_counts

            # c. 按数量细分组生成点云
            unique_counts = torch.unique(chosen_counts)
            for c in unique_counts:
                count_val = int(c.item())
                sub_mask = (chosen_counts == c) 
                final_indices = indices[sub_mask] # 全局 env indices

                # 调用对应的模板生成函数
                pts = None
                if t_id_val == 0:
                    pts = _tmpl_v_wedge_2d(count_val, sz_v)
                elif t_id_val == 1:
                    pts = _tmpl_rect_2d(count_val, s_min, aspect=2.0)
                elif t_id_val == 2:
                    pts = _tmpl_square_2d(count_val, s_min)
                elif t_id_val == 3:
                    pts = _tmpl_rect_3d(count_val, s_min, sz_v, aspect_xy=2.0)
                elif t_id_val == 4:
                    pts = _tmpl_cube_3d(count_val, s_min, sz_v)
                elif t_id_val == 5:
                    pts = _tmpl_rect_3d_reverse(count_val, s_min, sz_v, aspect_xy=2.0)
                else:
                    pts = _tmpl_rect_2d(count_val, s_min)

                # 翻转 X (朝向 -X)
                pts = pts.clone()
                pts[..., 0] *= -1.0

                # 填入 buffer
                local_pos_buffer[final_indices, :count_val, :] = pts.unsqueeze(0)

        # ==================================================================
        #  后续处理：旋转、放置、冻结
        # ==================================================================

        # 保存有效数量供友机出生逻辑使用
        self._current_active_count[env_ids] = active_counts_buffer

        # ---- 计算环上中心 ----
        theta = 2.0 * math.pi * torch.rand(N, device=dev, dtype=dtype)
        centers = torch.stack([
            origins[:, 0] + R_center * torch.cos(theta),
            origins[:, 1] + R_center * torch.sin(theta)
        ], dim=1)
        if center_jitter > 0.0:
            centers = centers + (torch.rand(N, 2, device=dev, dtype=dtype) - 0.5) * (2.0 * center_jitter)

        # ---- 旋转到 goal 方向 ----
        head_vec = (goal_e[:, :2] - centers)
        head = head_vec / head_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        c, s = head[:, 0], head[:, 1]
        
        Rm = torch.stack([
            torch.stack([c, -s], dim=-1),
            torch.stack([s,  c], dim=-1)
        ], dim=1) # [N, 2, 2]

        # 旋转 XY
        local_xy = local_pos_buffer[:, :, :2] # [N, E_max, 2]
        xy_rot = torch.matmul(local_xy, Rm.transpose(1, 2)) 
        xy = centers.unsqueeze(1) + xy_rot

        # 处理高度 Z
        local_z = local_pos_buffer[:, :, 2:3]
        z_bottom = hmin + torch.rand(N, 1, 1, device=dev, dtype=dtype) * max(1e-6, (hmax - hmin))
        z_abs = origins[:, 2:3].unsqueeze(1) + z_bottom + local_z

        enemy_pos = torch.cat([xy, z_abs], dim=-1) # [N, E_max, 3]

        # 写入位置
        self.enemy_pos[env_ids] = enemy_pos

        # ---- 应用冻结 (Freeze) ----
        idx_e = torch.arange(E_max, device=dev).unsqueeze(0) # [1, 30]
        cnts  = active_counts_buffer.unsqueeze(1)            # [N, 1]

        # 真实存在的掩码
        exists_mask = idx_e < cnts # [N, E_max]

        self._enemy_exists_mask[env_ids]  = exists_mask
        self._enemy_active_count[env_ids] = active_counts_buffer

        # 不存在的敌机设为冻结
        self.enemy_frozen[env_ids] = ~exists_mask

        # 捕获点同步
        self.enemy_capture_pos[env_ids] = enemy_pos

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self.friendly_visualizer is None:
                from isaaclab.markers import VisualizationMarkers, Loitering_Munition_MARKER_CFG
                import copy
                import colorsys
                f_cfg = Loitering_Munition_MARKER_CFG.copy()
                f_cfg.prim_path = "/Visuals/FriendlyModel"

                # 取出原始的 USD 配置作为模板
                base_marker = f_cfg.markers["mymodel"]

                # 生成 self.M 种明显不同的颜色（HSV 均匀取样）
                def _color_wheel(n):
                    cols = []
                    for i in range(max(1, n)):
                        h = i / float(max(1, n))
                        s = 0.8
                        v = 0.9
                        r, g, b = colorsys.hsv_to_rgb(h, s, v)
                        cols.append((r, g, b))
                    return cols

                colors = _color_wheel(self.M)

                markers = {}
                for i in range(self.M):
                    cfg_i = copy.deepcopy(base_marker)
                    # 缩放还是你原来那样
                    if hasattr(cfg_i, "scale"):
                        cfg_i.scale = (30.0, 30.0, 30.0)
                    # 每个 prototype 一个不同的材质颜色
                    cfg_i.visual_material = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=colors[i]
                    )
                    markers[f"mymodel_{i}"] = cfg_i

                # 用我们新建的一批 prototype 替换原来的 markers
                f_cfg.markers = markers

                self.friendly_visualizer = VisualizationMarkers(f_cfg)
                self.friendly_visualizer.set_visibility(True)
                # 坐标轴可视化友机
                # if HAS_AXIS_MARKER and AXIS_MARKER_CFG is not None:
                #     f_cfg = AXIS_MARKER_CFG.copy()
                #     f_cfg.prim_path = "/Visuals/FriendlyAxis"
                #     f_cfg.markers["frame"].scale = (1, 1, 1)
                #     self.friendly_visualizer = VisualizationMarkers(f_cfg)
                # self.friendly_visualizer.set_visibility(True)
            if self.cfg.debug_vis_enemy and self.enemy_visualizer is None:
                if getattr(self.cfg, "enemy_render_as_sphere", True) and HAS_SPHERE_MARKER:
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
            if self.friendly_visualizer is not None:
                self.friendly_visualizer.set_visibility(False)
            if self.enemy_visualizer is not None:
                self.enemy_visualizer.set_visibility(False)
            if self.centroid_marker is not None:
                self.centroid_marker.set_visibility(False)
            if self.ray_marker is not None:
                self.ray_marker.set_visibility(False)

    def _flatten_agents(self, X: torch.Tensor) -> torch.Tensor:
        return X.reshape(-1, X.shape[-1])

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """四元数乘法: 结果表示先做 q1 再做 q2（w,x,y,z）"""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=-1)

    def _debug_vis_callback(self, event):
        # ------------------ 友机可视化 (修正版：移到地下隐藏) ------------------
        if self.friendly_visualizer is not None:
            # 1. 准备原始数据
            fr_quats = self._friendly_world_quats_zup()          # [N,M,4]
            pos      = self.fr_pos.clone()                       # [N,M,3] 克隆一份以免修改物理状态
            quat     = fr_quats                                  # [N,M,4]
            scale    = torch.ones_like(pos)                      # [N,M,3] 默认缩放 1.0

            # 2. 生成“本局有效”掩码
            #    idx_m: [1, M]
            idx_m = torch.arange(self.M, device=self.device).unsqueeze(0)
            #    counts: [N, 1]
            counts = self._current_active_count.unsqueeze(1)
            #    is_inactive: [N, M] (注意这里是取反，找出无效的)
            is_inactive = idx_m >= counts

            # 3. 将无效友机移到地下极远处，并缩放为 0
            #    这样 USD 仍然认为它们存在，就不会报 Empty InstanceIndices，也不会报 Index Out of Bounds
            if is_inactive.any():
                # 移到 Z = -1000.0
                pos[is_inactive] = torch.tensor([0.0, 0.0, -1000.0], device=self.device)
                # 缩放设为 0 (双重保险，确保看不见)
                scale[is_inactive] = 0.0

            # 4. 修正旋转 (四元数乘法，保持原有逻辑)
            ang = math.pi / 2
            q_fix = torch.tensor(
                [math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)],
                device=self.device,
                dtype=quat.dtype,
            ).expand_as(quat)
            quat = self._quat_mul(quat, q_fix)
            quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)

            # 5. 生成 marker indices
            #    这里必须生成完整的 [0, 1, ..., M-1] * N，即使有些是隐藏的
            #    这样 Visualizer 的索引就永远是固定的，不会越界
            marker_indices = torch.arange(self.M, device=self.device, dtype=torch.long).repeat(self.num_envs)
            
            # 6. 展平数据
            pos_flat   = self._flatten_agents(pos)
            quat_flat  = self._flatten_agents(quat)
            scale_flat = self._flatten_agents(scale)

            # 7. 提交渲染
            self.friendly_visualizer.visualize(
                translations=pos_flat,
                orientations=quat_flat,
                scales=scale_flat,
                marker_indices=marker_indices,
            )

        # ------------------ 敌机可视化 (保持不变) ------------------
        if self.enemy_visualizer is not None:
            pos_flat    = self.enemy_pos.reshape(-1, 3)
            exists_flat = self._enemy_exists_mask.reshape(-1)
            if exists_flat.any():
                self.enemy_visualizer.visualize(translations=pos_flat[exists_flat])
            else:
                self.enemy_visualizer.visualize(translations=torch.empty(0, 3, device=self.device))

    def _setup_scene(self):
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _dir_from_yaw_pitch(self, yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        # z-up: (cos p cos y, cos p sin y, sin p)。返回在z-up世界坐标系下，这个yaw/pitch对应的单位方向向量，目前看没有问题
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
            # v = torch.cross(z, d)
            v = torch.linalg.cross(z, d, dim=-1)
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
            if getattr(self.cfg, "gimbal_axis_vis_enable", True):
                dir_c = self._dir_from_yaw_pitch(Y, T)         # [S,3]
                tip_c = P + R * dir_c                          # [S,3]
                mids = P + 0.5 * (tip_c - P)                   # [S,3]
                dirs = tip_c - P                               # [S,3]

                # 轴线方向的单位向量
                lens = dirs.norm(dim=-1, keepdim=True)         # [S,1]
                dirs_norm = dirs / lens.clamp_min(1e-8)        # [S,3]
                quat_wxyz = _quat_wxyz_from_z_to_dir(dirs_norm)  # [S,4]

                # 每条射线的 scale = [1, 1, length]
                scale = torch.ones((S, 3), device=dev, dtype=P.dtype)  # [S,3]
                scale[:, 2] = lens.squeeze(-1)                         # 把 z 方向改成长度

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

    def _gimbal_control(self):
        # 从速度中反推姿态角
        # vel = self.fr_vel_w  # [N, M, 3]  Z-up 世界速度
        # vx, vy, vz = vel[..., 0], vel[..., 1], vel[..., 2]  # z-up
        # sp_xy = torch.sqrt((vx * vx + vy * vy).clamp_min(1e-9))

        # # ==== Yaw（偏航角，绕 Z 轴）====
        # yaw   = torch.atan2(vy, vx)
        # # ==== Pitch（俯仰角，绕 Y 轴）====
        # pitch = torch.atan2(vz, sp_xy)

        # self._gimbal_yaw = yaw  # 存储以供云台使用
        # self._gimbal_pitch = pitch

        # # 直接拿取姿态角
        self._gimbal_yaw = self._wrap_pi(self.psi_v)
        self._gimbal_pitch = self.theta
        # print("yaw:",yaw)
        # print("pitch:",pitch)
        # print("psi_v:",self._wrap_pi(self.psi_v))
        # print("theta:",self.theta)

        # 可视化
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
        az  = torch.atan2(dy, dx)                                # 方位角（azimuth）敌机相对于云台的水平偏角
        horiz = torch.sqrt((dx*dx + dy*dy).clamp_min(eps))       # 水平距离
        el  = torch.atan2(dz, horiz)                             # 俯仰角（elevation）
        dist = torch.linalg.norm(rel, dim=-1)                    # 欧氏距离

        gy = self._gimbal_yaw.unsqueeze(-1); gp = self._gimbal_pitch.unsqueeze(-1)
        dyaw   = torch.abs(self._wrap_pi(az - gy))
        dpitch = torch.abs(el - gp)
        in_fov = (dyaw <= half_h) & (dpitch <= half_v)
        in_rng = (dist <= Rcam)

        alive_e = (~self.enemy_frozen).unsqueeze(1)                 # [N,1,E]

        # m = in_fov & in_rng & alive_e  # [N_env, N_fr, N_en]
        # print("env 0, friend 0 mask:\n", m[0, 0])         # 打印第0个env、第0个友机看到的敌机
        # print("env 0 all friends:\n", m[0])               # 打印第0个环境所有友机的可见情况
        # print("env 0, friend 0, enemies idx:\n", m[0,0].nonzero())

        # m = in_fov & in_rng & alive_e  # [N_env, N_fr, N_en]
        # env_id = 0
        # print("======= env", env_id, "gimbal visible enemies per friend =======")
        # for fr_id in range(M):
        #     vis_idx = m[env_id, fr_id].nonzero(as_tuple=True)[0]  # [K] 敌机下标
        #     print(f"env {env_id}, friend {fr_id}, enemies idx:", vis_idx.tolist())
        # print("=======================================================")

        exists_alive_e = (self._enemy_exists_mask & (~self.enemy_frozen)).unsqueeze(1)  # [N,1,E]
        return in_fov & in_rng & exists_alive_e

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

        pos_i = self.fr_pos.unsqueeze(2)      # [N, M, 1, 3]
        pos_j = self.fr_pos.unsqueeze(1)      # [N, 1, M, 3]
        rel = pos_j - pos_i                   # [N, M, M, 3]，i -> j

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

        eye = torch.eye(M, dtype=torch.bool, device=self.device).unsqueeze(0).expand(N,-1,-1) # [N,M,M] 排除自身
        alive = (~self.friend_frozen)

        # m = in_fov & in_rng & alive  # [N_env, N_fr, N_en]
        # print("1env 0, friend 0 mask:\n", m[0, 0])         # 打印第0个env、第0个友机看到的敌机
        # print("1env 0 all friends:\n", m[0])               # 打印第0个环境所有友机的可见情况
        # print("1env 0, friend 0, friends idx:\n", m[0,0].nonzero())

        # m = in_fov & in_rng & alive  # [N_env, N_fr, N_en]
        # env_id = 0
        # print("======= FRIEND gimbal visible (env", env_id, ") =======")
        # for fr_i in range(M):
        #     vis_idx = m[env_id, fr_i].nonzero(as_tuple=True)[0]  # [K]，所有 j 的索引
        #     print(f"env {env_id}, friend {fr_i} sees friends:", vis_idx.tolist())
        # print("=======================================================")

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

    def _update_wave_launch(self):
        if (not getattr(self, "friend_wave_enable", False)) or (self.friend_wave_count <= 1):
            return

        # 1. 计算每架友机到各自 env 原点的水平距离 dist_to_origin: [N, M]
        origins_xy = self.terrain.env_origins[:, :2]                 # [N, 2]
        diff_xy = self.fr_pos[..., :2] - origins_xy.unsqueeze(1)     # [N, M, 2]
        dist_to_origin = torch.linalg.norm(diff_xy, dim=-1)          # [N, M]

        # 2. 波次信息
        # wave_idx[m]       : 第 m 架友机所属的波次 (0,1,2,...)
        # stage[n]          : 第 n 个 env 当前已经发射到第几波
        wave_idx = self.friend_wave_index                            # [M]
        stage    = self.friend_wave_stage                            # [N]

        # 3. 当前波次掩码：
        # mask_cur[n, m] = True 表示：第 n 个 env 中的第 m 架友机属于“该 env 当前波次”
        # wave_idx.unsqueeze(0) -> [1, M]
        # stage.unsqueeze(1)    -> [N, 1]
        mask_cur = (wave_idx.unsqueeze(0) == stage.unsqueeze(1))     # [N, M]

        # 4. 只在当前波次上统计距离最大值
        #    非当前波次的友机用 0 距离占位（不会影响 max）
        current_wave_dist = torch.where(mask_cur, dist_to_origin,
                                        torch.zeros_like(dist_to_origin))    # [N, M]
        max_dist, _ = current_wave_dist.max(dim=1)                           # [N]

        # 5. 判定哪些 env 需要触发下一波：
        #    - 还有下一波可发 (stage < friend_wave_count - 1)
        #    - 当前波次的最远距离 >= 发射阈值
        has_next_wave = stage < (self.friend_wave_count - 1)         # [N]
        trigger = has_next_wave & (max_dist >= self.cfg.friend_wave_launch_dist)          # [N]

        if not trigger.any():
            return

        # 6. 对需要触发的 env，波次计数 +1
        stage_new = torch.where(trigger, stage + 1, stage)
        self.friend_wave_stage = stage_new

        # 7. 新波次掩码：
        #    mask_next[n, m] = True 表示：
        #       - 第 n 个 env 触发了下一波 (trigger[n] 为 True)
        #       - 第 m 架友机属于该 env 的“新波次 stage_new[n]”
        mask_next = (
            (wave_idx.unsqueeze(0) == stage_new.unsqueeze(1)) &      # 属于该 env 的新波次
            trigger.unsqueeze(1)                                     # 且该 env 被触发
        )                                                            # [N, M]

        # 8. 解冻这些友机（把它们从 friend_frozen 中移除）
        # 获取当前的永久禁用掩码 (idx >= active_count)
        active_counts = self._current_active_count  # [N]
        fr_idx = torch.arange(self.M, device=self.device).unsqueeze(0) # [1, M]
        permanent_disable_mask = (fr_idx >= active_counts.unsqueeze(1)) # [N, M]
        
        # 逻辑：NewFrozen = (OldFrozen & (~mask_next)) | permanent_disable_mask
        # 即：尝试解冻 mask_next 选中的，但如果它在永久禁用名单里，强制保持 True
        self.friend_frozen = (self.friend_frozen & (~mask_next)) | permanent_disable_mask

    # —————————————————— ↑↑↑ 工具/可视化区 ↑↑↑ ——————————————————

    # ============================ MARL交互实现 ============================
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for i, ag in enumerate(self.possible_agents):
            a = actions[ag].to(self.device, self.fr_pos.dtype)          # [N,3]

            # 只用前三维：ny, nz ∈ [-1,1]；throttle ∈ [-1,1]→[0,1]
            ny = a[:, 0].clamp(-1.0, 1.0)
            nz = a[:, 1].clamp(-1.0, 1.0)
            throttle = (a[:, 2].clamp(-1.0, 1.0) + 1.0) * 0.5

            # 写回各 agent 对应列
            self._ny[:, i] = ny * self.cfg.ny_max_g
            self._nz[:, i] = nz * self.cfg.nz_max_g
            self.Vm[:, i]  = self.cfg.Vm_min + throttle * (self.cfg.Vm_max - self.cfg.Vm_min)

    def _apply_action(self):
        dt = float(self.physics_dt)
        r = float(self.cfg.hit_radius)
        is_first_substep = ((self._sim_step_counter - 1) % self.cfg.decimation) == 0
        if is_first_substep:
            self._newly_frozen_friend[:] = False
            self._newly_frozen_enemy[:]  = False

        # 得到步首位置/冻结状态
        fr_pos0 = self.fr_pos.clone()
        en_pos0 = self.enemy_pos.clone()
        fz0 = self.friend_frozen.clone()
        ez0 = self.enemy_frozen.clone()

        # ---------- 判断是否在步首命中 ----------
        active_pair0 = (~fz0).unsqueeze(2) & (~ez0).unsqueeze(1)  # [N,M,E]
        if active_pair0.any():
            diff0 = fr_pos0.unsqueeze(2) - en_pos0.unsqueeze(1)   # [N,M,E,3]
            dist0 = torch.linalg.norm(diff0, dim=-1)              # [N,M,E]
            hit_pair0 = (dist0 <= r) & active_pair0               # [N,M,E]
            # 敌机是否被命中（沿友机维度），不能用hit_pair0.any(dim=2)因为这样只知道友机打中了，但是会出现多友机命中同一敌机的情况，需要选最近的友机作为击中者
            newly_hitted_enemy = hit_pair0.any(dim=1)             # [N,E] 返回的是第N个环境的第E个敌机是否被命中。沿着友机维度M看，有没有命中该敌机，从上往下看，再从左往右。.any(dim=1)传入的 dim 是被消掉（被扫描）的那一轴；剩下的轴就是你“固定住”的索引。
            if newly_hitted_enemy.any():
                # print("hit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # —— 为每个“本步新冻敌机”选最近友机作为击中者 ——
                INF = torch.tensor(float("inf"), device=self.device, dtype=dist0.dtype)
                dist_masked0 = torch.where(hit_pair0, dist0, INF)     # [N,M,E] 存储的是命中对的距离，未命中对为 +inf
                hitter_idx   = dist_masked0.argmin(dim=1)             # [N,E]存的是第N个环境中敌机E被哪个友机打中的友机索引。对每个(n,e)，在友机维M上找最小距离对应的友机索引j*，即击中者，二维矩阵就是竖着那一列找最小的索引。如果有一列全是+inf（即该敌机未被任何友机命中），argmin会返回0
                # newly_hitted_enemy形状是[N, E]（环境×敌机的布尔掩码）。newly_hitted_enemy.nonzero(as_tuple=False) 返回形状[K, 2]的整型张量，每一行是一个(n, e)。再 .T 转置成 [2, K]
                env_idx, enemy_idx = newly_hitted_enemy.nonzero(as_tuple=False).T       # [K]，将第N个环境中被击中的第E个敌机的环境与敌机的索引拿出来。env_idx是这些新命中敌机的场景索引n列表，enemy_idx是对应敌机索引e列表（长度为K），一维。可以避免dist_masked0中未命中对的+inf，返回索引0的干扰
                friend_idx = hitter_idx[env_idx, enemy_idx]              # [K]，取出每个“新命中敌机”的击中者友机索引 j*，用上面拿出来的索引去取打中敌机的友机索引

                # —— 仅冻结击中者（友机侧）和被击中敌机（敌机侧） ——
                hit_friend_mask = torch.zeros_like(self.friend_frozen)    # [N,M] bool
                hit_friend_mask[env_idx, friend_idx] = True               # 标记第env_idx个环境中，第friend_idx个友机击中了敌机

                # —— 只把击中者记为“新冻友机”，并写捕获点 ——
                self._newly_frozen_friend |= hit_friend_mask               # [N,M] 用于记录当前步新冻友机，在奖励阶段用
                self._newly_frozen_enemy |= newly_hitted_enemy             # [N,E] 记录“本步新冻敌机”,在奖励阶段用

                # self.enemy_capture_pos[newly_hitted_enemy]      = en_pos0[newly_hitted_enemy]
                # self.friend_capture_pos[env_idx, friend_idx]    = en_pos0[newly_hitted_enemy]
                self.friend_frozen |= hit_friend_mask                     # 冻结击中者,self.friend_frozen用于全局记录冻结状态
                self.enemy_frozen  |= newly_hitted_enemy                  # 冻结被击中者

        # ---------- 分波发射：基于当前位置更新波次并解冻新一波 ----------
        self._update_wave_launch()

        fz = self.friend_frozen
        ez = self.enemy_frozen

        # ---------- 友机姿态/速度（冻结为0） ----------
        cos_th_now = torch.cos(self.theta).clamp_min(1e-6)                  # 当前俯仰角的余弦值
        Vm_eff = torch.where(fz, torch.zeros_like(self.Vm), self.Vm)        # 网络映射过来的友机的速度
        Vm_eps = Vm_eff.clamp_min(1e-6)
        theta_rate = self.g0 * (self._ny - cos_th_now) / Vm_eps
        psi_rate   = - self.g0 * self._nz / (Vm_eps * cos_th_now)
        theta_rate = torch.where(fz, torch.zeros_like(theta_rate), theta_rate)
        psi_rate   = torch.where(fz, torch.zeros_like(psi_rate),   psi_rate)

        THETA_RATE_LIMIT = 1.0
        PSI_RATE_LIMIT   = 1.0
        theta_rate = torch.clamp(theta_rate, -THETA_RATE_LIMIT, THETA_RATE_LIMIT)
        psi_rate   = torch.clamp(psi_rate,   -PSI_RATE_LIMIT,   PSI_RATE_LIMIT)

        self.theta = self.theta + theta_rate * dt                                               # 俯仰角 θt+1​=θt​+θ˙t​Δt
        self.psi_v = (self.psi_v + psi_rate * dt + math.pi) % (2.0 * math.pi) - math.pi         # 偏航角

        sin_th, cos_th = torch.sin(self.theta), torch.cos(self.theta)
        sin_ps, cos_ps = torch.sin(self.psi_v), torch.cos(self.psi_v)
        Vxm = Vm_eff * cos_th * cos_ps
        Vym = Vm_eff * sin_th
        Vzm = -Vm_eff * cos_th * sin_ps
        V_m = torch.stack([Vxm, Vym, Vzm], dim=-1)
        fr_vel_w_step = y_up_to_z_up(V_m)

        # 敌机速度（始终沿-axis_hat方向，以恒定速度前进，冻结为0）
        v_enemy_xy = -self._axis_hat_xy
        zeros_z = torch.zeros_like(v_enemy_xy[:, :1]) 
        v_move_3d = torch.cat([v_enemy_xy, zeros_z], dim=-1)
        v_move_expanded = v_move_3d.unsqueeze(1).expand(-1, self.E, -1)
        enemy_vel_step = v_move_expanded * float(self.cfg.enemy_speed)
        enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)

        # ---------- 推进 ----------
        fr_pos1 = fr_pos0 + fr_vel_w_step * dt
        en_pos1 = en_pos0 + enemy_vel_step * dt

        # ---------- 冻结 ----------
        if fz.any():
            fr_vel_w_step = torch.where(fz.unsqueeze(-1), torch.zeros_like(fr_vel_w_step), fr_vel_w_step)
            fr_pos1       = torch.where(fz.unsqueeze(-1), torch.zeros_like(self.friend_capture_pos), fr_pos1)
        if ez.any():
            enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)
            en_pos1        = torch.where(ez.unsqueeze(-1), torch.zeros_like(self.enemy_capture_pos), en_pos1)

        # ---------- 将积分后的速度写到全局变量中，并更新位置 ----------
        self.fr_vel_w  = fr_vel_w_step
        self.enemy_vel = enemy_vel_step
        self.fr_pos    = fr_pos1
        self.enemy_pos = en_pos1

        # ---------- 云台与可视化 ----------
        self._gimbal_control()
        self._refresh_enemy_cache()
        self._update_traj_vis()

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        N, M, E = self.num_envs, self.M, self.E
        dev = self.device
        dtype = self.fr_pos.dtype

        # --- 权重 ---
        centroid_weight  = float(getattr(self.cfg, "centroid_approach_weight", 1.0))
        v_to_c_weight    = float(getattr(self.cfg, "vel_to_centroid_weight", 0.0))
        hit_weight       = float(getattr(self.cfg, "hit_reward_weight", 100.0))
        fb_weight        = float(getattr(self.cfg, "w_gimbal_friend_block", 0.1))
        ec_weight        = float(getattr(self.cfg, "w_gimbal_enemy_cover", 0.1))
        enemy_reach_goal_weight = float(getattr(self.cfg, "enemy_reach_goal_penalty_weight", 100.0))
        R0               = float(getattr(self.cfg, "vc_zero_inside", 10.0))   # 近距离屏蔽半径
        friend_too_high_penalty_weight = float(getattr(self.cfg, "friend_too_high_penalty_weight", 0.0))  # 友机飞得过高惩罚权重
        friend_too_low_penalty_weight  = float(getattr(self.cfg, "friend_too_low_penalty_weight", 0.0))  # 友机飞得过高惩罚权重
        enemy_all_killed_reward_weight = float(getattr(self.cfg, "all_kill_weight", 100.0))
        target_align_weight = float(getattr(self.cfg, "w_target_align", 0.01))  # 速度对齐最近目标方向的权重
        leak_penalty_weight = float(getattr(self.cfg, "leak_penalty_weight", 0.05))  # 漏敌机惩罚权重
        leak_margin         = float(getattr(self.cfg, "leak_margin", 1.0))          # 漏敌机轴向裕度
        # --- 活跃掩码 / 质心 ---
        friend_active    = (~self.friend_frozen)                     # [N,M] bool
        enemy_active     = self._enemy_active                        # [N,E] bool  # 只看“存在且未冻结”
        enemy_active_any = self._enemy_active_any                    # [N]   bool
        vis_fe = self._gimbal_enemy_visible_mask()                   # [N, M, E]
        vis_ff = self._gimbal_friend_visible_mask()                  # [N, M, M]
        # 本局“启用”的友机（索引 < 当前 env 的 active_count）
        idx_f = torch.arange(M, device=dev).unsqueeze(0)            # [1, M]
        active_counts = self._current_active_count.unsqueeze(1)     # [N, 1]
        friend_enabled = (idx_f < active_counts)                    # [N, M] bool

        # ———————————————————— 质心接近增量（距离减小为正） ————————————————————
        centroid = self._enemy_centroid.unsqueeze(1).expand(N, M, 3)                                                # [N,M,3]
        diff = centroid - self.fr_pos                                                                               # [N,M,3]
        dist_now = torch.linalg.norm(diff, dim=-1)                                                                  # [N,M] 当前友机距离质心的距离
        dist_to_centroid_now = torch.where(enemy_active_any.unsqueeze(1), dist_now, self.prev_dist_centroid)        # 当敌机全灭时，保持距离不变，避免最后一步大幅负增量
        delta_dist = self.prev_dist_centroid - dist_to_centroid_now                                                 # [N,M]
        has_lock = vis_fe.any(dim=-1)                                                                               # [N, M]
        gate_lock = (~has_lock).float()
        final_centroid_gate = (dist_to_centroid_now > R0).float() * gate_lock
        centroid_each = delta_dist * gate_lock * friend_active.float()                                              # [N,M] 只计未冻结友机

        # ———————————————————— 速度朝向质心的对齐奖励（正向投影，单位m/s）————————————————————
        e_hat_c = torch.where(dist_to_centroid_now.unsqueeze(-1) > 1e-6, diff / dist_to_centroid_now.unsqueeze(-1), torch.zeros_like(diff))  # [N,M,3]
        v = self.fr_vel_w                                                # [N,M,3]
        v_proj_c = (v * e_hat_c).sum(dim=-1)                             # [N,M] (m/s)，(a*b).sum(dim=-1)表示向量点积,但由于b是单位向量，所以是投影长度
        vel_to_centroid_each = friend_active.float() * final_centroid_gate * torch.clamp(v_proj_c, min=0.0) # [N,M]

        # ———————————————————— 奖励agent靠近最近的敌机单位向量 ————————————————————
        target_align_each = torch.zeros((N, M), device=dev, dtype=dtype)
        if E > 0 and target_align_weight != 0.0:
            eps = 1e-9
            rel_all  = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)  # [N,M,E,3]
            dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)
            dir_all  = rel_all / dist_all                                       # [N,M,E,3]

            cam_dir = self._dir_from_yaw_pitch(self._gimbal_yaw, self._gimbal_pitch)  # [N,M,3]
            cam_dir = cam_dir.unsqueeze(2)                                              # [N,M,1,3]

            cos_ang = (cam_dir * dir_all).sum(dim=-1)
            cos_ang = cos_ang.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            angle   = torch.acos(cos_ang)                                              # [N,M,E]

            large_angle = math.pi
            angle_for_sort = torch.where(vis_fe,angle,torch.full_like(angle, large_angle),)      # [N,M,E]

            sort_idx = angle_for_sort.argsort(dim=-1)                                  # [N,M,E]

            # 视野内的单位向量
            e_hat_all = dir_all * vis_fe.unsqueeze(-1).float()                         # [N,M,E,3]

            # 最近那个
            first_idx = sort_idx[..., 0:1]                                             # [N,M,1]
            nearest_dir = torch.gather(e_hat_all,2,first_idx.unsqueeze(-1).expand(-1, -1, -1, 3),).squeeze(2)            # [N,M,3]

            # 对于“一个敌机都没看到”的友机，把方向置 0，避免给错奖励
            has_any = vis_fe.any(dim=-1)                                               # [N,M]
            nearest_dir = torch.where(has_any.unsqueeze(-1),nearest_dir,torch.zeros_like(nearest_dir),)           # [N,M,3]

            # 友机速度沿该方向的投影（负值截断为 0，只奖励正向靠近）
            v = self.fr_vel_w                                                          # [N,M,3]
            v_proj_target = (v * nearest_dir).sum(dim=-1)                              # [N,M]
            target_align_each = friend_active.float() * torch.clamp(v_proj_target, min=0.0)

        # ———————————————————— 命中奖励 ————————————————————
        per_agent_hit = self._newly_frozen_friend.float()              # [N,M]

        # ———————————————————— 全歼奖励 ————————————————————
        enemy_exists = self._enemy_exists_mask                                   # [N,E]
        mission_success = ((~enemy_exists) | self.enemy_frozen).all(dim=1, keepdim=True).float()

        # ———————————————————— 导引头视野内不能有友机惩罚 ————————————————————
        alive_friend_nums  = friend_active.sum(dim=1).to(dtype)                                   # [N] sum(dim=1)意味着沿着第二维求和，在[N,M]中也就是沿着第一行从左到右求和，行不变列变。sum(dim=0)就是沿着第一列求和。
        alive_friend_nums_ = (alive_friend_nums - 1.0).clamp_min(1.0).unsqueeze(1)                # [N,1] 活着的友机最多能看到友机数目
        pen_friend_each_cnt = vis_ff.float().sum(dim=2)                                                   # [N,M] 计算每个友机看到的友机数，沿着第三维M求和，第一行从左到右加起来
        penalty_friend_each = (pen_friend_each_cnt / alive_friend_nums_) * friend_active.float()  # [N,M] ∈[0,1] 友->友可见（遮挡）占比：每机看到的友机数 / 它最多能看到的友机数

        # ———————————————————— 导引头视野内敌机数目奖励 ————————————————————
        E_alive = (~self.enemy_frozen).sum(dim=1).to(dtype)                             # [N]
        den_e_each = E_alive.clamp_min(1.0).unsqueeze(1)                                # [N,1]
        enemy_count_each_cnt = vis_fe.float().sum(dim=-1)                                       # [N,M]
        enemy_count_each = (enemy_count_each_cnt / den_e_each) * friend_active.float()  # [N,M] ∈[0,1]

        # ———————————————————— 敌人质心抵达目标点惩罚 ————————————————————
        cen = self._enemy_centroid                                                                # [N,3]
        diff_c = cen[..., :2] - self._goal_e[..., :2]                                             # [N,2]
        dist2_c = diff_c.square().sum(dim=-1)                                                     # [N]
        enemy_goal_any = dist2_c < (float(self.cfg.enemy_goal_radius) ** 2)                       # [N] bool
        enemy_reach_goal_any = enemy_goal_any.float().unsqueeze(1) * friend_enabled.float()

        # ———————————————————— 友机飞的过高/低惩罚 ————————————————————
        z = self.fr_pos[:, :, 2]                                              # [N,M] 友机高度
        z_enemy_max, _ = self.enemy_pos[:, :, 2].max(dim=1)                       # [N] 每个环境中敌机的最高高度
        z_enemy_max = z_enemy_max.unsqueeze(1)                                    # [N,1]
        overshoot_z = (z - (z_enemy_max + 1.0)).clamp_min(0.0)                     # [N,M]
        penalty_friend_high_each = overshoot_z * friend_active.float()        # [N,M]
        gate_low = (dist_to_centroid_now < 50.0).float()
        lowshoot_z = (7.0 - z).clamp_min(0.0)                                # [N,M]
        penalty_friend_low_each = lowshoot_z * friend_active.float() * gate_low          # [N,M]

        # ———————————————————— overshoot惩罚 ————————————————————
        leak_each = torch.zeros((N, M), device=dev, dtype=dtype)              # [N,M]
        if leak_penalty_weight != 0.0 and E > 0:
            gk_3d   = self._goal_e                  # [N,3]
            axis_3d = self._axis_hat                # [N,3]

            axis_xy  = axis_3d[..., :2]
            norm_xy  = torch.linalg.norm(axis_xy, dim=-1, keepdim=True).clamp_min(1e-6)
            axis_hat = torch.cat(
                [axis_xy / norm_xy, torch.zeros_like(axis_3d[..., 2:3])],
                dim=-1,
            )                                      # [N,3]

            # 沿目标->敌团轴的标量投影
            sf = ((self.fr_pos    - gk_3d.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [N,M]
            se = ((self.enemy_pos - gk_3d.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [N,E]

            INF     = torch.tensor(float("inf"),     dtype=dtype, device=dev)
            NEG_INF = torch.tensor(float("-inf"),    dtype=dtype, device=dev)
            sf_mask = torch.where(friend_active, sf, INF)          # 冻结友机用 +inf 屏蔽
            se_mask = torch.where(enemy_active,  se, NEG_INF)      # 冻结敌机用 -inf 屏蔽

            # friend_min: 还活着的友机里，最靠“后面”的那个（离目标最近）
            friend_min = sf_mask.min(dim=1, keepdim=True).values   # [N,1]

            # “漏敌机”：敌机比 friend_min 更靠近目标 leak_margin 以上
            leaked_enemy = enemy_active & (se_mask < (friend_min - leak_margin))  # [N,E] bool
            num_leaked   = leaked_enemy.float().sum(dim=1, keepdim=True)          # [N,1]

            leak_each = num_leaked * friend_active.float()                        # [N,M]

        # --- 合成 per-agent reward ---
        # 质心接近
        r_each = centroid_weight * centroid_each                                  # [N,M]
        # 速度朝质心对齐
        r_each = r_each + v_to_c_weight * vel_to_centroid_each                    # [N,M]
        # 靠近最近敌机单位向量
        r_each = r_each + target_align_weight * target_align_each                  # [N,M]
        # 拦截奖励
        r_each = r_each + hit_weight * per_agent_hit                              # [N,M]
        # 云台项
        r_each = r_each - fb_weight * penalty_friend_each + ec_weight * enemy_count_each   # [N,M]
        # 敌人抵达目标点惩罚（均摊到每个友机）
        r_each = r_each - enemy_reach_goal_weight * enemy_reach_goal_any          # [N,M]
        # 全部歼灭奖励
        r_each = r_each + mission_success * enemy_all_killed_reward_weight * friend_enabled.float()
        # 友机飞得过高/低惩罚
        r_each = r_each - friend_too_high_penalty_weight * penalty_friend_high_each - friend_too_low_penalty_weight * penalty_friend_low_each  # [N,M]
        # overshoot
        r_each = r_each - leak_penalty_weight * leak_each
        # --- 写出字典 ---
        rewards = {agent: r_each[:, i] for i, agent in enumerate(self.possible_agents)}
        # --- 状态缓存/一次性标志 ---
        self.prev_dist_centroid = dist_to_centroid_now
        self._newly_frozen_enemy[:]  = False
        self._newly_frozen_friend[:] = False

        # ----------------------debug ----------------------
        r_centroid     = centroid_weight * centroid_each                                 # [N,M]
        r_v_to_c       = v_to_c_weight * vel_to_centroid_each                            # [N,M]
        r_hit          = hit_weight * per_agent_hit                                      # [N,M]
        r_gimbal_f     = - fb_weight * penalty_friend_each                               # [N,M]
        r_gimbal_e     = ec_weight * enemy_count_each                                    # [N,M]
        r_enemy_goal   = - enemy_reach_goal_weight * enemy_reach_goal_any                # [N,M]
        r_all_killed   = mission_success * enemy_all_killed_reward_weight * friend_enabled.float()  # [N,M]
        r_target_align = target_align_weight * target_align_each                         # [N,M]
        r_overshoot    = - leak_penalty_weight * leak_each                               # [N,M]
        r_fr_high      = - friend_too_high_penalty_weight * penalty_friend_high_each     # [N,M]
        r_fr_low       = - friend_too_low_penalty_weight  * penalty_friend_low_each      # [N,M]

        # 总 reward（和 r_each 一致）
        reward = (
            r_centroid
            + r_v_to_c
            + r_hit
            + r_gimbal_f
            + r_gimbal_e
            + r_enemy_goal
            + r_all_killed
            + r_target_align
            + r_overshoot
            + r_fr_high
            + r_fr_low
        )  # [N,M]

        # --- episode 级累计统计 ---
        if "total" not in self.episode_sums:
            base = torch.zeros_like(reward)
            self.episode_sums["total"]           = base.clone()
            self.episode_sums["centroid"]        = base.clone()
            self.episode_sums["vel_to_centroid"] = base.clone()
            self.episode_sums["hit"]             = base.clone()
            self.episode_sums["gimbal_f"]        = base.clone()
            self.episode_sums["gimbal_e"]        = base.clone()
            self.episode_sums["enemy_goal"]      = base.clone()
            self.episode_sums["all_killed"]      = base.clone()
            self.episode_sums["target_align"]    = base.clone()
            self.episode_sums["overshoot"]       = base.clone()
            self.episode_sums["friend_too_high"] = base.clone()
            self.episode_sums["friend_too_low"]  = base.clone()

        self.episode_sums["total"]           += reward
        self.episode_sums["centroid"]        += r_centroid
        self.episode_sums["vel_to_centroid"] += r_v_to_c
        self.episode_sums["hit"]             += r_hit
        self.episode_sums["gimbal_f"]        += r_gimbal_f
        self.episode_sums["gimbal_e"]        += r_gimbal_e
        self.episode_sums["enemy_goal"]      += r_enemy_goal
        self.episode_sums["all_killed"]      += r_all_killed
        self.episode_sums["target_align"]    += r_target_align
        self.episode_sums["overshoot"]       += r_overshoot
        self.episode_sums["friend_too_high"] += r_fr_high
        self.episode_sums["friend_too_low"]  += r_fr_low
        # ----------------------debug----------------------

        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        N         = self.num_envs
        device    = self.device
        r2_goal = float(self.cfg.enemy_goal_radius) ** 2
        xy_max2 = float(self.cfg.enemy_cluster_ring_radius + 50.0) ** 2
        # ---------- 基本终止判据（逐env） ----------
        enemy_exists = self._enemy_exists_mask                                    # [N,E]
        # “存在的敌机 都被冻结”才算成功；不存在的槽位自动视作满足
        success_all_enemies = ((~enemy_exists) | self.enemy_frozen).all(dim=1)   # [N]
        if success_all_enemies.any():
            print("all enemies destroied!!!!!!")

        z = self.fr_pos[:, :, 2]                                                  # 对每个环境、每个友机，都取坐标向量的索引 2（即第 3 个分量）也就是飞机的z高度
        z_enemy_max, _ = self.enemy_pos[:, :, 2].max(dim=1)                       # [N] 每个环境中敌机的最高高度
        z_enemy_max = z_enemy_max.unsqueeze(1)                                    # [N,1]
        out_z_any = ((z < 0.0) | (z > (z_enemy_max + 5.0))).any(dim=1)           # [N] Z 越界

        origin_xy = self.terrain.env_origins[:, :2].unsqueeze(1)
        dxy = self.fr_pos[..., :2] - origin_xy
        out_xy_any = (dxy.square().sum(dim=-1) > xy_max2).any(dim=1)              # [N] XY 越界。dxy.square()是逐元素平方，dx^2,dy^2，然后sum(dim=-1)是把最后一个维度加起来，得到dx^2+dy^2，然后和xy_max2比大小

        nan_inf_any = ~torch.isfinite(self.fr_pos).all(dim=(1, 2))                # [N] NaN/Inf

        # 敌人抵达目标点
        cen = self._enemy_centroid  # [N,3]
        diff_c = cen[..., :2] - self._goal_e[..., :2]  # [N,2]
        dist2_c = diff_c.square().sum(dim=-1)          # [N]
        enemy_goal_any = dist2_c < r2_goal
        if enemy_goal_any.any():
            print("enemy_reach_goal!!!!!!")

        overshoot_any  = torch.zeros(N, dtype=torch.bool, device=device)  # [N]
        alive_mask = ~(success_all_enemies | out_z_any | out_xy_any | nan_inf_any | enemy_goal_any)  # [N]
        idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)
        stage = self.friend_wave_stage[idx]
        can_overshoot = stage >= (self.friend_wave_count - 1)

        # if can_overshoot.any():
        if alive_mask.any():
            tol = float(getattr(self.cfg, "overshoot_tol", 2.0))
            idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)          # [n]
            friend_active = (~self.friend_frozen[idx])                    # [n,M]
            enemy_exists  = self._enemy_exists_mask[idx]                  # [n,E]
            enemy_active  = enemy_exists & (~self.enemy_frozen[idx])      # [n,E]
            have_both = friend_active.any(dim=1) & enemy_active.any(dim=1)
            if have_both.any():
                k_idx = have_both.nonzero(as_tuple=False).squeeze(-1)     # [k]
                gk_3d    = self._goal_e[idx][k_idx]                       # [k,3]
                axis_3d  = self._axis_hat[idx][k_idx]                     # [k,3]

                axis_xy  = axis_3d[..., :2]                               # [k,2]
                norm_xy  = torch.linalg.norm(axis_xy, dim=-1, keepdim=True).clamp_min(1e-6)
                axis_hat = torch.cat([axis_xy / norm_xy, torch.zeros_like(axis_3d[..., 2:3])], dim=-1)        # [k,3]

                sf = ((self.fr_pos[idx][k_idx]    - gk_3d.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [k,M] 友机在目标轴上的投影
                se = ((self.enemy_pos[idx][k_idx] - gk_3d.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [k,E]

                INF     = torch.tensor(float("inf"),  dtype=sf.dtype, device=sf.device)
                NEG_INF = torch.tensor(float("-inf"), dtype=sf.dtype, device=sf.device)
                sf_masked_for_min = torch.where(friend_active[k_idx], sf, INF)
                se_masked_for_max = torch.where(enemy_active[k_idx],  se, NEG_INF)

                friend_min = sf_masked_for_min.min(dim=1).values          # [k]
                enemy_max  = se_masked_for_max.max(dim=1).values          # [k]
                separated  = friend_min > (enemy_max + tol)
                overshoot_any[idx[k_idx]] = separated

        died     = out_z_any | out_xy_any | nan_inf_any | success_all_enemies | enemy_goal_any | overshoot_any  # [N]
        time_out = self.episode_length_buf >= self.max_episode_length - 1                                       # [N]

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
                "out_of_bounds_xy"   : cnt_oob_xy,
                "out_of_bounds_z"    : cnt_oob_z,
                "overshoot"          : cnt_overshoot,
                "enemy_goal"         : cnt_enemygoal,
                "success_all_enemies": cnt_success,
                "other"              : cnt_other
            }

        return dones, truncs

    def _reset_idx(self, env_ids: torch.Tensor | None):
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
            # === 打印本次 reset 的 env 的 reward 各分量 episode 累积和 ===
            # 注意：此时 episode_sums 里存的是“上一段 episode”的累计值
            if len(self.episode_sums) > 0 and env_ids is not None and len(env_ids) > 0:
                print("Reward components (sum over episode, per env; sum over agents):")
                for name, buf in self.episode_sums.items():
                    # buf: [num_envs, M]，先对 agent 维求和 → [num_envs]
                    vals = buf[env_ids].sum(dim=1)  # 每个 env 的总和
                    mean = vals.mean().item()
                    vmin = vals.min().item()
                    vmax = vals.max().item()
                    print(f"  {name:<16}: mean={mean:10.3f}  min={vmin:10.3f}  max={vmax:10.3f}")
                print("---------------------------------------------------------")
            if len(env_ids) > 0 and len(self.episode_sums) > 0:
                env0 = env_ids[0].item()
                M = self.M

                print(f"Reward components per agent for env {env0} (this episode):")
                for name, buf in self.episode_sums.items():
                    # buf: [N, M]
                    row = buf[env0]  # [M]，这一 env 下每个 agent 的累计奖励
                    # 打印成列表好看一点
                    vals = row.detach().cpu().tolist()
                    # 如果 agent 太多，可以只打印前几个
                    # vals = vals[:10]
                    print(f"  {name:<16}: {vals}")
                print("---------------------------------------------------------")

            # === 新增：打印拦截率（冻结敌机数 / 总敌机数） ===
            if self.E > 0 and env_ids is not None and len(env_ids) > 0:
                exists     = self._enemy_exists_mask[env_ids]          # [N_reset,E]
                frozen     = self.enemy_frozen[env_ids] & exists       # 只统计真实存在的敌机
                frozen_cnt = frozen.sum(dim=1)                         # [N_reset]
                total_per_env = exists.sum(dim=1).clamp_min(1)         # [N_reset]
                rate = frozen_cnt.float() / total_per_env.float()      # [N_reset]

                print("Interception rate per env (frozen existing enemies / existing enemies):")
                for i_local, env_id in enumerate(env_ids.tolist()):
                    c   = int(frozen_cnt[i_local].item())
                    tot = int(total_per_env[i_local].item())
                    r   = rate[i_local].item()
                    print(f"  Env {env_id}: {c} / {tot} = {r:.3f}")
                print(
                    f"  Summary: mean={rate.mean().item():.3f}  "
                    f"min={rate.min().item():.3f}  max={rate.max().item():.3f}"
                )

        N = len(env_ids)
        dev, dtype = self.device, self.fr_pos.dtype
        origins = self.terrain.env_origins[env_ids]

        # 清零 episode 统计
        for k in list(self.episode_sums.keys()):
            self.episode_sums[k][env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        # 清空冻结状态与捕获点
        self.friend_frozen[env_ids] = False
        self.enemy_frozen[env_ids]  = False
        self._newly_frozen_friend[env_ids] = False
        self._newly_frozen_enemy[env_ids]  = False
        self.friend_capture_pos[env_ids] = 0.0
        self.enemy_capture_pos[env_ids]  = 0.0

        # 轨迹缓存重置
        self._traj_len[env_ids] = 0
        self._traj_buf[env_ids] = 0.0

        # --------------- 敌机出生 ---------------
        self._spawn_enemy(env_ids)

        # === 刷新敌团缓存（保证 _axis_hat / _enemy_centroid 与本轮出生一致）===
        self._refresh_enemy_cache()

        # 重置敌机初速度
        axis_hat = self._axis_hat[env_ids]                      # [N_reset, 3]
        axis_xy = axis_hat[..., :2]                             # [N_reset, 2]
        norm_xy = torch.linalg.norm(axis_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        axis_xy_unit = axis_xy / norm_xy                        # [N_reset, 2]
        axis_level = torch.cat([axis_xy_unit, torch.zeros_like(axis_hat[..., 2:3])], dim=-1)         # [N_reset, 3]
        v_dir0 = (-axis_level).unsqueeze(1).expand(-1, self.E, -1)  # [N_reset, E, 3]
        enemy_vel0 = v_dir0 * float(self.cfg.enemy_speed)           # [N_reset, E, 3]
        self.enemy_vel[env_ids] = enemy_vel0

        # --------------- 友方出生(交错立体队形版) ---------------
        # 1. 计算朝向：axis_hat 指向“原点->质心”，所以面向质心用 axis_hat
        axis_hat_xy = self._axis_hat[env_ids, :2]                     # [N,2]
        face_xy     = axis_hat_xy
        face_norm   = torch.linalg.norm(face_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        f_hat       = face_xy / face_norm                             # [N,2] 前向(指向敌团)
        r_hat       = torch.stack([-f_hat[..., 1], f_hat[..., 0]], dim=-1)  # [N,2]

        # 2. 队形参数配置
        agents_per_row = int(self.cfg.agents_per_row)       # 每排数量 (建议 10)
        lat_spacing    = float(self.cfg.lat_spacing)      # 横向间隔 (同一排飞机间距)
        row_spacing    = float(self.cfg.row_spacing)      # 纵向间隔 (排与排之间距，需>4m以避开水平FOV)
        row_height_diff= float(self.cfg.row_height_diff)      # 高度阶梯 (后排比前排高 1m)
        base_altitude  = float(self.cfg.flight_altitude)

        # 3. 计算每个 Agent 的局部坐标 (M个)
        # 索引: 0~9 为第一排, 10~19 为第二排...
        agent_idxs = torch.arange(self.M, device=dev, dtype=dtype)
        row_idxs   = (agent_idxs // agents_per_row)   # [M] 第几排 (0, 0... 1, 1...)
        col_idxs   = (agent_idxs % agents_per_row)    # [M] 第几列 (0, 1... 0, 1...)

        # --- 纵向(X_local):沿f_hat反方向延伸(0, -5, -10...)
        x_local = - row_idxs * row_spacing

        # --- 横向 (Y_local): 沿 r_hat 展开 + 交错偏移
        # 基础位置：以中心为原点向两边展开
        # 例如 10架：-9, -7, ..., 7, 9 (单位: 半个间隔)
        y_local = (col_idxs - (agents_per_row - 1) / 2.0) * lat_spacing

        # 交错逻辑：如果是奇数排(第1,3排)，整体横移半个间隔(1.0m)
        # 这样后排正好对准前排的空隙
        stagger_shift = (row_idxs % 2) * (lat_spacing / 2.0)
        # 为了保持整体重心居中，奇数排 +0.5*S，偶数排其实是 0。
        # 加上 shift 后，奇数排会稍微偏右一点，这是正常的交错。
        # y_local = y_local + stagger_shift

        # --- 高度 (Z_local): 阶梯状上升
        z_local = base_altitude + row_idxs * row_height_diff

        # 4. 转换到世界坐标系 [N, M, 3]
        # 广播机制: [M] -> [1, M, 1] 与 [N, 1, 2] 结合
        x_expand = x_local.view(1, self.M, 1)    # [1, M, 1]
        y_expand = y_local.view(1, self.M, 1)    # [1, M, 1]
        f_expand = f_hat.unsqueeze(1)            # [N, 1, 2]
        r_expand = r_hat.unsqueeze(1)            # [N, 1, 2]

        # XY 平面位置 = 原点 + X_local * 前向 + Y_local * 右向
        offsets_xy = x_expand * f_expand + y_expand * r_expand  # [N, M, 2]

        fr0 = torch.empty(N, self.M, 3, device=dev, dtype=self.fr_pos.dtype)
        fr0[..., :2] = origins[:, :2].unsqueeze(1) + offsets_xy
        fr0[..., 2]  = origins[:, 2].unsqueeze(1) + z_local.view(1, self.M)

        # 5. 写入状态
        self.fr_pos[env_ids]   = fr0
        self.fr_vel_w[env_ids] = 0.0
        self.Vm[env_ids]       = 0.0

        # 6.姿态初始化(保持指向敌团质心)
        # 计算每架飞机到质心的向量
        centroid = self._enemy_centroid[env_ids]                     # [N,3]
        d = centroid.unsqueeze(1) - self.fr_pos[env_ids]             # [N,M,3]

        # 初始化距离缓存
        self.prev_dist_centroid[env_ids] = torch.linalg.norm(d, dim=-1)

        # 计算 Yaw (Z-up)
        psi0 = torch.atan2(d[..., 1], d[..., 0])
        psi0 = ((psi0 + math.pi) % (2.0 * math.pi)) - math.pi
        self.psi_v[env_ids] = psi0

        # 计算 Pitch (Y-up 动力学的 theta，即 Z-up 下的 Elevation)
        d_norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        sin_th = (d[..., 2] / d_norm.squeeze(-1)).clamp(-1.0+1e-6, 1.0-1e-6)
        theta0 = torch.asin(sin_th)
        self.theta[env_ids] = theta0

        # 云台重置 (Strapdown: 强制等于机体姿态)
        self._gimbal_yaw[env_ids]   = psi0
        self._gimbal_pitch[env_ids] = theta0

        # 动力学状态清零
        # self._ny[env_ids] = 0.0
        self._ny[env_ids] = torch.cos(theta0)
        self._nz[env_ids] = 0.0
        # --------------- 友方出生 结束 ---------------

        # 获取本组 env 的有效数量（刚才在 spawn_enemy 里算出来的，比如 27 或 16）
        active_counts_fr = self._current_active_count[env_ids]  # [N]

        # 生成掩码：Index >= active_count 的友机需要被“永久冻结”
        fr_idx = torch.arange(self.M, device=dev).unsqueeze(0)   # [1, M]
        thresholds_fr = active_counts_fr.unsqueeze(1)            # [N, 1]

        # permanent_disable_mask: 这些友机本局完全不参战
        permanent_disable_mask = (fr_idx >= thresholds_fr)       # [N, M]

        # -------- 分波发射 / 冻结处理 --------
        self.friend_wave_stage[env_ids] = 0
        N_reset = env_ids.shape[0]
        if self.friend_wave_enable and (self.friend_wave_count > 1):
            self.friend_capture_pos[env_ids] = fr0 # 假设 fr0 是你前面算好的友机位置
            wave_idx = self.friend_wave_index.to(dev)

            # 第0波掩码
            first_wave_mask = (wave_idx == 0).unsqueeze(0).expand(N_reset, self.M)

            # 初始冻结 = (非第0波) 或 (永久禁用)
            # 也就是说：即使你是第0波的第5架，如果本局只需要4架，那你也被冻结
            is_frozen = (~first_wave_mask) | permanent_disable_mask

            self.friend_frozen[env_ids] = is_frozen
        else:
            # 如果不分波，直接冻结多余的
            self.friend_frozen[env_ids] = permanent_disable_mask

    # def _get_observations(self) -> dict[str, torch.Tensor]:
    #     N, M, E = self.num_envs, self.M, self.E
    #     dev, dtype = self.device, self.fr_pos.dtype
    #     eps = 1e-9

    #     # ====================== 1. 友机相对观测 ======================
    #     pos_i = self.fr_pos.unsqueeze(2)   # [N,M,1,3]
    #     pos_j = self.fr_pos.unsqueeze(1)   # [N,1,M,3]
    #     dist_ij_raw = torch.linalg.norm(pos_j - pos_i, dim=-1)  # [N,M,M]

    #     # 把"自己"和"冻结友机"推到排序末尾
    #     large = torch.full_like(dist_ij_raw, 1e6)
    #     eye = torch.eye(M, device=dev, dtype=torch.bool).unsqueeze(0)
    #     friend_alive = (~self.friend_frozen)
    #     both_alive = friend_alive.unsqueeze(1) & friend_alive.unsqueeze(2)
    #     valid_pair = (~eye) & both_alive
    #     dist_ij = torch.where(valid_pair, dist_ij_raw, large)

    #     # 排序：近 -> 远
    #     sorted_idx = dist_ij.argsort(dim=-1)[:, :, :M-1]  # [N,M,M-1]

    #     # Gather 友机位置/速度
    #     other_pos_sorted = torch.gather(
    #         self.fr_pos.unsqueeze(1).expand(N, M, M, 3),
    #         2, sorted_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    #     )
    #     other_vel_sorted = torch.gather(
    #         self.fr_vel_w.unsqueeze(1).expand(N, M, M, 3),
    #         2, sorted_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    #     )

    #     # 转为相对量
    #     self_pos = self.fr_pos.unsqueeze(2)
    #     self_vel = self.fr_vel_w.unsqueeze(2)
    #     zeros_self = torch.zeros_like(self_pos)

    #     rel_pos_sorted = other_pos_sorted - self_pos
    #     rel_vel_sorted = other_vel_sorted - self_vel

    #     # 拼接友机信息 [N, M, 3*(M-1) * 2] -> [N, M, 6(M-1)]
    #     all_pos_sorted = torch.cat([zeros_self, rel_pos_sorted], dim=2).reshape(N, M, 3 * M)
    #     all_vel_sorted = torch.cat([zeros_self, rel_vel_sorted], dim=2).reshape(N, M, 3 * M)

    #     # ====================== 2. 单目标导引头观测 ======================
    #     # 目标：只输出视野中心那个敌机的方向向量 (3维) + 锁定标志 (1维)
    #     if E > 0:
    #         # --- A. 基础几何计算 ---
    #         vis_fe = self._gimbal_enemy_visible_mask()  # [N, M, E]
            
    #         rel_all  = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2) # [N,M,E,3]
    #         dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)
    #         dir_all  = rel_all / dist_all  # [N,M,E,3] 所有敌机的单位向量

    #         # --- B. 计算与云台光轴的夹角 ---
    #         cam_dir = self._dir_from_yaw_pitch(self._gimbal_yaw, self._gimbal_pitch).unsqueeze(2) # [N,M,1,3]
    #         # 点积求夹角余弦
    #         cos_ang = (cam_dir * dir_all).sum(dim=-1).clamp(-1.0+1e-6, 1.0-1e-6)
    #         angle   = torch.acos(cos_ang) # [N,M,E]

    #         # --- C. 筛选逻辑 (Seeker Logic) ---
    #         # 1. 屏蔽：把视野外(不可见)的敌机角度设为无穷大
    #         large_angle = 100.0
    #         angle_masked = torch.where(vis_fe, angle, torch.tensor(large_angle, device=dev))

    #         # 2. 优选：找到角度最小的那个敌机索引
    #         # min_vals: [N, M] 最小角度值
    #         # min_inds: [N, M] 对应敌机的索引
    #         min_vals, min_inds = angle_masked.min(dim=-1)

    #         # 3. 判定：是否真的锁定? (如果最小角度依然很大，说明视野全是空的)
    #         has_lock = (min_vals < (large_angle - 1.0)) # [N, M] Bool

    #         # 4. 提取：使用索引抓取对应的 3D 向量
    #         # min_inds 形状 [N, M] -> 扩展为 [N, M, 1, 3] 以适配 dir_all
    #         gather_idx = min_inds.unsqueeze(-1).unsqueeze(-1).expand(N, M, 1, 3)
    #         # best_vec: [N, M, 3] (这就是"可能"被锁定的敌机方向)
    #         best_vec = torch.gather(dir_all, 2, gather_idx).squeeze(2)

    #         # --- D. 最终输出掩码 ---
    #         # 如果没锁定(has_lock=False)，强制输出 0 向量
    #         single_target_vec = torch.where(
    #             has_lock.unsqueeze(-1), 
    #             best_vec, 
    #             torch.zeros_like(best_vec)
    #         )
            
    #         # 锁定标志位 (给神经网络明确信号：现在是有目标状态还是盲飞状态)
    #         lock_feat = has_lock.unsqueeze(-1).float() # [N, M, 1]
    #     else:
    #         single_target_vec = torch.zeros((N, M, 3), device=dev, dtype=dtype)
    #         lock_feat         = torch.zeros((N, M, 1), device=dev, dtype=dtype)

    #     # ====================== 3.自身状态 ======================
    #     # 本局“启用”的友机：索引 < 当前 env 的 active_count
    #     idx_f = torch.arange(M, device=dev).unsqueeze(0)            # [1, M]
    #     active_counts = self._current_active_count.unsqueeze(1)     # [N, 1]
    #     friend_enabled_feat = (idx_f < active_counts).to(dtype=dtype).unsqueeze(-1)  # [N, M, 1]
    #     self_pos_abs = self.fr_pos
    #     self_vel_abs = self.fr_vel_w

    #     # 敌团质心 (Search Guidance)：当 lock_feat=0 时，Agent 依赖这个飞
    #     cen = self._enemy_centroid
    #     rel_c = cen.unsqueeze(1) - self.fr_pos
    #     dist_c = torch.linalg.norm(rel_c, dim=-1, keepdim=True).clamp_min(eps)
    #     e_hat_c = rel_c / dist_c  # [N, M, 3]

    #     # Agent ID (One-hot)
    #     agent_id_feat = self._agent_id_onehot.expand(N, -1, -1).to(dtype=dtype)

    #     # ====================== 4. 拼接总观测 ======================
    #     obs_each = torch.cat(
    #         [
    #             all_pos_sorted,          # 3M
    #             all_vel_sorted,          # 3M
    #             friend_enabled_feat,     # 1
    #             self_pos_abs,            # 3
    #             self_vel_abs,            # 3
    #             e_hat_c,                 # 3  <-- 盲飞时的导航
    #             dist_c,                  # 1
    #             single_target_vec,       # 3  <-- 这里的维度现在固定为 3 了
    #             lock_feat,               # 1  <-- 告诉网络"single_target_vec"是否有效
    #             agent_id_feat,           # M
    #         ],
    #         dim=-1,
    #     )

    #     obs_dict = {ag: obs_each[:, i, :] for i, ag in enumerate(self.possible_agents)}

    #     return obs_dict

    def _get_observations(self) -> dict[str, torch.Tensor]:
        N, M, E = self.num_envs, self.M, self.E
        dev, dtype = self.device, self.fr_pos.dtype
        eps = 1e-9

        # ====================== 友机相对量（先做排序索引） ======================
        # fr_pos, fr_vel_w: [N, M, 3]
        pos_i = self.fr_pos.unsqueeze(2)                    # [N,M,1,3]
        pos_j = self.fr_pos.unsqueeze(1)                    # [N,1,M,3]
        dist_ij = torch.linalg.norm(pos_j - pos_i, dim=-1)   # [N,M,M] 友机i到友机j的欧氏距离
        dist_ij += torch.eye(M, device=dev, dtype=dtype).unsqueeze(0) * 1e6  # [N,M,M] 自身距离设极大，避免排序选到自己。torch.eye(M, device=dev, dtype=dtype)是创建一个MxM的单位矩阵，然后unsqueeze(0)变成1xMxM，广播后加到dist_ij上
        # 获取每个友机视角下的“其他友机”排序索引（近到远）
        sorted_idx = dist_ij.argsort(dim=-1)[:, :, :M-1]  # [N,M,M-1] 沿着最后一维把每个 (n,i,:) 这行从小到大排序，返回的是索引。[:, :, :M-1]意思是取前M-1个最近的友机索引（排除自己）
        # 按索引取“其他友机”的绝对位置/速度（后面再转相对）
        other_pos_sorted = torch.gather(
            self.fr_pos.unsqueeze(1).expand(N, M, M, 3),
            2, sorted_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # [N,M,M-1,3] gather函数是根据sorted_idx索引，从self.fr_pos中沿着第2维取值，得到每个友机看到的其他友机的位置，按距离排序
        other_vel_sorted = torch.gather(
            self.fr_vel_w.unsqueeze(1).expand(N, M, M, 3),
            2, sorted_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # [N,M,M-1,3]
        # —— 相对位置：self 记 0，其它友机用 j-i —— 
        self_pos = self.fr_pos.unsqueeze(2)  # [N,M,1,3]
        self_vel = self.fr_vel_w.unsqueeze(2)  # [N,M,1,3]
        zeros_self = torch.zeros_like(self_pos)  # [N,M,1,3]
        rel_pos_sorted = other_pos_sorted - self_pos  # [N,M,M-1,3]
        rel_vel_sorted = other_vel_sorted - self_vel  # [N,M,M-1,3]
        all_pos_sorted = torch.cat([zeros_self, rel_pos_sorted], dim=2).reshape(N, M, 3 * M)  # [N,M,3M]
        all_vel_sorted = torch.cat([zeros_self, rel_vel_sorted], dim=2).reshape(N, M, 3 * M)  # [N,M,3M]

        # ====================== 敌机方向（单位向量：敌 - 友）与相对速度 ======================
        if E > 0:
            # 1) 可见 + 未冻结 mask
            vis_fe = self._gimbal_enemy_visible_mask()                                     # [N,M,E]

            # 2) 敌机 LOS 单位向量（未排序）
            rel_all  = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)              # [N,M,E,3]
            dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)     # [N,M,E,1]
            dir_all  = rel_all / dist_all                                                  # [N,M,E,3] 未 mask 的方向

            # 3) 云台中心光轴方向（每个友机一个）
            cam_dir = self._dir_from_yaw_pitch(self._gimbal_yaw, self._gimbal_pitch)       # [N,M,3]
            cam_dir = cam_dir.unsqueeze(2)                                                 # [N,M,1,3]

            # 4) 计算和光轴的夹角：angle = arccos( d_cam · d_en )
            cos_ang = (cam_dir * dir_all).sum(dim=-1)                                      # [N,M,E]只要它们都做过normalize（或构造时就是单位向量），(cam_dir*dir_all).sum(-1)就是夹角余弦
            cos_ang = cos_ang.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            angle   = torch.acos(cos_ang)                                                  # [N,M,E]

            # 对“视野内”的敌机用真实角度；视野外直接给一个很大的角度，排在最后
            large_angle = math.pi  # 或者更大也行
            angle_for_sort = torch.where(
                vis_fe,
                angle,
                torch.full_like(angle, large_angle),
            )                                                                              # [N,M,E]

            # 5) 按夹角从小到大排序，得到每个友机视角下的敌机索引
            sort_idx = angle_for_sort.argsort(dim=-1)                                      # [N,M,E]

            # 6) 先应用可见 mask 得到真正的 e_hat_all，然后按 sort_idx 重排
            e_hat_all = dir_all * vis_fe.unsqueeze(-1).float()                             # [N,M,E,3]

            e_hat_all_sorted = torch.gather(
                e_hat_all,
                2,
                sort_idx.unsqueeze(-1).expand_as(e_hat_all),
            )                                                                              # [N,M,E,3]

            # 7) 展平到 [N,M,3E]，此时每个友机看到的敌机是
            #    “按与云台中心光轴夹角从小到大”的顺序排的
            e_hat_flat = e_hat_all_sorted.reshape(N, M, 3 * E)                             # [N,M,3E]
        else:
            e_hat_flat    = torch.zeros((N, M, 0), device=dev, dtype=dtype)

        # ====================== 自身状态（绝对位置 + 绝对速度） ======================
        idx_f = torch.arange(M, device=dev).unsqueeze(0)            # [1, M]
        active_counts = self._current_active_count.unsqueeze(1)     # [N, 1]
        friend_enabled_feat = (idx_f < active_counts).to(dtype=dtype).unsqueeze(-1)  # [N, M, 1]
        self_pos_abs = self.fr_pos                       # [N,M,3]
        self_vel_abs = self.fr_vel_w                     # [N,M,3]

        # ====================== 战术引导观测 (Tactical Obs) ======================
        # 敌团质心
        cen = self._enemy_centroid                   # [N, 3]
        rel_c = cen.unsqueeze(1) - self.fr_pos       # [N,M,3]
        dist_c = torch.linalg.norm(rel_c, dim=-1, keepdim=True).clamp_min(eps)  # [N,M,1]
        e_hat_c = rel_c / dist_c                     # [N,M,3]

        # ====================== 友机id ======================
        agent_id_feat = self._agent_id_onehot.expand(N, -1, -1).to(dtype=dtype)  # [N, M, M]

        # ====================== 拼接总观测 ======================
        obs_each = torch.cat(
            [
                all_pos_sorted,          # 3M
                all_vel_sorted,          # 3M
                e_hat_c,                 # 3
                dist_c,                  # 1
                e_hat_flat,              # 3E
                friend_enabled_feat,     # 1
                self_pos_abs,            # 3
                self_vel_abs,            # 3
                agent_id_feat,           # M
            ],
            dim=-1,
        )  # [N,M, 6M + 4E + 10]

        obs_dict = {ag: obs_each[:, i, :] for i, ag in enumerate(self.possible_agents)}

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
        "skrl_ippo_cfg_entry_point":  f"{agents.__name__}:L_M_interception_swarm_ippo_new.yaml",
        # "skrl_ippo_cfg_entry_point":  f"{agents.__name__}:L_M_interception_swarm_ippo_old.yaml", # 一个agent一个policy
    },
)
