from __future__ import annotations

import math
import os
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
from envs.interception_utils.bearing_estimation import CooperativeCVEKFEstimator
from envs.interception_utils.visualization import VisualizationHelper

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
    friendly_size: int = 10
    enemy_size: int = 10

    # 敌机相关
    debug_vis_enemy = True
    enemy_height_min = 10.0
    enemy_height_max = 15.0
    enemy_speed = 5.0
    enemy_target_alt = 10.0
    enemy_goal_radius = 1.0
    enemy_cluster_ring_radius: float = 100.0  # 敌机的生成距离
    enemy_cluster_radius: float = 15.0        # 敌机团的半径(固定队形中未使用)
    enemy_min_separation: float = 5.0         # 敌机间最小水平间隔
    enemy_vertical_separation: float = 5.0    # 立体队形敌机间最小垂直间隔
    hit_radius = 1.0                          # 命中半径
    enemy_max_num: int = 10                   # 敌机最多数量（可变编队时使用）
    enemy_min_num: int = 5                   # 敌机最少数量（可变编队时使用）
    friend_follow_enemy_num: bool = True      # 便捷开关：是否让友机数量自动跟随敌机数量（一对一）

    # 友方控制/速度范围/位置间隔
    Vm_min = 0.0
    Vm_max = 13.0
    ny_max_g = 3.0
    nz_max_g = 3.0
    flight_altitude = 5.0              # 友机最底层飞机高度
    # 友机队形参数
    agents_per_row: int     = 10       # 每排数量 (建议 10)
    lat_spacing: float      = 3.0      # 横向间隔 (同一排飞机间距)
    row_spacing: float      = 5.0      # 纵向间隔 (排与排之间距，需>4m以避开水平FOV)
    row_height_diff: float  = 3.0      # 高度阶梯 (后排比前排高 1m)

    # 观测相关配置
    obs_k_target: int = 10   # 观测最近的多少个敌机
    obs_k_friends: int = 10   # 观测最近的多少个友机

    # 奖励相关权重配置
    # centroid_approach_weight: float = 0.02
    # hit_reward_weight: float = 10.0
    # mission_success_weight: float = 10.0
    # leak_penalty_weight: float = 0.01
    # enemy_reach_goal_penalty_weight: float = 10.0
    # leak_margin: float = 1.0
    # w_gimbal_friend_block: float = 0.4
    # w_explore: float = 0.005
    # vc_zero_inside: float = 15.0
    # target_guide_weight: float = 0.02
    # coverage_reward_weight: float = 0.01       # 敌机覆盖奖励权重
    # # conflict_penalty_weight: float = 0.04


    centroid_approach_weight: float = 0.02
    hit_reward_weight: float = 10.0
    mission_success_weight: float = 10.0
    leak_margin: float = 1.0
    leak_penalty_weight: float = 0.01
    enemy_reach_goal_penalty_weight: float = 10.0
    w_gimbal_friend_block: float = 0.0
    w_explore: float = 0.005
    vc_zero_inside: float = 15.0
    target_guide_weight: float = 0.02
    coverage_reward_weight: float = 0.03       # 敌机覆盖奖励权重
    conflict_penalty_weight: float = 0.05



    # 友机-友机避障参数
    friend_collision_radius: float = 0.5          # 每架友机的虚拟球半径 (m)，两机间距 < 2*radius 视为碰撞
    friend_collision_penalty_weight: float = 0.5  # 友机之间发生碰撞的惩罚权重

    # ==== Gimabl VIS ====
    gimbal_vis_enable: bool = False          # 云台视野可视化开关
    gimbal_axis_vis_enable: bool = False     # 可视化云台光轴
    gimbal_fov_h_deg: float = 10.0      # 水平总 FOV（度）
    gimbal_fov_v_deg: float = 12.0      # 垂直总 FOV（度）
    gimbal_range_deg: float = 30.0      # 相对机体限位 ±30°
    gimbal_rate_deg:  float = 20.0      # 角速度 20°/s
    gimbal_effective_range: float = 100.0  # 云台“有效拍摄距离”（米）

    # ==== Bearing Vis ====
    bearing_vis_enable: bool = False       # 是否可视化 bearing 射线与估计点
    bearing_vis_max_envs: int = 1          # 可视化的前几个 env
    bearing_vis_num_friends: int = 30       # 每个 env 画前多少个友机的射线
    bearing_vis_num_enemies: int = 30       # 每个 env 画前多少个敌机/估计点
    bearing_vis_length: float = 100.0      # 射线长度（米）

    # ==== Traj Vis ====
    traj_vis_enable: bool = False            # 轨迹可视化开关
    traj_vis_max_envs: int = 1              # 只画前几个 env
    traj_vis_len: int = 500                 # 每个友机最多保留多少个轨迹点（循环缓冲）
    traj_vis_every_n_steps: int = 2         # 每隔多少个物理步记录/刷新一次
    traj_marker_size: tuple[float,float,float] = (0.05, 0.05, 0.05)  # 面包屑小方块尺寸

    # 频率
    episode_length_s = 50.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

    # for debug
    per_train_data_print: bool = False       # reset中打印

    # —— 单 agent 观测/动作维（用于 MARL 的 per-agent 空间）——
    single_observation_space: int = 9     # 将在 __post_init__ 基于 E 自动覆盖
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

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space      = spaces.Box(low=-1.0,   high=1.0,   shape=(1,), dtype=np.float32)
    state_space       = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    clip_action       = 1.0
    possible_agents: list[str] = []
    action_spaces: dict[str, gym.Space] = {}
    observation_spaces: dict[str, gym.Space] = {}

    def __post_init__(self):
        M = self.friendly_size if getattr(self, "friendly_size", None) is not None else self.swarm_size
        E = self.enemy_size    if getattr(self, "enemy_size", None)    is not None else self.swarm_size

        # agent 名单
        self.possible_agents = [f"drone_{i}" for i in range(int(M))]

        # 单智能体维度
        # single_obs_dim = 6 * self.obs_k_friends + 4 * self.obs_k_target + 6
        # single_obs_dim = 10 * self.obs_k_friends + 4 * self.obs_k_target + 6
        single_obs_dim = 10 * self.obs_k_friends + 4 * self.obs_k_target + 6 + 5*3*(M-1)
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
        act_dim = int(cfg.single_action_space)
        # single_obs_dim = 6 * cfg.obs_k_friends + 4 * cfg.obs_k_target + 6
        # single_obs_dim = 10 * cfg.obs_k_friends + 4 * cfg.obs_k_target + 6
        single_obs_dim = 10 * cfg.obs_k_friends + 4 * cfg.obs_k_target + 6 + 5*3*(M-1)
        cfg.single_observation_dim = single_obs_dim
        cfg.single_observation_space = single_obs_dim

        agents = [f"drone_{i}" for i in range(M)]
        cfg.possible_agents = agents

        ma_act_space  = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,),        dtype=np.float32)
        ma_obs_space  = spaces.Box(low=-np.inf, high=np.inf, shape=(single_obs_dim,), dtype=np.float32)
        cfg.action_spaces      = {a: ma_act_space for a in agents}
        cfg.observation_spaces = {a: ma_obs_space for a in agents}

        # 计算实际状态维度
        # 友机位置、速度、姿态(M*10) + 敌机位置、速度(E*6) + 敌机团中心位置(3)
        state_dim = M * 10 + E * 6 + 3
        cfg.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

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
        self._enemy_count = torch.full((N,), E, dtype=torch.long, device=dev)

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
        self._traj_buf  = torch.zeros(self.num_envs, self.M, int(self.cfg.traj_vis_len), 3, device=dev, dtype=dtype)  # [N,M,K,3]
        self._traj_len  = torch.zeros(self.num_envs, self.M, device=dev, dtype=torch.long) # [N,M]
        self._traj_markers: list[VisualizationMarkers] = []

        # —— 云台角 ——
        self._gimbal_yaw   = torch.zeros(N, self.M, device=dev, dtype=dtype)  # [-pi,pi)
        self._gimbal_pitch = torch.zeros(N, self.M, device=dev, dtype=dtype)  # 仰角

        # ------------------ 可视化与调试 ------------------
        self.friendly_visualizer = None
        self.enemy_visualizer    = None
        self.centroid_marker     = None
        self._fov_marker         = None
        self._traj_markers = []  # per-friend trajectory markers

        # Bearing 调试可视化
        self._bearing_ray_markers = []
        self._bearing_est_marker = None
        self._dbg_bearings = torch.zeros(N, self.M, self.E, 3, device=dev, dtype=dtype)
        self._dbg_est_pos_world = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)
        self._dbg_vis_fe = torch.zeros(N, self.M, self.E, dtype=torch.bool, device=dev)
        
        # 创建可视化辅助实例
        self._vis_helper = VisualizationHelper(self)
        self._vis_helper._bearing_ray_markers = self._bearing_ray_markers
        self._vis_helper._bearing_est_marker = self._bearing_est_marker
        self._vis_helper._traj_markers = self._traj_markers
        # 同步 gimbal FOV marker（如果需要的话）
        if hasattr(self, "_gimbal_fov_ray_marker"):
            self._vis_helper._gimbal_fov_ray_marker = self._gimbal_fov_ray_marker

        self.set_debug_vis(self.cfg.debug_vis)

        # ------------------- 奖励与指标缓存 ----------------------
        self.prev_dist_centroid = torch.zeros(N, M, device=self.device, dtype=torch.float32)

        # 用于统计“隐式分配质量”的 episode 级指标（仅做 logging，不参与梯度与决策）：
        # - _metric_steps:           每个 env 累计了多少个物理步（用于求时间平均）
        # - _metric_assign_coverage: 每步的"被看到敌机占比”之和
        # - _metric_conflict:        每步的"多友机追同一敌机占比”之和
        self._metric_steps = torch.zeros(N, device=dev, dtype=torch.float32)              # [N]
        self._metric_assign_coverage_sum = torch.zeros(N, device=dev, dtype=torch.float32)  # [N]
        self._metric_conflict_sum = torch.zeros(N, device=dev, dtype=torch.float32)         # [N]

        # ------------------ 敌团缓存（每步更新） ------------------
        self._enemy_centroid_init = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._enemy_centroid      = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._enemy_active        = torch.zeros(N, self.E, device=dev, dtype=torch.bool)
        self._enemy_active_any    = torch.zeros(N, device=dev, dtype=torch.bool)
        self._goal_e              = None
        self._axis_hat            = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._axis_hat_xy         = torch.zeros(N, 2, device=dev, dtype=dtype)
        self.enemy_goal_height    = torch.zeros(N, 1, device=dev, dtype=dtype)

        # ---- agent id one-hot feature ----
        self._agent_id_onehot = torch.eye(self.M, device=dev, dtype=torch.float32).unsqueeze(0)
        self._mission_failed_latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # —————————————————— ↓↓↓↓↓工具/可视化区↓↓↓↓↓ ——————————————————
    def _friendly_world_quats_zup(self) -> torch.Tensor:
        """由zup系下欧拉角的pitch是抬头正,低头负。而转换到四元数中pitch是抬头负、低头正。从print来看直接从速度反算yaw、pitch和直接拿yaw、pitch数值好像是一样的""" 
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
        exists = self._enemy_exists_mask                     # [N,E]
        enemy_active = exists & (~self.enemy_frozen)         # [N,E]
        e_mask = enemy_active.unsqueeze(-1).float()          # [N,E,1]
        sum_pos = (self.enemy_pos * e_mask).sum(dim=1)
        cnt     = e_mask.sum(dim=1).clamp_min(1.0)
        centroid = sum_pos / cnt

        self._enemy_centroid = centroid
        self._enemy_active   = enemy_active
        self._enemy_active_any = enemy_active.any(dim=1)

        axis = centroid - self._goal_e
        norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._axis_hat = axis / norm                # 敌方目标点指向敌团质心的单位向量

        axis_xy = centroid[:, :2] - self._goal_e[:, :2]
        norm_xy = axis_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._axis_hat_xy = axis_xy / norm_xy

    def _spawn_enemy(self, env_ids: torch.Tensor):
        # ---- 基本量 ----
        dev = self.fr_pos.device
        dtype = self.fr_pos.dtype
        env_ids = env_ids.to(dtype=torch.long, device=dev)
        N = env_ids.shape[0]

        # 槽位数（固定，和训练时一致）
        E_slots = int(self.E)

        origins_all = self.terrain.env_origins
        if origins_all.device != dev:
            origins_all = origins_all.to(dev)
        origins = origins_all[env_ids]  # [N, 3]

        if self._goal_e is None:
            self._rebuild_goal_e()
        goal_e = self._goal_e[env_ids]  # [N, 3]

        # 敌机实际启用数量上下界（可随便改）
        E_min_cfg = int(getattr(self.cfg, "enemy_min_num", 12))
        E_max_cfg = int(getattr(self.cfg, "enemy_max_num", E_slots))
        E_min = max(1, min(E_slots, E_min_cfg))
        E_max = max(E_min, min(E_slots, E_max_cfg))

        s_min = float(self.cfg.enemy_min_separation)                            # 敌机间隔
        sz_v = float(getattr(self.cfg, "enemy_vertical_separation", s_min))     # 垂直间隔
        hmin = float(self.cfg.enemy_height_min)                                 # 敌机最小高度
        hmax = float(self.cfg.enemy_height_max)                                 # 敌机最大高度(立体队形会超过这个高度，这里的最小最高值仅作为初始生成高度范围)
        R_center = float(getattr(self.cfg, "enemy_cluster_ring_radius", 8.0))   # 敌机团距离原点的生成距离

        # 泊松盘相关参数
        eta_poisson = float(getattr(self.cfg, "enemy_poisson_eta", 0.7))
        r_small = float(getattr(self.cfg, "enemy_cluster_radius", 10.0))

        # ==================================================================
        #  工具函数：中心化 / 网格 / 各种队形模板
        # ==================================================================
        def _centerize(xyz: torch.Tensor) -> torch.Tensor:
            return xyz - xyz.mean(dim=-2, keepdim=True)

        def _rect2d_dims_exact(E: int, aspect_pref: float = 2.0, aspect_max: float = 3.0):
            best = None
            best_rc = None
            for r in range(1, int(math.sqrt(E)) + 1):
                if E % r != 0:
                    continue
                c = E // r
                aspect = max(c / r, r / c)
                if aspect > aspect_max:
                    continue
                err = abs((c / r) - aspect_pref)
                score = (err, aspect)
                if best is None or score < best:
                    best = score
                    best_rc = (r, c)
            return best_rc

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
            xs = torch.arange(cols, dtype=dtype, device=dev)
            ys = torch.arange(rows, dtype=dtype, device=dev)
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

        # ---- 队形模板 ----
        def _tmpl_v_wedge_2d(E: int, s: float) -> torch.Tensor:
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)
            step = s / math.sqrt(2.0)
            if E == 1:
                return torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=dev)
            K = (E - 1) // 2
            ks = torch.arange(1, K + 1, dtype=dtype, device=dev)
            up = torch.stack([ks * step, ks * step, torch.zeros_like(ks)], dim=-1)
            down = torch.stack([ks * step, -ks * step, torch.zeros_like(ks)], dim=-1)
            pts = torch.cat([torch.zeros(1, 3, dtype=dtype, device=dev), up, down], dim=0)
            if (E - 1) % 2 == 1:
                extra_k = torch.tensor([(K + 1) * step], dtype=dtype, device=dev)
                extra = torch.stack([extra_k, extra_k, torch.zeros_like(extra_k)], dim=-1)
                pts = torch.cat([pts, extra], dim=0)
            return _centerize(pts[:E, :])

        def _tmpl_rect_2d(E: int, s: float, aspect: float = 2.0) -> torch.Tensor:
            rc = _rect2d_dims_exact(E, aspect_pref=aspect, aspect_max=3.0)
            if rc is None:
                r, c = _rect2d_dims(E, aspect)
            else:
                r, c = rc
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
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)
            n = round(E ** (1.0 / 3.0))
            assert n ** 3 == E, f"Cube 模板只应该接到完全立方数, got E={E}"
            xs = torch.arange(n, dtype=dtype, device=dev)
            ys = torch.arange(n, dtype=dtype, device=dev)
            zs = torch.arange(n, dtype=dtype, device=dev)
            X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
            Xf, Yf, Zf = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
            base_xyz = torch.stack([Xf * s, Yf * s, Zf * sz_], dim=-1)
            return _centerize(base_xyz)

        def _tmpl_rect_3d_reverse(E: int, s: float, sz_: float, aspect_xy: float = 2.0) -> torch.Tensor:
            L = 2
            cap_layer = max(1, math.ceil(E / L))
            r, c = _best_rc(cap_layer, aspect_xy)
            xyz = _grid3d(c, r, L, s, s, sz_)[:E, :]
            return xyz

        def _tmpl_poisson_3d(E: int, s: float, eta: float = 0.7) -> torch.Tensor:
            """
            在局部坐标系内做 Poisson disk 采样，返回 [E,3]。
            XY: Poisson disk 分布。
            Z : 在高度范围内随机分布（相对于中心平面）。
            """
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)

            two_pi = 2.0 * math.pi
            r_needed = 0.5 * s * math.sqrt(E / max(eta, 1e-6))
            r_env = max(r_small, r_needed * 1.02)

            BATCH        = 128
            MAX_ROUNDS   = 256
            STAGN_ROUNDS = 5
            GROW_FACTOR  = 1.05

            pts = torch.zeros(E, 2, dtype=dtype, device=dev)
            filled = 0
            stagn  = 0
            s2 = s * s

            # --- 1. 生成 2D Poisson Disk (XY) ---
            for _ in range(MAX_ROUNDS):
                if filled >= E:
                    break

                u = torch.rand(BATCH, device=dev, dtype=dtype)
                v = torch.rand(BATCH, device=dev, dtype=dtype)
                rr  = r_env * torch.sqrt(u.clamp_min(1e-12))
                ang = two_pi * v
                cand = torch.stack([rr * torch.cos(ang), rr * torch.sin(ang)], dim=-1)

                if filled == 0:
                    pts[0] = cand[0]
                    filled = 1
                    stagn = 0
                    continue

                diff = cand.unsqueeze(1) - pts[:filled].unsqueeze(0)
                sq   = (diff ** 2).sum(dim=-1)
                min_sq, _ = sq.min(dim=1)
                ok = min_sq >= s2

                if ok.any():
                    idx = torch.nonzero(ok, as_tuple=False)[0, 0]
                    pts[filled] = cand[idx]
                    filled += 1
                    stagn = 0
                else:
                    stagn += 1
                    if stagn >= STAGN_ROUNDS:
                        r_env *= GROW_FACTOR
                        stagn = 0

            # 备份策略
            if filled < E:
                EXTRA_GROW_STEPS = 8
                for _ in range(EXTRA_GROW_STEPS):
                    if filled >= E:
                        break
                    r_env *= GROW_FACTOR

                    u = torch.rand(BATCH, device=dev, dtype=dtype)
                    v = torch.rand(BATCH, device=dev, dtype=dtype)
                    rr  = r_env * torch.sqrt(u.clamp_min(1e-12))
                    ang = two_pi * v
                    cand = torch.stack([rr * torch.cos(ang), rr * torch.sin(ang)], dim=-1)

                    if filled == 0:
                        pts[0] = cand[0]
                        filled = 1
                        continue

                    diff = cand.unsqueeze(1) - pts[:filled].unsqueeze(0)
                    sq   = (diff ** 2).sum(dim=-1)
                    min_sq, _ = sq.min(dim=1)
                    ok = min_sq >= s2

                    if ok.any():
                        idx = torch.nonzero(ok, as_tuple=False)[0, 0]
                        pts[filled] = cand[idx]
                        filled += 1

            if filled < E:
                raise RuntimeError(
                    f"Poisson template failed for E={E}, s_min={s}. "
                    f"Consider increasing enemy_cluster_radius or decreasing E/s_min."
                )

            # --- 2. 生成随机 Z 高度 ---
            # 我们希望生成的Z是在一个范围内随机的。
            # 为了配合后面第4步中统一加上的 z_bottom (hmin ~ hmax)，这里的 Z 应该是一个相对偏移。
            # 假设我们希望整个集群的高度范围大致是 hmax - hmin。
            # 我们生成 [-half_spread, +half_spread] 的随机偏移。
            vertical_spread = (hmax - hmin) * 0.5
            z = (torch.rand(E, 1, dtype=dtype, device=dev) - 0.5) * (2.0 * vertical_spread)
            
            xyz = torch.cat([pts, z], dim=-1)               # [E,3]
            return _centerize(xyz)

        # ==================================================================
        # 1) 先在 [E_min, E_max] 内，为每种模板计算“合法数量”
        # ==================================================================
        def get_valid_counts(tmpl_id: int, min_n: int, max_n: int):
            valid = []
            if tmpl_id == 0: # V 字
                for x in range(min_n, max_n + 1):
                    if x % 2 == 1: valid.append(x)
            elif tmpl_id == 1: # Rect 2D
                for x in range(min_n, max_n + 1):
                    if _rect2d_dims_exact(x, aspect_pref=2.0, aspect_max=3.0) is not None:
                        valid.append(x)
            elif tmpl_id == 2: # Square 2D
                k = 1
                while True:
                    sq = k * k
                    if sq > max_n: break
                    if sq >= min_n: valid.append(sq)
                    k += 1
            elif tmpl_id in [3, 5]: # Rect 3D / Reverse
                for x in range(min_n, max_n + 1):
                    if x % 2 != 0: continue
                    cap = x // 2
                    r, c = _best_rc(cap, aspect_xy=2.0)
                    if r * c == cap: valid.append(x)
            elif tmpl_id == 4: # Cube 3D
                k = 1
                while True:
                    cb = k ** 3
                    if cb > max_n: break
                    if cb >= min_n: valid.append(cb)
                    k += 1
            elif tmpl_id == 6: # Poisson 3D (New)
                for x in range(min_n, max_n + 1):
                    valid.append(x)
            return valid

        # 汇总所有模板“能拼出”的数量
        all_valid_counts = set()
        for t in range(7):   # 模板 0~6
            all_valid_counts.update(get_valid_counts(t, E_min, E_max))
        all_valid_counts = sorted(list(all_valid_counts))

        if not all_valid_counts:
            all_valid_counts = list(range(E_min, E_max + 1))

        counts_tensor = torch.tensor(all_valid_counts, device=dev, dtype=torch.long)
        rand_idx = torch.randint(0, len(all_valid_counts), (N,), device=dev)
        chosen_counts = counts_tensor[rand_idx]  # [N]

        self._enemy_count[env_ids] = chosen_counts
        if hasattr(self, "_current_active_count"):
            self._current_active_count[env_ids] = chosen_counts

        # ==================================================================
        # 2) 对于每个数量 Ei，再选一个能拼出 Ei 的模板 ID
        # ==================================================================
        def _valid_templates_for_E(Ei: int) -> list[int]:
            t_list = []
            for tid in range(7):
                vc = get_valid_counts(tid, Ei, Ei)
                if len(vc) > 0:
                    t_list.append(tid)
            if not t_list:
                return list(range(7))
            return t_list

        template_ids = torch.zeros(N, dtype=torch.long, device=dev)
        for i in range(N):
            Ei = int(chosen_counts[i].item())
            valid_tpl = _valid_templates_for_E(Ei)
            tid = valid_tpl[torch.randint(0, len(valid_tpl), (), device=dev).item()]
            template_ids[i] = tid

        # ==================================================================
        # 3) 生成“局部坐标系下”的敌机点云
        # ==================================================================
        local_pos_buffer = torch.zeros(N, E_slots, 3, device=dev, dtype=dtype)

        for t_id in range(7):
            env_mask = (template_ids == t_id)
            if not env_mask.any():
                continue
            idx_env = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
            counts_this = chosen_counts[idx_env]

            unique_counts = torch.unique(counts_this)
            for c in unique_counts:
                count_val = int(c.item())
                sub_mask = (counts_this == c)
                final_indices = idx_env[sub_mask]

                if t_id == 0:
                    pts = _tmpl_v_wedge_2d(count_val, sz_v)
                elif t_id == 1:
                    pts = _tmpl_rect_2d(count_val, s_min, aspect=2.0)
                elif t_id == 2:
                    pts = _tmpl_square_2d(count_val, s_min)
                elif t_id == 3:
                    pts = _tmpl_rect_3d(count_val, s_min, sz_v, aspect_xy=2.0)
                elif t_id == 4:
                    pts = _tmpl_cube_3d(count_val, s_min, sz_v)
                elif t_id == 5:
                    pts = _tmpl_rect_3d_reverse(count_val, s_min, sz_v, aspect_xy=2.0)
                elif t_id == 6:
                    pts = _tmpl_poisson_3d(count_val, s_min, eta_poisson)
                else:
                    pts = _tmpl_rect_2d(count_val, s_min)

                pts = pts.clone()
                pts[..., 0] *= -1.0 # 翻转 X

                local_pos_buffer[final_indices, :count_val, :] = pts.unsqueeze(0)

        # ==================================================================
        # 4) 旋转 / 平移到世界坐标 + 处理高度
        # ==================================================================
        theta = 2.0 * math.pi * torch.rand(N, device=dev, dtype=dtype)
        centers = torch.stack([
            origins[:, 0] + R_center * torch.cos(theta),
            origins[:, 1] + R_center * torch.sin(theta)
        ], dim=1)

        head_vec = (goal_e[:, :2] - centers)
        head = head_vec / head_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        c, s = head[:, 0], head[:, 1]

        Rm = torch.stack([
            torch.stack([c, -s], dim=-1),
            torch.stack([s,  c], dim=-1)
        ], dim=1)

        local_xy = local_pos_buffer[:, :, :2]
        xy_rot = torch.matmul(local_xy, Rm.transpose(1, 2))
        xy = centers.unsqueeze(1) + xy_rot

        # 高度处理
        local_z = local_pos_buffer[:, :, 2:3]
        
        # 基础高度也在 hmin 到 hmax 之间随机
        # 注意：如果是泊松采样，local_z 已经有波动。叠加后可能会略微超出 hmin/hmax。
        # 如果需要严格限制，可以在下面 clamp。这里保持原有逻辑，叠加一个随机底座高度。
        z_bottom = hmin + torch.rand(N, 1, 1, device=dev, dtype=dtype) * max(1e-6, (hmax - hmin))
        z_abs = origins[:, 2:3].unsqueeze(1) + z_bottom + local_z

        enemy_pos = torch.cat([xy, z_abs], dim=-1)

        # ==================================================================
        # 5) 根据 chosen_counts 构造 exists_mask / frozen
        # ==================================================================
        idx_e = torch.arange(E_slots, device=dev).unsqueeze(0)
        cnts = chosen_counts.unsqueeze(1)
        exists_mask = idx_e < cnts

        enemy_pos = torch.where(exists_mask.unsqueeze(-1), enemy_pos, torch.zeros_like(enemy_pos))

        self.enemy_pos[env_ids] = enemy_pos
        self.enemy_capture_pos[env_ids] = enemy_pos

        self._enemy_exists_mask[env_ids] = exists_mask
        self.enemy_frozen[env_ids] = ~exists_mask

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
                self.friendly_visualizer.set_visibility(True)

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

    def _debug_vis_callback(self, event):
        # if self.friendly_visualizer is not None:
        #     pos = self.fr_pos.reshape(-1, 3)  # [N*M,3]
        #     quat = torch.zeros(pos.shape[0], 4, device=self.device)
        #     quat[:, 0] = 1.0  # 单位四元数 (w=1,x=0,y=0,z=0)
        #     scale = torch.ones_like(pos)
        #     self.friendly_visualizer.visualize(translations=pos, orientations=quat, scales=scale)
        if self.friendly_visualizer is not None:
            fr_quats = self._friendly_world_quats_zup()
            self.friendly_visualizer.visualize(
                translations=self._flatten_agents(self.fr_pos),
                orientations=self._flatten_agents(fr_quats)
            )
        # if self.enemy_visualizer is not None:
        #     self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))
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

    def _wrap_pi(self, x: torch.Tensor) -> torch.Tensor:
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    def _gimbal_control(self):
        self._gimbal_yaw = self._wrap_pi(self.psi_v)
        self._gimbal_pitch = self.theta

        # 可视化
        if getattr(self.cfg, "gimbal_vis_enable", False):
            k = int(getattr(self.cfg, "gimbal_vis_stride", 1))  # 每 k 步可视化一次
            if (k <= 1) or (int(self.progress_buf[0].item()) % k == 0):
                self._vis_helper.update_gimbal_fov_vis()

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
        exists_alive_e = (self._enemy_exists_mask & (~self.enemy_frozen)).unsqueeze(1)  # [N,1,E]

        # m = in_fov & in_rng & exists_alive_e  # [N_env, N_fr, N_en]
        # env_id = 0
        # print("======= env", env_id, "gimbal visible enemies per friend =======")
        # for fr_id in range(M):
        #     vis_idx = m[env_id, fr_id].nonzero(as_tuple=True)[0]  # [K] 敌机下标
        #     print(f"env {env_id}, friend {fr_id}, enemies idx:", vis_idx.tolist())
        # print("=======================================================")

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
        # env_id = 0
        # print("======= FRIEND gimbal visible (env", env_id, ") =======")
        # for fr_i in range(M):
        #     vis_idx = m[env_id, fr_i].nonzero(as_tuple=True)[0]  # [K]，所有 j 的索引
        #     print(f"env {env_id}, friend {fr_i} sees friends:", vis_idx.tolist())
        # print("=======================================================")

        return in_fov & in_rng & (~eye) & alive.unsqueeze(2) & alive.unsqueeze(1)

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

    @torch.no_grad()
    def _continuous_hit_detection(self,
                                fr_pos0: torch.Tensor,
                                en_pos0: torch.Tensor,
                                fr_vel_w_step: torch.Tensor,
                                enemy_vel_step: torch.Tensor,
                                dt: float):
        """在 [t, t+dt] 内做友机-敌机连续碰撞检测（CCD），更新 frozen 与 capture_pos

        采用相对运动：d(t)=d0+v_rel*t，判断是否存在 t∈[0,dt] 使 |d(t)|<=r。
        若命中，则为每个“新命中的敌机”选择一个责任友机（最早碰撞时刻 t_hit 最小），并冻结该友机和敌机。
        """
        r = float(self.cfg.hit_radius)

        fz = self.friend_frozen    # [N,M]
        ez = self.enemy_frozen     # [N,E]
        active_pair = (~fz).unsqueeze(2) & (~ez).unsqueeze(1)   # [N,M,E]
        if not active_pair.any():
            return

        # [N,M,E,3]
        d0 = fr_pos0.unsqueeze(2) - en_pos0.unsqueeze(1)
        v_rel = fr_vel_w_step.unsqueeze(2) - enemy_vel_step.unsqueeze(1)

        d0_sq = (d0 * d0).sum(dim=-1)                             # [N,M,E]
        a = (v_rel * v_rel).sum(dim=-1)                           # [N,M,E]
        b = 2.0 * (d0 * v_rel).sum(dim=-1)                        # [N,M,E]
        c = d0_sq - (r * r)                                       # [N,M,E]

        EPS_A = 1e-8
        small_a = a < EPS_A

        # 初始化：未命中记为 +inf
        INF = torch.tensor(float("inf"), device=self.device, dtype=d0_sq.dtype)
        t_hit_all = torch.full_like(d0_sq, INF)

        # 情况1：几乎无相对运动 => 距离基本不变，只看起点是否在半径内
        inside_start = (d0_sq <= (r * r)) & active_pair
        t_hit_all = torch.where(inside_start, torch.zeros_like(t_hit_all), t_hit_all)

        # 情况2：正常相对运动 => 解二次不等式 a t^2 + b t + c <= 0
        a_safe = a.clamp_min(EPS_A)
        disc = b * b - 4.0 * a_safe * c
        valid_disc = (disc >= 0.0) & (~small_a) & active_pair

        if valid_disc.any():
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
            t1 = (-b - sqrt_disc) / (2.0 * a_safe)
            t2 = (-b + sqrt_disc) / (2.0 * a_safe)

            # 命中条件：区间 [t1,t2] 与 [0,dt] 有交集
            hit = valid_disc & (t2 >= 0.0) & (t1 <= dt)

            # 取最早进入半径的时刻
            t_first = torch.clamp(t1, 0.0, dt)
            t_hit_all = torch.where(hit, torch.minimum(t_hit_all, t_first), t_hit_all)

        # 命中对
        hit_pair = torch.isfinite(t_hit_all) & (t_hit_all <= dt) & active_pair
        if not hit_pair.any():
            return

        newly_hitted_enemy = hit_pair.any(dim=1)  # [N,E]
        if not newly_hitted_enemy.any():
            return

        # 为每个敌机选择“责任友机”：t_hit 最小
        t_masked = torch.where(hit_pair, t_hit_all, INF)          # [N,M,E]
        hitter_idx = t_masked.argmin(dim=1)                       # [N,E]

        env_idx, enemy_idx = newly_hitted_enemy.nonzero(as_tuple=False).T  # [K]
        friend_idx = hitter_idx[env_idx, enemy_idx]                          # [K]
        t_hit = t_hit_all[env_idx, friend_idx, enemy_idx].unsqueeze(-1)      # [K,1]

        fr_hit_pos = fr_pos0[env_idx, friend_idx] + fr_vel_w_step[env_idx, friend_idx] * t_hit
        en_hit_pos = en_pos0[env_idx, enemy_idx]   + enemy_vel_step[env_idx, enemy_idx] * t_hit

        # 捕获点：取中点（可视化更干净）
        cap_pos = 0.5 * (fr_hit_pos + en_hit_pos)
        # cap_pos = 0.0
        self.friend_capture_pos[env_idx, friend_idx] = cap_pos
        self.enemy_capture_pos[env_idx, enemy_idx]   = cap_pos

        hit_friend_mask = torch.zeros_like(self.friend_frozen)
        hit_friend_mask[env_idx, friend_idx] = True

        self._newly_frozen_friend |= hit_friend_mask
        self._newly_frozen_enemy  |= newly_hitted_enemy

        self.friend_frozen |= hit_friend_mask
        self.enemy_frozen  |= newly_hitted_enemy

    def _apply_action(self):
        dt = float(self.physics_dt)
        is_first_substep = ((self._sim_step_counter - 1) % self.cfg.decimation) == 0
        if is_first_substep:
            self._newly_frozen_friend[:] = False
            self._newly_frozen_enemy[:]  = False

        # 步首状态
        fr_pos0 = self.fr_pos.clone()
        en_pos0 = self.enemy_pos.clone()

        fz = self.friend_frozen
        ez = self.enemy_frozen

        # ---------- 友机姿态/速度（冻结为0） ----------
        cos_th_now = torch.cos(self.theta).clamp_min(1e-6)
        Vm_eff = torch.where(fz, torch.zeros_like(self.Vm), self.Vm)
        Vm_eps = Vm_eff.clamp_min(1e-6)

        theta_rate = self.g0 * (self._ny - cos_th_now) / Vm_eps
        psi_rate   = -self.g0 * self._nz / (Vm_eps * cos_th_now)

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

        # ---------- 敌机速度（冻结为0） ----------
        v_enemy_xy = -self._axis_hat_xy
        zeros_z = torch.zeros_like(v_enemy_xy[:, :1])
        v_move_3d = torch.cat([v_enemy_xy, zeros_z], dim=-1)
        v_move_expanded = v_move_3d.unsqueeze(1).expand(-1, self.E, -1)
        enemy_vel_step = v_move_expanded * float(self.cfg.enemy_speed)
        enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)

        # ---------- 连续命中检测（关键） ----------
        self._continuous_hit_detection(
            fr_pos0=fr_pos0,
            en_pos0=en_pos0,
            fr_vel_w_step=fr_vel_w_step,
            enemy_vel_step=enemy_vel_step,
            dt=dt,
        )

        # 命中后冻结 mask 可能更新了
        fz = self.friend_frozen
        ez = self.enemy_frozen

        # ---------- 推进 ----------
        fr_pos1 = fr_pos0 + fr_vel_w_step * dt
        en_pos1 = en_pos0 + enemy_vel_step * dt

        # ---------- 冻结 ----------
        if fz.any():
            fr_vel_w_step = torch.where(fz.unsqueeze(-1), torch.zeros_like(fr_vel_w_step), fr_vel_w_step)
            fr_pos1       = torch.where(fz.unsqueeze(-1), self.friend_capture_pos, fr_pos1)
        if ez.any():
            enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)
            en_pos1        = torch.where(ez.unsqueeze(-1), self.enemy_capture_pos, en_pos1)

        # ---------- 写回 ----------
        self.fr_vel_w  = fr_vel_w_step
        self.enemy_vel = enemy_vel_step
        self.fr_pos    = fr_pos1
        self.enemy_pos = en_pos1

        # ---------- 云台与可视化 ----------
        self._gimbal_control()
        self._refresh_enemy_cache()
        if getattr(self.cfg, "traj_vis_enable", False):
            self._vis_helper.update_traj_vis()
        if getattr(self.cfg, "bearing_vis_enable", False):
            self._vis_helper.update_bearing_vis()

    # ippo版本
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        N, M, E = self.num_envs, self.M, self.E
        dev = self.device
        dtype = self.fr_pos.dtype

        # --- 权重 ---
        centroid_weight  = float(getattr(self.cfg, "centroid_approach_weight", 1.0))
        hit_weight       = float(getattr(self.cfg, "hit_reward_weight", 100.0))
        fb_weight        = float(getattr(self.cfg, "w_gimbal_friend_block", 0.1))
        explore_weight        = float(getattr(self.cfg, "w_explore", 0.1))
        R0               = float(getattr(self.cfg, "vc_zero_inside", 10.0))   # 近距离屏蔽半径
        mission_success_weight = float(getattr(self.cfg, "mission_success_weight", 10.0))
        leak_margin         = float(getattr(self.cfg, "leak_margin", 1.0))          # 漏敌机轴向裕度
        friend_collision_radius = float(getattr(self.cfg, "friend_collision_radius", 0.5))         # 友机-友机虚拟球避障惩罚
        friend_collision_penalty_weight = float(getattr(self.cfg, "friend_collision_penalty_weight", 0.5))
        target_guide_weight = float(getattr(self.cfg, "target_guide_weight", 0.0))
        coverage_reward_weight = float(getattr(self.cfg, "coverage_reward_weight", 0.005))  # 覆盖率奖励权重
        conflict_penalty_weight = float(getattr(self.cfg, "conflict_penalty_weight", 0.02))  # 冲突惩罚权重
        enemy_reach_goal_weight = float(getattr(self.cfg, "enemy_reach_goal_penalty_weight", 10.0))
        leak_penalty_weight = float(getattr(self.cfg, "leak_penalty_weight", 0.01))  # 漏敌机惩罚权重

        # 团队奖励相关权重
        team_reward_enabled = bool(getattr(self.cfg, "team_reward_enabled", False))
        team_success_weight = float(getattr(self.cfg, "team_success_weight", 100.0))
        team_coverage_weight = float(getattr(self.cfg, "team_coverage_weight", 10.0))
        team_failure_weight = float(getattr(self.cfg, "team_failure_weight", 50.0))

        # --- 活跃掩码 / 质心 ---
        friend_active    = (~self.friend_frozen)                     # [N,M] bool
        enemy_active     = (~self.enemy_frozen)                      # [N,E] bool
        enemy_active_any = self._enemy_active_any                    # [N]   bool
        vis_fe = self._gimbal_enemy_visible_mask()                   # [N, M, E]
        vis_ff = self._gimbal_friend_visible_mask()                  # [N, M, M]

        # ———————————————————— 友机-友机虚拟球避障惩罚（基于中心距离） ————————————————————
        collision_penalty_each = torch.zeros((N, M), device=dev, dtype=dtype)
        if M > 1 and friend_collision_penalty_weight != 0.0 and friend_collision_radius > 0.0:
            p_i = self.fr_pos.unsqueeze(2)  # [N,M,1,3]
            p_j = self.fr_pos.unsqueeze(1)  # [N,1,M,3]
            dist_ff = torch.linalg.norm(p_i - p_j, dim=-1)  # [N,M,M]
            eye = torch.eye(M, device=dev, dtype=torch.bool).unsqueeze(0)  # [1,M,M]
            fa_i = friend_active.unsqueeze(2)
            fa_j = friend_active.unsqueeze(1)
    
            # 排除自身并只考虑活跃友机
            valid_mask = fa_i & fa_j & (~eye)  # [N,M,M]
    
            # 距离越近惩罚越大：使用 exp(-dist / radius) 形式的连续惩罚
            safe_dist_ff = dist_ff.clamp_min(1e-6)  # 避免除零
            collision_penalty_raw = torch.where(
                valid_mask,
                torch.exp(-safe_dist_ff / friend_collision_radius),  # 距离越近惩罚越大
                torch.zeros_like(dist_ff)
            )  # [N,M,M]
    
            # 对每个友机，累加与其他友机的碰撞惩罚
            collision_penalty_each = collision_penalty_raw.sum(dim=-1) * friend_active.float()  # [N,M]

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

        # ———————————————————— bearing接近奖励 ————————————————————
        target_guide_reward = torch.zeros((N, M), device=dev, dtype=dtype)
        if E > 0:
            # 1. 准备几何数据（仅用于 reward 端的几何关系，视作 privileged）
            rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)                       # [N,M,E,3]
            dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(1e-6)
            dir_all = rel_all / dist_all                                                           # 指向敌机的单位向量 [N,M,E,3]
            cam_dir = self._dir_from_yaw_pitch(self._gimbal_yaw, self._gimbal_pitch).unsqueeze(2)  # [N,M,1,3] 云台光轴方向

            # 2. 光轴与各 bearing 的夹角矩阵 [N, M, E]
            cos_ang = (cam_dir * dir_all).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)  # [N,M,E]
            angle_matrix = torch.acos(cos_ang)  # [N,M,E]

            # 只在“云台可见 & 敌机存活”的槽位上允许选择，其他位置角度视为极大
            vis_effective = vis_fe & enemy_active.unsqueeze(1)                 # [N,M,E]
            large_angle = torch.full_like(angle_matrix, 1e9)
            angle_valid = torch.where(vis_effective, angle_matrix, large_angle)

            # 3. 每个友机选一个"意向敌机索引" intent_idx: [N,M]
            intent_idx = angle_valid.argmin(dim=-1)                   # [N,M]
            intent_valid = vis_effective.any(dim=-1) & friend_active  # [N,M] 这架友机的"意向目标"是可信且应被纳入冲突统计/分配逻辑的

            # 4. 构造引导奖励：鼓励速度方向对准"意向"目标方向（配合覆盖率和冲突奖励）
            has_intent = intent_valid  # [N,M] 该友机有有效的意向目标
            safe_intent_indices = intent_idx.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)  # 获取意向目标的单位向量
            intent_dir = torch.gather(dir_all, 2, safe_intent_indices).squeeze(2)             # 意向目标方向 [N, M, 3]

            # 5. 计算奖励：我的速度方向 vs 意向目标方向
            v_norm = self.fr_vel_w / self.fr_vel_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            v_dot_target = (v_norm * intent_dir).sum(dim=-1).clamp_min(0.0)  # [N, M]  投影：越对准意向目标，分越高

            # 6. 最终引导奖励
            target_guide_reward = has_intent.float() * v_dot_target * friend_active.float()

        with torch.no_grad():
            if E > 0:
                # - intent_idx:    [N,M]，每个友机当前“想要追”的敌机索引
                # - intent_valid:  [N,M]，该意向是否有效（在 FOV 内且友机存活）
                enemy_idx_grid = torch.arange(E, device=dev).view(1, 1, E)        # [1,1,E]
                intent_expand = intent_idx.unsqueeze(-1)                          # [N,M,1]
                intent_match = (intent_expand == enemy_idx_grid) & intent_valid.unsqueeze(-1)  # [N,M,E]
                per_enemy_counts = intent_match.sum(dim=1).float()                # [N,E]

                # 存活敌机数量
                active_enemy = enemy_active                                       # [N,E] bool
                active_counts = active_enemy.sum(dim=1).clamp_min(1).float()      # [N]

                # 覆盖：至少被一架友机看到的敌机
                visible_enemy = (vis_fe.any(dim=1)) & active_enemy                # [N,E] 至少被一架友机看到的存活敌机
                covered_counts = visible_enemy.float().sum(dim=1)                 # [N] 被覆盖的敌机数量
                CoverageRate = covered_counts / active_counts                         # [N] 覆盖率 = 被看到的敌机数 / 存活敌机数

                # 冲突：被2架及以上友机同时作为意向目标的存活敌机
                conflict_enemy = (per_enemy_counts >= 2.0) & active_enemy         # [N,E]
                conflict_counts = conflict_enemy.float().sum(dim=1)               # [N]
                conflict_rate = conflict_counts / active_counts                   # [N]

                # -------------------- 差分奖励：基于意向目标的一对一分配 --------------------
                # 团队分配分数：G = w_u*unique - w_c*conflict - w_0*uncovered
                # 差分奖励：D_i = G - G^{(-i)}，其中 G^{(-i)} 为移除友机 i 的反事实分数
                counts_e = intent_match.sum(dim=1).to(torch.long)                                       # [N,E]这个敌机当前被多少个友机意向
                active_e = enemy_active.to(torch.bool)                                                  # [N,E]
                active_cnt = active_e.sum(dim=1).clamp_min(1).to(dtype)                                 # [N]

                unique = ((counts_e == 1) & active_e).to(dtype).sum(dim=1) / active_cnt                 # env n中“一对一”分配成功的敌机比例
                conflict = ((counts_e >= 2) & active_e).to(dtype).sum(dim=1) / active_cnt               # env n中被 2 架及以上友机同时抢的敌机比例
                uncovered = ((counts_e == 0) & active_e).to(dtype).sum(dim=1) / active_cnt              # env n中没人选它（没有任何友机 intent）的敌机比例

                w_u = float(getattr(self.cfg, "diff_unique_w", 1.0))
                w_c = float(getattr(self.cfg, "diff_conflict_w", 1.0))
                w_0 = float(getattr(self.cfg, "diff_uncovered_w", 1.0))

                G_assign = w_u * unique - w_c * conflict - w_0 * uncovered                               # [N] 团队分配质量分

                # 如果拿掉友机 i，团队的 unique/conflict/uncovered 比例会是多少
                counts_woi = counts_e.unsqueeze(1) - intent_match.to(torch.long)                         # [N,M,E]
                unique_woi = ((counts_woi == 1) & active_e.unsqueeze(1)).to(dtype).sum(dim=2) / active_cnt.unsqueeze(1)
                conflict_woi = ((counts_woi >= 2) & active_e.unsqueeze(1)).to(dtype).sum(dim=2) / active_cnt.unsqueeze(1)
                uncovered_woi = ((counts_woi == 0) & active_e.unsqueeze(1)).to(dtype).sum(dim=2) / active_cnt.unsqueeze(1)

                G_woi = w_u * unique_woi - w_c * conflict_woi - w_0 * uncovered_woi                       # [N,M]

                # 覆盖率奖励
                coverage_reward_each = (G_assign.unsqueeze(1) - G_woi) * friend_active.to(dtype)          # [N,M]

                if conflict_penalty_weight != 0.0:
                    # 计算每个友机对每个敌机的综合评分（距离 + 角度）
                    fr_pos_expanded = self.fr_pos.unsqueeze(2)  # [N,M,1,3]
                    enemy_pos_expanded = self.enemy_pos.unsqueeze(1)  # [N,1,E,3]
                    distances = torch.norm(fr_pos_expanded - enemy_pos_expanded, dim=-1)  # [N,M,E]

                    # 距离评分：距离越近评分越高
                    dist_score = torch.exp(-distances / 50.0)  # [N,M,E] 50m为特征距离

                    # 角度评分：角度越小评分越高
                    angle_score = torch.exp(-angle_matrix / (torch.pi / 6))  # [N,M,E] π/6 ≈ 30°为特征角度

                    # 综合评分：距离和角度的加权平均
                    w_dist, w_angle = 0.6, 0.4  # 权重
                    composite_score = w_dist * dist_score + w_angle * angle_score  # [N,M,E]

                    # 只对有意向的友机-敌机对评分有效
                    composite_score = torch.where(intent_match, composite_score, torch.zeros_like(composite_score))

                    # 对于每个有冲突的敌机，在竞争该敌机的友机中比较评分
                    # intent_match: [N,M,E] 哪些友机选择了哪个敌机
                    # composite_score: [N,M,E] 每个友机对每个敌机的评分

                    # 对于每个敌机，找到其竞争友机中的最高评分
                    NEG_INF = torch.tensor(float('-inf'), device=dev, dtype=composite_score.dtype)
                    masked_scores = torch.where(intent_match, composite_score, NEG_INF)  # [N,M,E]
                    max_scores_per_enemy = masked_scores.max(dim=1).values  # [N,E] 每个敌机在每个环境中的最高评分

                    # 找到需要惩罚的友机：选择了敌机且评分低于该敌机的最高分
                    # 扩展max_scores用于广播比较
                    max_scores_expanded = max_scores_per_enemy.unsqueeze(1)  # [N,1,E]
                    lower_than_max = (composite_score < max_scores_expanded - 1e-6)  # [N,M,E]

                    # intent_match & lower_than_max 已经包含了要惩罚的友机
                    should_penalize = intent_match & lower_than_max  # [N,M,E]

                    # 累加所有敌机的惩罚（一个友机可能因为多个敌机而受到惩罚）
                    conflict_penalty_each = should_penalize.any(dim=2).float()  # [N,M]
                    conflict_penalty_each = conflict_penalty_each * friend_active.float()  # [N,M]

                # 累积到 episode 级指标缓存（后续在 reset 时做平均）
                self._metric_assign_coverage_sum += CoverageRate
                self._metric_conflict_sum += conflict_rate
                self._metric_steps += 1.0

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

        # ———————————————————— 探索鼓励 ————————————————————
        if E > 0:
            # per-enemy: whether any friend currently intends this enemy
            per_enemy_intended = intent_match.any(dim=1)  # [N,E] 该敌机被至少一个友机作为意向目标
            visible_enemy_cnt = vis_fe.sum(dim=-1)  # [N,M] 友机当前能看到的敌机总数

            # per-friend: whether current intent target is valid (intends and that enemy is active)
            has_intent = intent_valid  # [N,M] 该友机有有效的意向目标
            safe_intent = intent_idx.clamp(min=0)
            intent_enemy_active = enemy_active.gather(1, safe_intent) & has_intent
            no_valid_intent = (~has_intent) | (~intent_enemy_active)  # [N,M] 该友机没有有效意向或意向目标已失效

            # per-friend: whether this friend sees at least one unintended enemy
            unintended_enemy_visible = (~per_enemy_intended) & enemy_active  # [N,E] 未被有意向且存活的敌机
            sees_unintended_enemy = (vis_fe & unintended_enemy_visible.unsqueeze(1)).any(dim=2)  # [N,M] 该友机是否看到了至少一个未被有意向的敌机
        else:
            visible_enemy_cnt = torch.zeros((N, M), dtype=torch.long, device=dev)
            no_valid_intent = torch.ones((N, M), dtype=torch.bool, device=dev)
            sees_unintended_enemy = torch.zeros((N, M), dtype=torch.bool, device=dev)

        # 探索奖励：两个条件的与关系 - 自己当前没有意向目标并发现了环境中未被有意向的敌机
        explore_condition = (no_valid_intent & sees_unintended_enemy)  # [N,M]
        explore_reward_each = (explore_condition.float() * friend_active.float()).clamp_max(1.0)


        # ———————————————————— 敌人质心抵达目标点惩罚 ————————————————————
        cen = self._enemy_centroid                                                                # [N,3]
        diff_c = cen[..., :2] - self._goal_e[..., :2]                                             # [N,2]
        dist2_c = diff_c.square().sum(dim=-1)                                                     # [N]
        enemy_goal_any = dist2_c < (float(self.cfg.enemy_goal_radius) ** 2)                       # [N] bool
        enemy_reach_goal_any = enemy_goal_any.float().unsqueeze(1) * friend_active.float() 

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

        # --- 计算团队奖励 ---
        team_reward = torch.zeros((N, M), device=dev, dtype=dtype)
        if team_reward_enabled:
            # 团队成功奖励：全歼敌机
            team_success_reward = mission_success * team_success_weight  # [N,1]

            # 团队覆盖奖励：基于整体覆盖率
            if E > 0:
                team_coverage_reward = CoverageRate * team_coverage_weight  # [N]
            else:
                team_coverage_reward = torch.zeros(N, device=dev, dtype=dtype)

            # 团队失败惩罚：敌机抵达目标
            team_failure_penalty = enemy_goal_any.float() * team_failure_weight  # [N]

            # 总团队奖励 = 成功奖励 + 覆盖奖励 - 失败惩罚
            team_reward_total = team_success_reward.squeeze(1) + team_coverage_reward - team_failure_penalty
            team_reward = team_reward_total.unsqueeze(1).expand(-1, M)  # [N,M] 所有智能体获得相同奖励
        else:
            # --- 合成 per-agent reward ---
            # 质心接近
            team_reward = centroid_weight * centroid_each                                  # [N,M]
            # 基于 bearing 意向分配的引导奖励
            team_reward = team_reward + target_guide_weight * target_guide_reward             # [N,M]
            # 拦截奖励
            team_reward = team_reward + hit_weight * per_agent_hit                              # [N,M]
            # 云台项
            team_reward = team_reward - fb_weight * penalty_friend_each                          # [N,M]
            # 探索奖励
            team_reward = team_reward + explore_weight * explore_reward_each
            # overshoot
            team_reward = team_reward - leak_penalty_weight * leak_each
            # 敌人抵达目标点惩罚（均摊到每个友机）
            team_reward = team_reward - enemy_reach_goal_weight * enemy_reach_goal_any          # [N,M]
            # 全部歼灭奖励
            team_reward = team_reward + mission_success * mission_success_weight  # [N,M]
            # 友机-友机虚拟球避障惩罚
            team_reward = team_reward - friend_collision_penalty_weight * collision_penalty_each
            # 覆盖率奖励
            team_reward = team_reward + coverage_reward_weight * coverage_reward_each
            # 冲突惩罚
            team_reward = team_reward - conflict_penalty_weight * conflict_penalty_each

        r_each = team_reward

        # --- 写出字典 ---
        rewards = {agent: r_each[:, i] for i, agent in enumerate(self.possible_agents)}
        # --- 状态缓存/一次性标志 ---
        self.prev_dist_centroid = dist_to_centroid_now
        self._newly_frozen_enemy[:]  = False
        self._newly_frozen_friend[:] = False
        self._prev_available_visible_cnt = visible_enemy_cnt

        # ========================== 1. 组装奖励字典 ==========================
        # 这里我们将“权重 * 原始项”直接存入字典，这就是 Agent 实际收到的奖励
        reward_terms = {
            "centroid":         centroid_weight * centroid_each,
            "hit":              hit_weight * per_agent_hit,
            "gimbal_friend":    -fb_weight * penalty_friend_each,
            "explore_enemy":     explore_weight * explore_reward_each,
            # "mission_failure":     -mission_failure_penalty,
            "overshoot":        -leak_penalty_weight * leak_each,
            "enemy_reach_goal": -enemy_reach_goal_weight * enemy_reach_goal_any,
            "all_killed":       mission_success_weight * mission_success,
            "target_guide":     target_guide_weight * target_guide_reward,
            "friend_collision": -friend_collision_penalty_weight * collision_penalty_each,
            "coverage":         coverage_reward_weight * coverage_reward_each,
            "conflict":         -conflict_penalty_weight * conflict_penalty_each,
        }

        # ========================== 2. 计算总奖励 ==========================
        # 将所有分量叠加
        r_total = sum(reward_terms.values())

        # ========================== 3. 累积 Episode Sums ==========================
        if "centroid" not in self.episode_sums:
            for k, v in reward_terms.items():
                if k == "all_killed":
                    # all_killed 单独处理为计数器
                    continue
                # 处理一下维度广播的问题，确保 buffer 是 [N, M]
                if v.ndim < r_total.ndim:
                    self.episode_sums[k] = torch.zeros_like(r_total)
                else:
                    self.episode_sums[k] = torch.zeros_like(v)
            # 初始化 all_killed 计数器
            self.episode_sums["all_killed_count"] = torch.zeros(N, device=dev, dtype=torch.long)

        # 执行累加（不累加 total，all_killed 改为计数）
        for k, v in reward_terms.items():
            if k == "all_killed":
                # all_killed 项只计数出现次数
                continue
            # 确保维度匹配
            if v.shape != self.episode_sums[k].shape:
                v = v.expand_as(self.episode_sums[k])
            self.episode_sums[k] += v

        # 单独处理 all_killed：计数而非累积奖励
        all_killed_happened = mission_success.squeeze(-1) > 0  # [N]
        self.episode_sums["all_killed_count"] += all_killed_happened.long()

        return rewards

    # mappo
    # def _get_rewards(self) -> dict[str, torch.Tensor]:
    #     N, M, E = self.num_envs, self.M, self.E
    #     dev = self.device
    #     dtype = self.fr_pos.dtype

    #     # =========================
    #     # 0) 旧权重（local shaping 继续沿用）
    #     # =========================
    #     centroid_weight  = float(getattr(self.cfg, "centroid_approach_weight", 1.0))
    #     fb_weight        = float(getattr(self.cfg, "w_gimbal_friend_block", 0.1))
    #     ec_weight        = float(getattr(self.cfg, "w_explore", 0.1))
    #     R0               = float(getattr(self.cfg, "vc_zero_inside", 10.0))
    #     friend_collision_radius = float(getattr(self.cfg, "friend_collision_radius", 0.5))
    #     friend_collision_penalty_weight = float(getattr(self.cfg, "friend_collision_penalty_weight", 0.5))
    #     target_switch_penalty_weight = float(getattr(self.cfg, "target_switch_penalty_weight", 0.0))  # 只统计，默认不扣分
    #     target_guide_weight = float(getattr(self.cfg, "target_guide_weight", 0.0))
    #     leak_penalty_weight = float(getattr(self.cfg, "leak_penalty_weight", 0.005))
    #     leak_margin         = float(getattr(self.cfg, "leak_margin", 1.0))
    #     conflict_penalty_weight = float(getattr(self.cfg, "conflict_penalty_weight", 0.05))  # local：只惩罚“抢同一目标且不是最近的那架”

    #     # =========================
    #     # 1) MAPPO Team/Local 混合权重（新增）
    #     # =========================
    #     alpha = float(getattr(self.cfg, "team_reward_mix", 0.75))  # 越大越“MAPPO”
    #     invM = 1.0 / max(1.0, float(M))

    #     # team：任务成败/进度/分配质量（共享）
    #     w_team_hit     = float(getattr(self.cfg, "w_team_hit",     15.0))
    #     w_team_success = float(getattr(self.cfg, "w_team_success", 20.0))
    #     w_team_fail    = float(getattr(self.cfg, "w_team_fail",    10.0))
    #     w_team_leak    = float(getattr(self.cfg, "w_team_leak",    0.005))
    #     w_team_cov     = float(getattr(self.cfg, "w_team_cov",     0.005))
    #     w_team_conf    = float(getattr(self.cfg, "w_team_conf",    0.002))
    #     w_team_step    = float(getattr(self.cfg, "w_team_step",    0.001))

    #     # local：给“命中者”一点点额外 credit（可选）
    #     w_local_hit_bonus = float(getattr(self.cfg, "w_local_hit_bonus", 1.0))

    #     # =========================
    #     # 2) 掩码 / 可见性
    #     # =========================
    #     friend_active    = (~self.friend_frozen)                 # [N,M] bool（命中后会立刻变 False）
    #     enemy_active     = (~self.enemy_frozen)                  # [N,E] bool
    #     enemy_active_any = self._enemy_active_any                # [N]   bool
    #     vis_fe = self._gimbal_enemy_visible_mask()               # [N,M,E]
    #     vis_ff = self._gimbal_friend_visible_mask()              # [N,M,M]
    #     friend_active_f = friend_active.float()
    #     friend_eligible = friend_active | self._newly_frozen_friend          # [N,M] bool
    #     friend_eligible_f = friend_eligible.float()

    #     # =========================
    #     # 3) 友机-友机避碰（local）- 距离越近惩罚越大
    #     # =========================
    #     collision_penalty_each = torch.zeros((N, M), device=dev, dtype=dtype)
    #     if M > 1 and friend_collision_penalty_weight != 0.0 and friend_collision_radius > 0.0:
    #         p_i = self.fr_pos.unsqueeze(2)  # [N,M,1,3]
    #         p_j = self.fr_pos.unsqueeze(1)  # [N,1,M,3]
    #         dist_ff = torch.linalg.norm(p_i - p_j, dim=-1)  # [N,M,M]
    #         eye = torch.eye(M, device=dev, dtype=torch.bool).unsqueeze(0)  # [1,M,M]
    #         fa_i = friend_active.unsqueeze(2)
    #         fa_j = friend_active.unsqueeze(1)

    #         valid_mask = fa_i & fa_j & (~eye)  # [N,M,M] 排除自身并只考虑活跃友机

    #         # 距离越近惩罚越大：使用 exp(-dist / radius) 形式的连续惩罚
    #         safe_dist_ff = dist_ff.clamp_min(1e-6)  # 避免除零
    #         collision_penalty_raw = torch.where(
    #             valid_mask,
    #             torch.exp(-safe_dist_ff / friend_collision_radius),  # 距离越近惩罚越大
    #             torch.zeros_like(dist_ff)
    #         )  # [N,M,M]

    #         # 对每个友机，累加与其他友机的碰撞惩罚
    #         collision_penalty_each = collision_penalty_raw.sum(dim=-1) * friend_active_f  # [N,M]

    #     # =========================
    #     # 4) 质心接近增量（local）
    #     # =========================
    #     centroid = self._enemy_centroid.unsqueeze(1).expand(N, M, 3)             # [N,M,3]
    #     diff = centroid - self.fr_pos                                            # [N,M,3]
    #     dist_now = torch.linalg.norm(diff, dim=-1)                               # [N,M]
    #     dist_to_centroid_now = torch.where(
    #         enemy_active_any.unsqueeze(1),
    #         dist_now,
    #         self.prev_dist_centroid
    #     )
    #     delta_dist = self.prev_dist_centroid - dist_to_centroid_now              # [N,M]

    #     has_lock = vis_fe.any(dim=-1)                                            # [N,M]
    #     gate_lock = (~has_lock).float()
    #     gate_far = (dist_to_centroid_now > R0).float()
    #     centroid_each = delta_dist * gate_lock * gate_far * friend_active_f      # [N,M]

    #     # =========================
    #     # 5) intent + guide + coverage/conflict（team/local）
    #     # =========================
    #     target_guide_reward = torch.zeros((N, M), device=dev, dtype=dtype)
    #     target_switch_penalty_each = torch.zeros((N, M), device=dev, dtype=dtype)
    #     coverage = torch.zeros((N,), device=dev, dtype=dtype)
    #     conflict_rate = torch.zeros((N,), device=dev, dtype=dtype)
    #     conflict_penalty_each = torch.zeros((N, M), device=dev, dtype=dtype)

    #     intent_idx = torch.zeros((N, M), device=dev, dtype=torch.long)
    #     intent_valid = torch.zeros((N, M), device=dev, dtype=torch.bool)
    #     intent_match = None
    #     visible_enemy_cnt = torch.zeros((N, M), device=dev, dtype=torch.long)

    #     if E > 0:
    #         with torch.no_grad():
    #             rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)     # [N,M,E,3]
    #             dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(1e-6)
    #             dir_all = rel_all / dist_all                                         # [N,M,E,3]

    #             cam_dir = self._dir_from_yaw_pitch(self._gimbal_yaw, self._gimbal_pitch).unsqueeze(2)  # [N,M,1,3]
    #             cos_ang = (cam_dir * dir_all).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)              # [N,M,E]
    #             angle_matrix = torch.acos(cos_ang)                                                       # [N,M,E]

    #             vis_effective = vis_fe & enemy_active.unsqueeze(1)
    #             large_angle = torch.full_like(angle_matrix, 1e9)
    #             angle_valid = torch.where(vis_effective, angle_matrix, large_angle)

    #             intent_idx = angle_valid.argmin(dim=-1)                                  # [N,M]
    #             intent_valid = vis_effective.any(dim=-1) & friend_active                 # [N,M]

    #             if target_switch_penalty_weight != 0.0:
    #                 if not hasattr(self, "_last_intent_targets"):
    #                     self._last_intent_targets = torch.full((N, M), -1, dtype=torch.long, device=dev)

    #                 prev_intent = self._last_intent_targets
    #                 safe_prev = prev_intent.clamp(min=0)
    #                 prev_still_valid = enemy_active.gather(1, safe_prev)
    #                 prev_valid_mask = (prev_intent >= 0) & prev_still_valid
    #                 prev_intent = torch.where(prev_valid_mask, prev_intent, torch.full_like(prev_intent, -1))

    #                 has_prev = prev_intent >= 0
    #                 has_curr = intent_valid
    #                 switched = has_prev & has_curr & (intent_idx != prev_intent)

    #                 target_switch_penalty_each = torch.zeros((N, M), device=dev, dtype=dtype)
    #                 self._metric_switch_sum += switched.float()

    #                 new_memory = torch.where(has_curr, intent_idx, torch.full_like(intent_idx, -1))
    #                 self._last_intent_targets = new_memory

    #             has_intent = intent_valid
    #             safe_intent_indices = intent_idx.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    #             intent_dir = torch.gather(dir_all, 2, safe_intent_indices).squeeze(2)  # [N,M,3]

    #             v_norm = self.fr_vel_w / self.fr_vel_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    #             v_dot_target = (v_norm * intent_dir).sum(dim=-1).clamp_min(0.0)        # [N,M]
    #             target_guide_reward = has_intent.float() * v_dot_target * friend_active_f

    #             enemy_idx_grid = torch.arange(E, device=dev).view(1, 1, E)
    #             intent_expand = intent_idx.unsqueeze(-1)
    #             intent_match = (intent_expand == enemy_idx_grid) & intent_valid.unsqueeze(-1)

    #             per_enemy_counts = intent_match.sum(dim=1).float()                      # [N,E]
    #             active_enemy = enemy_active
    #             active_counts = active_enemy.sum(dim=1).clamp_min(1).float()            # [N]

    #             assigned_enemy = (per_enemy_counts > 0) & active_enemy
    #             covered_counts = assigned_enemy.float().sum(dim=1)                      # [N]
    #             coverage = covered_counts / active_counts

    #             conflict_enemy = (per_enemy_counts >= 2.0) & active_enemy
    #             conflict_counts = conflict_enemy.float().sum(dim=1)                     # [N]
    #             conflict_rate = conflict_counts / active_counts

    #             if conflict_penalty_weight != 0.0:
    #                 fr_pos_expanded = self.fr_pos.unsqueeze(2)          # [N,M,1,3]
    #                 enemy_pos_expanded = self.enemy_pos.unsqueeze(1)    # [N,1,E,3]
    #                 distances = torch.norm(fr_pos_expanded - enemy_pos_expanded, dim=-1)  # [N,M,E]

    #                 intent_match_f = intent_match
    #                 INF = torch.tensor(float("inf"), device=dev, dtype=dtype)
    #                 masked_distances = torch.where(intent_match_f, distances, INF)          # [N,M,E]
    #                 min_distances = masked_distances.min(dim=1, keepdim=True).values        # [N,1,E]

    #                 is_conflict_and_not_closest = (
    #                     intent_match_f
    #                     & conflict_enemy.unsqueeze(1)
    #                     & (distances > (min_distances + 1e-6))
    #                 )  # [N,M,E]
    #                 conflict_penalty_each = is_conflict_and_not_closest.any(dim=2).float() * friend_active_f

    #             self._metric_assign_coverage_sum += coverage
    #             self._metric_conflict_sum += conflict_rate
    #             self._metric_steps += 1.0

    #             visible_enemy_cnt = vis_fe.sum(dim=-1).to(torch.long)  # [N,M]

    #     # =========================
    #     # 6) 命中（本步事件）
    #     # =========================
    #     per_agent_hit = self._newly_frozen_friend.float()  # [N,M]

    #     # =========================
    #     # 7) 全歼
    #     # =========================
    #     enemy_exists = self._enemy_exists_mask
    #     mission_success = ((~enemy_exists) | self.enemy_frozen).all(dim=1, keepdim=True).float()  # [N,1]

    #     # =========================
    #     # 8) 友机遮挡惩罚（local）
    #     # =========================
    #     alive_friend_nums = friend_active.sum(dim=1).to(dtype)                      # [N]
    #     alive_friend_nums_ = (alive_friend_nums - 1.0).clamp_min(1.0).unsqueeze(1)  # [N,1]
    #     pen_friend_each_cnt = vis_ff.float().sum(dim=2)                             # [N,M]
    #     penalty_friend_each = (pen_friend_each_cnt / alive_friend_nums_) * friend_active_f

    #     # =========================
    #     # 9) 探索奖励（local）
    #     # =========================
    #     explore_reward_each = torch.zeros((N, M), device=dev, dtype=dtype)
    #     if E > 0 and intent_match is not None:
    #         per_enemy_intended = intent_match.any(dim=1)                       # [N,E]
    #         has_intent = intent_valid                                          # [N,M]
    #         safe_intent = intent_idx.clamp(min=0)                              # [N,M]
    #         intent_enemy_active = enemy_active.gather(1, safe_intent) & has_intent
    #         no_valid_intent = (~has_intent) | (~intent_enemy_active)           # [N,M]

    #         unintended_enemy = enemy_active & (~per_enemy_intended)            # [N,E]
    #         sees_unintended_enemy = (vis_fe & unintended_enemy.unsqueeze(1)).any(dim=-1)  # [N,M]

    #         explore_condition = (no_valid_intent & sees_unintended_enemy)
    #         explore_reward_each = (explore_condition.float() * friend_active_f).clamp_max(1.0)

    #     # =========================
    #     # 10) 敌方到点惩罚（team）
    #     # =========================
    #     cen = self._enemy_centroid
    #     diff_c = cen[..., :2] - self._goal_e[..., :2]
    #     dist2_c = diff_c.square().sum(dim=-1)
    #     mission_failed = dist2_c < (float(self.cfg.enemy_goal_radius) ** 2)      # [N] bool

    #     # =========================
    #     # 11) 漏防/越位（team）
    #     # =========================
    #     num_leaked = torch.zeros((N,), device=dev, dtype=dtype)
    #     if leak_penalty_weight != 0.0 and E > 0:
    #         gk_3d = self._goal_e      # [N,3]
    #         axis_3d = self._axis_hat  # [N,3]

    #         axis_xy = axis_3d[..., :2]
    #         norm_xy = torch.linalg.norm(axis_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    #         axis_hat = torch.cat([axis_xy / norm_xy, torch.zeros_like(axis_3d[..., 2:3])], dim=-1)  # [N,3]

    #         sf = ((self.fr_pos    - gk_3d.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [N,M]
    #         se = ((self.enemy_pos - gk_3d.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [N,E]

    #         INF = torch.tensor(float("inf"), dtype=dtype, device=dev)
    #         NEG_INF = torch.tensor(float("-inf"), dtype=dtype, device=dev)
    #         sf_mask = torch.where(friend_active, sf, INF)
    #         se_mask = torch.where(enemy_active,  se, NEG_INF)

    #         friend_min = sf_mask.min(dim=1, keepdim=True).values  # [N,1]
    #         leaked_enemy = enemy_active & (se_mask < (friend_min - leak_margin))  # [N,E]
    #         num_leaked = leaked_enemy.float().sum(dim=1)  # [N]

    #     # =========================
    #     # 12) MAPPO：team reward（每 env 一个标量）
    #     # =========================
    #     friend_any = (friend_eligible.sum(dim=1) > 0).float()  # [N] 这里用 eligible 更合理

    #     alive_enemy_cnt = (self._enemy_exists_mask & (~self.enemy_frozen)).float().sum(dim=1).clamp_min(1.0)  # [N]
    #     new_kill = self._newly_frozen_enemy.float().sum(dim=1)  # [N]

    #     r_team_hit = w_team_hit * (new_kill / alive_enemy_cnt)                        # [N]
    #     r_team_success = w_team_success * mission_success.squeeze(-1)                  # [N]
    #     r_team_fail = -w_team_fail * mission_failed.float()                            # [N]
    #     r_team_leak = -w_team_leak * num_leaked                                        # [N]
    #     r_team_alloc = (w_team_cov * coverage) - (w_team_conf * conflict_rate)         # [N]
    #     r_team_time = -w_team_step * torch.ones_like(r_team_hit)                       # [N]

    #     r_team = (r_team_hit + r_team_success + r_team_fail + r_team_leak + r_team_alloc + r_team_time) * friend_any  # [N]
    #     r_team_each = r_team.unsqueeze(1) * friend_eligible_f  # [N,M]

    #     # =========================
    #     # 13) local shaping（每 agent）
    #     # =========================
    #     # 13.1 先算“只给存活者”的 shaping（避免冻结者继续刷 shaping）
    #     r_local_shaping = torch.zeros((N, M), device=dev, dtype=dtype)
    #     r_local_shaping += centroid_weight * centroid_each
    #     r_local_shaping += target_guide_weight * target_guide_reward
    #     r_local_shaping += ec_weight * explore_reward_each
    #     r_local_shaping -= fb_weight * penalty_friend_each
    #     r_local_shaping -= friend_collision_penalty_weight * collision_penalty_each
    #     r_local_shaping -= conflict_penalty_weight * conflict_penalty_each
    #     r_local_shaping = r_local_shaping * friend_active_f  # 只给活着的

    #     # 13.2 hit bonus 单独加：不再被 friend_active 抹掉（因为命中当步 friend_frozen 已经 True）
    #     r_hit_bonus_each = w_local_hit_bonus * per_agent_hit  # [N,M] 只有 newly_frozen_friend 才为 1

    #     r_local = r_local_shaping + r_hit_bonus_each

    #     # =========================
    #     # 14) 混合输出（MAPPO）
    #     # =========================
    #     r_each = alpha * r_team_each + (1.0 - alpha) * r_local   # [N,M]
    #     # rewards = {agent: r_each[:, i] for i, agent in enumerate(self.possible_agents)}
    #     rewards = {agent: r_each[:, i:i+1] for i, agent in enumerate(self.possible_agents)}

    #     # =========================
    #     # 15) 状态缓存/一次性标志
    #     # =========================
    #     self.prev_dist_centroid = dist_to_centroid_now
    #     self._newly_frozen_enemy[:] = False
    #     self._newly_frozen_friend[:] = False
    #     self._prev_available_visible_cnt = visible_enemy_cnt

    #     # =========================
    #     # 16) 统计（episode_sums）
    #     # =========================
    #     reward_terms = {
    #         # team：分发给 eligible（命中当步也有）
    #         "team_hit":      alpha * (r_team_hit.unsqueeze(1) * friend_eligible_f),
    #         "team_success":  alpha * (r_team_success.unsqueeze(1) * friend_eligible_f),
    #         "team_fail":     alpha * (r_team_fail.unsqueeze(1) * friend_eligible_f),
    #         "team_leak":     alpha * (r_team_leak.unsqueeze(1) * friend_eligible_f),
    #         "team_alloc":    alpha * (r_team_alloc.unsqueeze(1) * friend_eligible_f),
    #         "team_time":     alpha * (r_team_time.unsqueeze(1) * friend_eligible_f),

    #         # local shaping：只给 active
    #         "centroid":      (1.0 - alpha) * (centroid_weight * centroid_each),
    #         "target_guide":  (1.0 - alpha) * (target_guide_weight * target_guide_reward),
    #         "explore_enemy": (1.0 - alpha) * (ec_weight * explore_reward_each),
    #         "gimbal_friend": (1.0 - alpha) * (-fb_weight * penalty_friend_each),
    #         "friend_collision": (1.0 - alpha) * (-friend_collision_penalty_weight * collision_penalty_each),
    #         "conflict_local":(1.0 - alpha) * (-conflict_penalty_weight * conflict_penalty_each),

    #         # hit_bonus：不再乘 active（否则命中当步会被抹掉）
    #         "hit_bonus":     (1.0 - alpha) * (w_local_hit_bonus * per_agent_hit),
    #         "all_killed":    mission_success,  # [N,1] 计数用
    #     }

    #     r_total = sum(v for k, v in reward_terms.items() if k != "all_killed")

    #     if "team_hit" not in self.episode_sums:
    #         for k, v in reward_terms.items():
    #             if k == "all_killed":
    #                 continue
    #             self.episode_sums[k] = torch.zeros_like(r_total)
    #         self.episode_sums["all_killed_count"] = torch.zeros(N, device=dev, dtype=torch.long)

    #     for k, v in reward_terms.items():
    #         if k == "all_killed":
    #             continue
    #         if v.shape != self.episode_sums[k].shape:
    #             v = v.expand_as(self.episode_sums[k])
    #         self.episode_sums[k] += v

    #     all_killed_happened = mission_success.squeeze(-1) > 0
    #     self.episode_sums["all_killed_count"] += all_killed_happened.long()

    #     return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        N         = self.num_envs
        M         = self.M
        device    = self.device
        r2_goal = float(self.cfg.enemy_goal_radius) ** 2
        xy_max2 = float(self.cfg.enemy_cluster_ring_radius + 50.0) ** 2
        # ---------- 基本终止判据（逐env） ----------
        enemy_exists = self._enemy_exists_mask                                    # [N,E]
        # “存在的敌机 都被冻结”才算成功；不存在的槽位自动视作满足
        success_all_enemies = ((~enemy_exists) | self.enemy_frozen).all(dim=1)   # [N]
        # success_all_enemies = self.enemy_frozen.all(dim=1)                        # [N] 敌全灭（友军成功），all函数用于判断输入中个张量是否都是True
        if success_all_enemies.any():
            print("all enemies destroied!!!!!!")

        z = self.fr_pos[:, :, 2]                                                  # 对每个环境、每个友机，都取坐标向量的索引 2（即第 3 个分量）也就是飞机的z高度
        z_enemy_max, _ = self.enemy_pos[:, :, 2].max(dim=1)                       # [N] 每个环境中敌机的最高高度
        z_enemy_max = z_enemy_max.unsqueeze(1)                                    # [N,1]
        out_z_any = ((z < 0.0) | (z > (z_enemy_max + 6.0))).any(dim=1)           # [N] Z 越界

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

        # 友机距离过近终止条件
        friend_collision_reset_threshold = float(getattr(self.cfg, "friend_collision_reset_threshold", 0.1))  # 默认0.5米
        friend_too_close_any = torch.zeros(N, dtype=torch.bool, device=device)  # [N]
        if M > 1:
            # 计算友机间最小距离
            p_i = self.fr_pos.unsqueeze(2)  # [N,M,1,3]
            p_j = self.fr_pos.unsqueeze(1)  # [N,1,M,3]
            dist_ff = torch.linalg.norm(p_i - p_j, dim=-1)  # [N,M,M]
            eye = torch.eye(M, device=device, dtype=torch.bool).unsqueeze(0)  # [1,M,M]
            # 排除自身，只考虑活跃友机
            friend_active = (~self.friend_frozen)  # [N,M]
            fa_i = friend_active.unsqueeze(2)
            fa_j = friend_active.unsqueeze(1)
            valid_dist_mask = fa_i & fa_j & (~eye)  # [N,M,M]
            # 将无效距离设为无穷大
            INF = torch.tensor(float('inf'), device=device, dtype=dist_ff.dtype)
            valid_dist = torch.where(valid_dist_mask, dist_ff, INF)
            # 找到每对友机间的最小距离
            min_dist_per_env = valid_dist.min(dim=-1).values.min(dim=-1).values  # [N]
            friend_too_close_any = min_dist_per_env < friend_collision_reset_threshold
            if friend_too_close_any.any():
                print("friend_too_close!!!!!!")

        overshoot_any  = torch.zeros(N, dtype=torch.bool, device=device)  # [N]
        alive_mask = ~(success_all_enemies | out_z_any | out_xy_any | nan_inf_any | enemy_goal_any)  # [N]
        idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)
        if alive_mask.any():
            tol = float(getattr(self.cfg, "overshoot_tol", 1.0))
            idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)          # [n]
            friend_active = (~self.friend_frozen[idx])                    # [n,M]
            enemy_exists  = self._enemy_exists_mask[idx]                  # [n,E]
            enemy_active  = enemy_exists & (~self.enemy_frozen[idx])      # [n,E]
            # enemy_active  = (~self.enemy_frozen[idx])                     # [n,E]
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
                separated  = friend_min >= (enemy_max + tol)
                overshoot_any[idx[k_idx]] = separated

        died     = out_z_any | out_xy_any | nan_inf_any | success_all_enemies | enemy_goal_any | overshoot_any | friend_too_close_any  # [N]
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
            cnt_friend_close   = take(friend_too_close_any)
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
                "friend_too_close"   : cnt_friend_close,
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
        # reset per-env previous-visibility buffer used for exploration discovery detection
        if not hasattr(self, "_prev_available_visible_cnt") or self._prev_available_visible_cnt is None:
            self._prev_available_visible_cnt = torch.zeros((self.num_envs, self.M), dtype=torch.long, device=self.device)
        else:
            self._prev_available_visible_cnt[env_ids] = 0

        # ========================== 打印各种统计信息 ==========================
        if getattr(self.cfg, "per_train_data_print", False):
            # === 打印上一次 episode 的终止原因 ===
            if hasattr(self, "extras") and isinstance(self.extras, dict) and "termination" in self.extras:
                term = self.extras["termination"]
                print("\n--- Episode Termination Summary ---")
                for k, v in term.items():
                    print(f"{k:<20}: {v}")
                print("-----------------------------------")
            # === 打印本次 reset 的 env 的 reward 各分量 episode 累积和 ===
            # 注意：此时 episode_sums 里存的是"上一段 episode"的累计值
            if len(self.episode_sums) > 0 and env_ids is not None and len(env_ids) > 0:
                print("Reward components (sum over episode, per env; sum over agents):")
                for name, buf in self.episode_sums.items():
                    if name == "all_killed_count":
                        # all_killed_count 是 [N] 形状，不需要对 agent 维求和
                        vals = buf[env_ids].float()  # [len(env_ids)]，转换为浮点数
                    else:
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
                    if name == "all_killed_count":
                        # all_killed_count 是 [N] 形状，是标量值
                        val = buf[env0].item()
                        print(f"  {name:<16}: {val}")
                    else:
                        # buf: [N, M]
                        row = buf[env0]  # [M]，这一 env 下每个 agent 的累计奖励
                        # 打印成列表好看一点
                        vals = row.detach().cpu().tolist()
                        # 如果 agent 太多，可以只打印前几个
                        # vals = vals[:10]
                        print(f"  {name:<16}: {vals}")
                print("---------------------------------------------------------")

            # === 打印拦截率（冻结敌机数 / 总敌机数） ===
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

        # ========================== LOGGING TO TENSORBOARD ==========================
        if not hasattr(self, "extras"):
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

        # 遍历所有累积的奖励分项
        if len(self.episode_sums) > 0:
            for k, v in self.episode_sums.items():
                if k == "all_killed_count":
                    # 单独处理 all_killed_count：显示为计数指标而非奖励值
                    metric_val = v[env_ids].float().mean()
                    self.extras["log"]["Episode_Metric/All_Killed_Count"] = metric_val
                else:
                    # 方式 A：平均每个智能体的平均奖励 (Average Reward per Agent)
                    # metric_val = v[env_ids].mean()

                    # 方式 B：整个集群的总奖励 (Total Swarm Reward)，所有重置环境中的所有友机各奖励分项的总和的平均，比如是所有agent的靠近质心奖励总和然后再除以重置的环境数求的平均
                    metric_val = v[env_ids].sum(dim=1).mean()

                    self.extras["log"][f"Episode_Reward/{k}"] = metric_val

                # 清零该环境的累积器
                self.episode_sums[k][env_ids] = 0.0

            # 拦截率统计 (Interception Rate)
            if self.E > 0:
                exists     = self._enemy_exists_mask[env_ids]          # [N_reset,E]
                frozen     = self.enemy_frozen[env_ids] & exists       # 只统计真实存在的敌机
                frozen_cnt = frozen.sum(dim=1)                         # [N_reset]
                total_per_env = exists.sum(dim=1).clamp_min(1)         # [N_reset]
                rate = frozen_cnt.float() / total_per_env.float()      # [N_reset]
                self.extras["log"]["Episode_Metric/Interception_Rate"] = rate.mean()

        # ========================== 隐式目标分配质量指标 ==========================
        # 这些指标完全基于环境内部统计，仅用于 TensorBoard 观测，不影响策略与梯度。
        if hasattr(self, "_metric_steps") and len(env_ids) > 0:
            steps = self._metric_steps[env_ids].clamp_min(1.0)  # [N_reset]

            # 1) 可见性覆盖率：每步"被至少一个友机看到的存活敌机占比"的时间平均
            cov_env = self._metric_assign_coverage_sum[env_ids] / steps  # [N_reset]
            self.extras["log"]["Episode_Metric/Assign_CoverageRate"] = cov_env.mean()

            # 2) 冲突率：每步“被>=2个友机同时分配的存活敌机占比”的时间平均
            conf_env = self._metric_conflict_sum[env_ids] / steps        # [N_reset]
            self.extras["log"]["Episode_Metric/Assign_ConflictRate"] = conf_env.mean()

            # 清零这些 env 的统计缓存，避免污染下一段 episode
            self._metric_steps[env_ids] = 0.0
            self._metric_assign_coverage_sum[env_ids] = 0.0
            self._metric_conflict_sum[env_ids] = 0.0

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

        # 重置协作 CV-EKF（几何观测 + 常速模型）
        if hasattr(self, "enemy_filter"):
            self.enemy_filter.reset(env_ids, init_pos=self.enemy_pos[env_ids])

        if hasattr(self, "_prev_phi_cover") and self._prev_phi_cover is not None:
            self._prev_phi_cover[env_ids] = 0.0

        if hasattr(self, "_prev_phi_local") and self._prev_phi_local is not None:
            self._prev_phi_local[env_ids, :] = 0.0

        if hasattr(self, "_mission_failed_latched"):
            self._mission_failed_latched[env_ids] = False

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
        
        # --- 纵向(X_local):沿f_hat反方向延伸(0, -5, -10...)
        x_local = - row_idxs * row_spacing

        # --- 横向 (Y_local): 从中间开始左右交替排列，保持对称性
        row_start = row_idxs * agents_per_row  # [M] 每架友机所在排的起始索引
        local_idx_in_row = agent_idxs - row_start  # [M] 在该排内的局部索引 (0,1,2,...)

        # 对于每架友机，计算它在对称排列中的位置
        # 偶数索引(0,2,4...): pos = local_idx // 2
        # 奇数索引(1,3,5...): pos = -(local_idx + 1) // 2
        is_even = (local_idx_in_row % 2 == 0)
        pos_indices = torch.where(
            is_even,
            local_idx_in_row // 2,  # 偶数：0, 1, 2, ...
            -(local_idx_in_row + 1) // 2  # 奇数：-1, -2, -3, ...
        ).to(dtype)

        # 转换为实际横向坐标（以中心为原点）
        y_local = pos_indices * lat_spacing

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

        # 友机数一对一匹配到敌机数（便捷开关）
        if getattr(self.cfg, "friend_follow_enemy_num", False):
            # 当前这些 env 的敌机数：shape [len(env_ids)]
            enemy_cnt = self._enemy_count[env_ids]               # long

            # 每个环境启用的友机数 = min(敌机数, 最大友机数)
            active_friend = torch.clamp(enemy_cnt, max=self.M)   # [len(env_ids)]

            # 构造 [len(env_ids), M] 的 index -> mask
            idx_f = torch.arange(self.M, device=self.device).unsqueeze(0)  # [1, M]
            active_mask = idx_f < active_friend.unsqueeze(1)               # [len(env_ids), M]

            # 启用的为 False（不冻结），多余的设为 True（冻结）
            self.friend_frozen[env_ids] = ~active_mask

    # 未加入距离友机最近的bearing信息的obs
    # def _get_observations(self) -> dict[str, torch.Tensor]:
    #     N, M, E = self.num_envs, self.M, self.E
    #     dev, dtype = self.device, self.fr_pos.dtype
    #     eps = 1e-9

    #     K_target  = self.cfg.obs_k_target
    #     K_friends = self.cfg.obs_k_friends

    #     # ====================== 1. 友机相对观测 (Top-K) ======================
    #     if M > 1:
    #         pos_i = self.fr_pos.unsqueeze(2)   # [N,M,1,3]
    #         pos_j = self.fr_pos.unsqueeze(1)   # [N,1,M,3]
    #         dist_ij_raw = torch.linalg.norm(pos_j - pos_i, dim=-1)  # [N,M,M]

    #         # 屏蔽自己和冻结友机
    #         large = torch.full_like(dist_ij_raw, 1e6)
    #         eye = torch.eye(M, device=dev, dtype=torch.bool).unsqueeze(0)
    #         friend_alive = (~self.friend_frozen)
    #         both_alive = friend_alive.unsqueeze(1) & friend_alive.unsqueeze(2)
    #         valid_pair = (~eye) & both_alive
    #         dist_ij = torch.where(valid_pair, dist_ij_raw, large)

    #         # 排序,返回沿给定维度按值升序排序张数的索引
    #         sorted_idx_all = dist_ij.argsort(dim=-1)

    #         # 拿取前K个友机的索引
    #         valid_k_fr = min(M - 1, K_friends)
    #         top_k_idx = sorted_idx_all[..., :valid_k_fr] # [N, M, valid_k_fr]

    #         # Gather 位置和速度
    #         gather_idx = top_k_idx.unsqueeze(-1).expand(-1, -1, -1, 3)

    #         closest_pos = torch.gather(self.fr_pos.unsqueeze(1).expand(N, M, M, 3), 2, gather_idx)
    #         closest_vel = torch.gather(self.fr_vel_w.unsqueeze(1).expand(N, M, M, 3), 2, gather_idx)

    #         # 转为相对量
    #         rel_pos = closest_pos - self.fr_pos.unsqueeze(2)
    #         rel_vel = closest_vel - self.fr_vel_w.unsqueeze(2)

    #         # Padding 到固定长度
    #         out_pos = torch.zeros(N, M, K_friends, 3, device=dev, dtype=dtype)
    #         out_vel = torch.zeros(N, M, K_friends, 3, device=dev, dtype=dtype)

    #         if valid_k_fr > 0:
    #             out_pos[:, :, :valid_k_fr, :] = rel_pos
    #             out_vel[:, :, :valid_k_fr, :] = rel_vel

    #         topk_pos_flat = out_pos.reshape(N, M, -1)
    #         topk_vel_flat = out_vel.reshape(N, M, -1)
    #     else:
    #         topk_pos_flat = torch.zeros(N, M, K_friends * 3, device=dev, dtype=dtype)
    #         topk_vel_flat = torch.zeros(N, M, K_friends * 3, device=dev, dtype=dtype)

    #     # ====================== 2. 敌机观测 (Top-K) ======================
    #     # 输出维度：K_target * 4
    #     if E > 0:
    #         vis_fe = self._gimbal_enemy_visible_mask()  # [N, M, E]

    #         rel_all  = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)
    #         dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)
    #         dir_all  = rel_all / dist_all

    #         cam_dir = self._dir_from_yaw_pitch(self._gimbal_yaw, self._gimbal_pitch).unsqueeze(2)
    #         cos_ang = (cam_dir * dir_all).sum(dim=-1).clamp(-1.0+1e-6, 1.0-1e-6)
    #         angle   = torch.acos(cos_ang)

    #         large_angle = 100.0
    #         angle_masked = torch.where(vis_fe, angle, torch.tensor(large_angle, device=dev))
    #         sorted_indices = angle_masked.argsort(dim=-1, descending=False)

    #         valid_k = min(E, K_target)
    #         top_k_idx = sorted_indices[..., :valid_k]

    #         gather_idx = top_k_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    #         best_vecs = torch.gather(dir_all, 2, gather_idx)
    #         best_angles = torch.gather(angle_masked, 2, top_k_idx)

    #         is_locked = (best_angles < large_angle)
    #         final_vecs = best_vecs * is_locked.unsqueeze(-1).float()
    #         final_locks = is_locked.unsqueeze(-1).float()

    #         target_obs_container = torch.zeros(N, M, K_target, 4, device=dev, dtype=dtype)
    #         valid_part = torch.cat([final_vecs, final_locks], dim=-1)
    #         target_obs_container[:, :, :valid_k, :] = valid_part
    #         target_feat_flat = target_obs_container.reshape(N, M, -1)
    #     else:
    #         target_feat_flat = torch.zeros((N, M, K_target * 4), device=dev, dtype=dtype)

    #     # ====================== 3. 自身状态 & ID ======================
    #     # self_pos_abs = self.fr_pos
    #     env_origins_expanded = self.terrain.env_origins.unsqueeze(1)  # [N, 1, 3]
    #     self_pos_abs = self.fr_pos - env_origins_expanded  # [N, M, 3] 相对位置
    #     self_vel_abs = self.fr_vel_w

    #     cen = self._enemy_centroid
    #     rel_c = cen.unsqueeze(1) - self.fr_pos
    #     dist_c = torch.linalg.norm(rel_c, dim=-1, keepdim=True).clamp_min(eps)
    #     e_hat_c = rel_c / dist_c

    #     agent_id_feat = self._agent_id_onehot.expand(N, -1, -1).to(dtype=dtype)

    #     # ====================== 4. 拼接 ======================
    #     obs_each = torch.cat(
    #         [
    #             topk_pos_flat,           # K_friends * 3
    #             topk_vel_flat,           # K_friends * 3
    #             self_pos_abs,            # 3
    #             self_vel_abs,            # 3
    #             # e_hat_c,                 # 3
    #             # dist_c,                  # 1
    #             target_feat_flat,        # K_target * 4
    #             # agent_id_feat,           # M
    #         ],
    #         dim=-1,
    #     )

    #     obs_dict = {ag: obs_each[:, i, :] for i, ag in enumerate(self.possible_agents)}

    #     return obs_dict

    def _get_observations(self) -> dict[str, torch.Tensor]:
        N, M, E = self.num_envs, self.M, self.E
        dev, dtype = self.device, self.fr_pos.dtype
        eps = 1e-9

        K_target  = self.cfg.obs_k_target
        K_friends = self.cfg.obs_k_friends

        # 先定义：活着的友机（后面 enemy 可见也建议用它 mask 一下）
        friend_alive = (~self.friend_frozen)  # [N,M]

        # ====================== 1. 友机相对观测 (Top-K) ======================
        top_k_friend_idx = None  # 关键：留着给后面 gather intent 用

        if M > 1:
            pos_i = self.fr_pos.unsqueeze(2)   # [N,M,1,3]
            pos_j = self.fr_pos.unsqueeze(1)   # [N,1,M,3]
            dist_ij_raw = torch.linalg.norm(pos_j - pos_i, dim=-1)  # [N,M,M]

            large = torch.full_like(dist_ij_raw, 1e6)
            eye = torch.eye(M, device=dev, dtype=torch.bool).unsqueeze(0)
            both_alive = friend_alive.unsqueeze(1) & friend_alive.unsqueeze(2)
            valid_pair = (~eye) & both_alive
            dist_ij = torch.where(valid_pair, dist_ij_raw, large)

            sorted_idx_all = dist_ij.argsort(dim=-1)

            valid_k_fr = min(M - 1, K_friends)
            top_k_friend_idx = sorted_idx_all[..., :valid_k_fr]  # [N, M, valid_k_fr]

            gather_idx3 = top_k_friend_idx.unsqueeze(-1).expand(-1, -1, -1, 3)

            closest_pos = torch.gather(self.fr_pos.unsqueeze(1).expand(N, M, M, 3), 2, gather_idx3)
            closest_vel = torch.gather(self.fr_vel_w.unsqueeze(1).expand(N, M, M, 3), 2, gather_idx3)

            rel_pos = closest_pos - self.fr_pos.unsqueeze(2)
            rel_vel = closest_vel - self.fr_vel_w.unsqueeze(2)

            out_pos = torch.zeros(N, M, K_friends, 3, device=dev, dtype=dtype)
            out_vel = torch.zeros(N, M, K_friends, 3, device=dev, dtype=dtype)
            if valid_k_fr > 0:
                out_pos[:, :, :valid_k_fr, :] = rel_pos
                out_vel[:, :, :valid_k_fr, :] = rel_vel

            topk_pos_flat = out_pos.reshape(N, M, -1)
            topk_vel_flat = out_vel.reshape(N, M, -1)
        else:
            valid_k_fr = 0
            topk_pos_flat = torch.zeros(N, M, K_friends * 3, device=dev, dtype=dtype)
            topk_vel_flat = torch.zeros(N, M, K_friends * 3, device=dev, dtype=dtype)

        # ====================== 2. 敌机观测 (Top-K) ======================
        # 输出维度：K_target * 4
        if E > 0:
            vis_fe = self._gimbal_enemy_visible_mask()  # [N, M, E]
            # 建议：冻结友机不应该“看到敌机”（否则 frozen agent 也会产生 intent）
            vis_fe = vis_fe & friend_alive.unsqueeze(-1)

            rel_all  = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)
            dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)
            dir_all  = rel_all / dist_all  # world LOS

            cam_dir = self._dir_from_yaw_pitch(self._gimbal_yaw, self._gimbal_pitch).unsqueeze(2)
            cos_ang = (cam_dir * dir_all).sum(dim=-1).clamp(-1.0+1e-6, 1.0-1e-6)
            angle   = torch.acos(cos_ang)

            large_angle = 100.0
            angle_masked = torch.where(vis_fe, angle, torch.tensor(large_angle, device=dev))
            sorted_enemy_idx = angle_masked.argsort(dim=-1, descending=False)

            valid_k_en = min(E, K_target)
            top_k_enemy_idx = sorted_enemy_idx[..., :valid_k_en]

            gather_idx3 = top_k_enemy_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            best_vecs   = torch.gather(dir_all, 2, gather_idx3)
            best_angles = torch.gather(angle_masked, 2, top_k_enemy_idx)

            is_locked   = (best_angles < large_angle)
            final_vecs  = best_vecs * is_locked.unsqueeze(-1).float()
            final_locks = is_locked.unsqueeze(-1).float()

            target_obs_container = torch.zeros(N, M, K_target, 4, device=dev, dtype=dtype)
            valid_part = torch.cat([final_vecs, final_locks], dim=-1)
            target_obs_container[:, :, :valid_k_en, :] = valid_part
            target_feat_flat = target_obs_container.reshape(N, M, -1)

            # ====================== 2.5 计算并“广播”队友 intent（Top-K friends） ======================
            # 自己的 intent：slot0（光轴最近）世界系方向 + lock
            self_intent_dir   = target_obs_container[:, :, 0, :3]          # [N,M,3]
            self_intent_valid = target_obs_container[:, :, 0, 3:4]         # [N,M,1] float(0/1)
            self_intent_valid = self_intent_valid * friend_alive.unsqueeze(-1).float()

            # gather 邻居队友的 intent
            out_intent = torch.zeros(N, M, K_friends, 4, device=dev, dtype=dtype)  # [N,M,Kf,4]
            if (top_k_friend_idx is not None) and (valid_k_fr > 0):
                g3 = top_k_friend_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # [N,M,k,3]
                g1 = top_k_friend_idx.unsqueeze(-1)                        # [N,M,k,1]

                intent_dir_all = self_intent_dir.unsqueeze(1).expand(N, M, M, 3)
                intent_val_all = self_intent_valid.unsqueeze(1).expand(N, M, M, 1)

                nb_dir = torch.gather(intent_dir_all, 2, g3)  # [N,M,k,3]
                nb_val = torch.gather(intent_val_all, 2, g1)  # [N,M,k,1]

                out_intent[:, :, :valid_k_fr, :3] = nb_dir
                out_intent[:, :, :valid_k_fr, 3:] = nb_val

            topk_intent_flat = out_intent.reshape(N, M, -1)  # [N,M,4*K_friends]
        else:
            target_feat_flat = torch.zeros((N, M, K_target * 4), device=dev, dtype=dtype)
            topk_intent_flat = torch.zeros((N, M, K_friends * 4), device=dev, dtype=dtype)

        # ====================== 3. 自身状态 ======================
        env_origins_expanded = self.terrain.env_origins.unsqueeze(1)  # [N, 1, 3]
        self_pos_abs = self.fr_pos - env_origins_expanded             # [N, M, 3]
        self_vel_abs = self.fr_vel_w                                  # [N, M, 3]

        # ====================== 3.5 其他友机视野内：按光轴角度从近到远的Top-5 bearing信息（复用 target_obs_container） ======================
        # 每个友机获得其他所有友机(M-1个)的 Top-5 bearing（世界系单位方向向量）
        # 维度: [N, M, (M-1)*5*3]
        k_fixed = 5
        topk_bearing_flat = torch.zeros(N, M, (M - 1) * k_fixed * 3, device=dev, dtype=dtype)

        if (E > 0) and (M > 1):
            # target_obs_container 的第三维是 K_target，所以最多只能取到 K_target
            valid_k = min(k_fixed, K_target)

            for obs_m in range(M):
                bearing_idx = 0
                for src_m in range(M):
                    if src_m == obs_m:
                        continue

                    # src_m 的 Top-K_target 目标，slot0..是按“离光轴角度从近到远”排好的
                    bearing = target_obs_container[:, src_m, :valid_k, :3]   # [N, valid_k, 3]
                    lock    = target_obs_container[:, src_m, :valid_k, 3:4]  # [N, valid_k, 1]
                    bearing = bearing * lock  # 再保险：不可见则置零（其实你前面 final_vecs 已经做过一次）

                    # 固定输出 5 个，不足的 padding 为 0
                    tmp = torch.zeros(N, k_fixed, 3, device=dev, dtype=dtype)
                    if valid_k > 0:
                        tmp[:, :valid_k, :] = bearing

                    start = bearing_idx * k_fixed * 3
                    end   = (bearing_idx + 1) * k_fixed * 3
                    topk_bearing_flat[:, obs_m, start:end] = tmp.reshape(N, -1)

                    bearing_idx += 1

        # 可选：冻结的“观测者”不需要任何观测，整段清零（避免 frozen agent 产生策略输出的干扰）
        topk_bearing_flat = topk_bearing_flat * friend_alive.unsqueeze(-1).float()

        # ====================== 4. 拼接 ======================
        obs_each = torch.cat(
            [
                topk_pos_flat,        # 3*K_friends
                topk_vel_flat,        # 3*K_friends
                topk_intent_flat,     # 4*K_friends
                self_pos_abs,         # 3
                self_vel_abs,         # 3
                target_feat_flat,     # 4*K_target
                topk_bearing_flat,    # (M-1)*5*3 - 其他友机bearing信息
            ],
            dim=-1,
        )

        return {ag: obs_each[:, i, :] for i, ag in enumerate(self.possible_agents)}

    def _get_states(self) -> torch.Tensor:
        N, M, E = self.num_envs, self.M, self.E
        dev, dtype = self.device, self.fr_pos.dtype

        # ===== 1. 友机状态：位置、速度、姿态 =====
        env_origins_expanded = self.terrain.env_origins.unsqueeze(1)  # [N, 1, 3]
        friend_pos_local = self.fr_pos - env_origins_expanded         # [N, M, 3] - 相对位置

        # 获取友机姿态（四元数）
        friend_quats = self._friendly_world_quats_zup()  # [N, M, 4]

        # 给冻结的友机乘上冻结标志（位置、速度、姿态置为0）
        friend_frozen_mask = (~self.friend_frozen).float().unsqueeze(-1)    # [N, M, 1]
        friend_pos_masked = friend_pos_local * friend_frozen_mask           # [N, M, 3]
        friend_vel_masked = self.fr_vel_w * friend_frozen_mask              # [N, M, 3]
        friend_quat_masked = friend_quats * friend_frozen_mask              # [N, M, 4]

        # 按照id平铺：每个友机的[位置3, 速度3, 姿态4] -> [N, M*10]
        friend_states = torch.cat([
            friend_pos_masked,    # [N, M, 3] - 位置
            friend_vel_masked,    # [N, M, 3] - 速度
            friend_quat_masked,   # [N, M, 4] - 姿态
        ], dim=-1).reshape(N, -1)  # [N, M*10]

        # ===== 2. 敌机状态：位置、速度 =====
        enemy_pos_local = self.enemy_pos - env_origins_expanded  # [N, E, 3]

        # 给冻结的敌机乘上冻结标志（位置、速度置为0）
        enemy_frozen_mask = (~self.enemy_frozen).float().unsqueeze(-1)  # [N, E, 1]
        enemy_pos_masked = enemy_pos_local * enemy_frozen_mask  # [N, E, 3]
        enemy_vel_masked = self.enemy_vel * enemy_frozen_mask    # [N, E, 3]

        # 按照id平铺：每个敌机的[位置3, 速度3] -> [N, E*6]
        enemy_states = torch.cat([
            enemy_pos_masked,    # [N, E, 3] - 位置
            enemy_vel_masked,    # [N, E, 3] - 速度
        ], dim=-1).reshape(N, -1)  # [N, E*6]

        # ===== 3. 敌机团中心位置 =====
        enemy_centroid_relative = self._enemy_centroid - env_origins_expanded.squeeze(1)  # [N, 3] - 相对环境坐标

        # ===== 4. 组装最终状态 =====
        states = torch.cat([
            friend_states,           # [N, M*10] - 友机位置、速度、姿态
            enemy_states,            # [N, E*6]  - 敌机位置、速度
            enemy_centroid_relative, # [N, 3]    - 敌机团中心位置
        ], dim=-1)

        return states

# ---------------- Gym 注册 ----------------
from config import agents

gym.register(
    id="FAST-Intercept-Swarm-Distributed",
    entry_point=FastInterceptionSwarmMARLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FastInterceptionSwarmMARLCfg,
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:L_M_interception_swarm_skrl_mappo_cfg.yaml",
        # "skrl_ppo_cfg_entry_point": f"{agents.__name__}:L_M_interception_swarm_skrl_mappo_cfg.yaml",
        "skrl_ippo_cfg_entry_point":  f"{agents.__name__}:L_M_interception_swarm_ippo_new.yaml",
        # "skrl_ippo_cfg_entry_point":  f"{agents.__name__}:L_M_interception_swarm_ippo_old.yaml", # 一个agent一个policy

        # 创新配置：基于IPPO的混合Critic架构（论文核心创新点）
        # "skrl_ippo_cfg_entry_point":  f"{agents.__name__}:swarm_interception_hybrid_critic.yaml",
    },
)
