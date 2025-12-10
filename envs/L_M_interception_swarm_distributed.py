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
    enemy_height_max = 15.0
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
    flight_altitude = 5.0

    obs_k_target: int = 10   # 观测最近的多少个敌机
    obs_k_friends: int = 15   # 观测最近的多少个友机

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
    row_height_diff: float  = 3.0      # 高度阶梯 (后排比前排高 1m)

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
        # single_obs_dim = 7 * int(M) + 10 * 4 + 10
        # single_obs_dim = 7 * int(M) + 14
        single_obs_dim = 6 * self.obs_k_friends + 4 * self.obs_k_target + 10
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
        # single_obs_dim = 7 * int(M) + 10 * 4 + 10
        # single_obs_dim = 7 * int(M) + 14
        single_obs_dim = 6 * cfg.obs_k_friends + 4 * cfg.obs_k_target + 10
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

        self._fr_vel_cmd = torch.zeros(N, self.M, 3, device=dev, dtype=dtype) # 11111111111111

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

        # ---- agent id one-hot feature ----
        self._agent_id_onehot = torch.eye(self.M, device=dev, dtype=torch.float32).unsqueeze(0)

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
        self._axis_hat = axis / norm                # 敌方目标点指向敌团质心的单位向量

        axis_xy = centroid[:, :2] - self._goal_e[:, :2]
        norm_xy = axis_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._axis_hat_xy = axis_xy / norm_xy

    def _spawn_enemy(self, env_ids: torch.Tensor):
        # ---- 基本量 ----
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
            up   = torch.stack([ks * step,  ks * step, torch.zeros_like(ks)], dim=-1) # 生成上半部分的点，坐标为 (k*step, k*step, 0)，沿 y=x 方向（45 度线）
            down = torch.stack([ks * step, -ks * step, torch.zeros_like(ks)], dim=-1) # 生成下半部分的点
            pts = torch.cat([torch.zeros(1, 3, dtype=dtype, device=dev), up, down], dim=0)  # [1+2K,3] 合并中心点和对称点
            # 若还有 1 架余数（E 为偶数），补在右上
            if (E - 1) % 2 == 1:
                extra_k = torch.tensor([(K + 1) * step], dtype=dtype, device=dev)
                extra   = torch.stack([extra_k, extra_k, torch.zeros_like(extra_k)], dim=-1)  # [1,3]
                pts = torch.cat([pts, extra], dim=0)
            return _centerize(pts[:E, :])

        # 1) 平面长方形（长边沿 +X）
        def _tmpl_rect_2d(E: int, s: float, aspect: float = 2.0) -> torch.Tensor:
            r, c = _rect2d_dims(E, aspect)
            xyz = _grid2d(r, c, s)[:E, :]
            return xyz

        # 1b) 平面正方形（近似方阵，aspect ≈ 1.0）
        def _tmpl_square_2d(E: int, s: float) -> torch.Tensor:
            # 直接复用 rect_2d，只是把宽高比强制为 1
            return _tmpl_rect_2d(E, s, aspect=1.0)

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

        # 3) 立体正方体（先取最大正方体 n^3，再在中间高度层横向补点）
        def _tmpl_cube_3d(E: int, s: float, sz_: float) -> torch.Tensor:
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)

            # 最大 n，使 n^3 <= E
            n = max(1, int(round(E ** (1.0 / 3.0))))
            while (n + 1) ** 3 <= E:
                n += 1
            while n ** 3 > E:
                n -= 1
            base_count = n ** 3

            # 基础 n×n×n 立方体
            xs = torch.arange(n, dtype=dtype, device=dev)
            ys = torch.arange(n, dtype=dtype, device=dev)
            zs = torch.arange(n, dtype=dtype, device=dev)
            X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")  # [n,n,n]

            Xf = X.reshape(-1)
            Yf = Y.reshape(-1)
            Zf = Z.reshape(-1)
            base_xyz = torch.stack(
                [Xf * s, Yf * s, Zf * sz_],
                dim=-1,
            )  # [base_count,3]

            pts = base_xyz
            rem = E - base_count
            if rem > 0:
                # 余下 rem 架放在中间高度层 z_mid，横着排
                z_mid = n // 2
                idx = torch.arange(rem, dtype=torch.long, device=dev)
                col_idx = idx // n   # 在 x 方向的附加列号：n, n+1, ...
                row_idx = idx % n    # 在 y 方向的位置：0..n-1

                x_extra = (n + col_idx).to(dtype) * s
                y_extra = row_idx.to(dtype) * s
                z_extra = torch.full((rem,), float(z_mid), dtype=dtype, device=dev) * sz_

                extra_xyz = torch.stack([x_extra, y_extra, z_extra], dim=-1)  # [rem,3]
                pts = torch.cat([pts, extra_xyz], dim=0)

            pts = pts[:E]
            return _centerize(pts)

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

        # 泊松采样生成敌机
        eta_poisson = float(getattr(self.cfg, "enemy_poisson_eta", 0.7))
        r_small = float(getattr(self.cfg, "enemy_cluster_radius", s_min))

        def _tmpl_poisson_2d(E: int, s: float, eta: float = 0.7) -> torch.Tensor:
            """在局部坐标系内做 Poisson disk 采样，返回 [E,3]，质心在原点，Z=0."""
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)

            two_pi = 2.0 * math.pi
            # 估算需要的半径
            r_needed = 0.5 * s * math.sqrt(E / max(eta, 1e-6))
            r_env = max(r_small, r_needed * 1.02)

            BATCH        = 128
            MAX_ROUNDS   = 256
            STAGN_ROUNDS = 5
            GROW_FACTOR  = 1.05

            pts = torch.zeros(E, 2, dtype=dtype, device=dev)  # [E,2]
            filled = 0
            stagn  = 0
            s2 = s * s

            for _ in range(MAX_ROUNDS):
                if filled >= E:
                    break

                u = torch.rand(BATCH, device=dev, dtype=dtype)
                v = torch.rand(BATCH, device=dev, dtype=dtype)
                rr  = r_env * torch.sqrt(u.clamp_min(1e-12))
                ang = two_pi * v
                cand = torch.stack([rr * torch.cos(ang), rr * torch.sin(ang)], dim=-1)  # [BATCH,2]

                if filled == 0:
                    pts[0] = cand[0]
                    filled = 1
                    stagn = 0
                    continue

                diff = cand.unsqueeze(1) - pts[:filled].unsqueeze(0)   # [BATCH,filled,2]
                sq   = (diff ** 2).sum(dim=-1)                         # [BATCH,filled]
                min_sq, _ = sq.min(dim=1)                              # [BATCH]
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

            z = torch.zeros(E, 1, dtype=dtype, device=dev)  # Z 全 0，后面统一处理高度
            xyz = torch.cat([pts, z], dim=-1)               # [E,3]
            return _centerize(xyz)

        # 组装为 [F,E,3]，一次性并行给所有 env 复用
        templates = torch.stack([
            _tmpl_v_wedge_2d(E, sz_v),                          # 0
            _tmpl_rect_2d(E, s_min, aspect=2.0),               # 1
            _tmpl_square_2d(E, s_min),                         # 2
            _tmpl_rect_3d(E, s_min, sz_v,  aspect_xy=2.0),     # 3
            _tmpl_cube_3d(E, s_min, sz_v),                     # 4
            _tmpl_rect_3d_reverse(E, s_min, sz_v,  aspect_xy=2.0),  # 5
            # _tmpl_poisson_2d(E, s_min, eta_poisson),           # 6 ← 泊松模板
        ], dim=0)
        POISSON_IDX = templates.shape[0] - 1  # 也就是 6

        # ---- 每个 env 随机挑一种队形（并行索引）----
        f_idx = torch.randint(low=0, high=templates.shape[0], size=(N,), device=dev)  # [N]
        local_xyz = templates[f_idx, :, :]  # [N,E,3]  <- 并行 gather，无 per-env 循环
        local_xyz = local_xyz.clone()
        local_xyz[..., 0] *= -1.0

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

        local_z = local_xyz[:, :, 2:3]  # [N,E,1]
        min_local_z, _ = local_z.min(dim=1, keepdim=True)  # [N,1,1]
        z_bottom = hmin + torch.rand(N, 1, 1, device=dev, dtype=dtype) * max(1e-6, (hmax - hmin))
        z_rel = z_bottom + (local_z - min_local_z)         # [N,E,1] 相对 origins[:,2]
        z_abs = origins[:, 2:3].unsqueeze(1) + z_rel       # 绝对高度 [N,E,1]

        enemy_pos = torch.cat([xy, z_abs], dim=-1)        # [N,E,3]

        # ---- 对于泊松模板的环境，覆盖高度为“每个敌机独立随机” ----
        # is_poisson_env = (f_idx == POISSON_IDX)    # [N] bool
        # if is_poisson_env.any():
        #     env_poisson = torch.nonzero(is_poisson_env, as_tuple=False).squeeze(-1)  # [N_p]
        #     Np = env_poisson.shape[0]
        #     # origins_poisson: 这些 env 的地面高度
        #     origins_poisson = origins[env_poisson]  # [N_p,3]

        #     # 每个 env、每个敌机一个独立随机高度 ∈ [hmin, hmax]
        #     z_rand = origins_poisson[:, 2:3].unsqueeze(1) + (
        #         hmin + torch.rand(Np, E, 1, device=dev, dtype=dtype) * max(1e-6, (hmax - hmin))
        #     )  # [N_p,E,1]

        #     enemy_pos[env_poisson, :, 2:3] = z_rand  # 覆盖 Z

        self.enemy_pos[env_ids] = enemy_pos               # 写回（一次性）

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
        if self.enemy_visualizer is not None:
            self.enemy_visualizer.visualize(translations=self._flatten_agents(self.enemy_pos))

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

    # def _apply_action(self):
    #     dt = float(self.physics_dt)
    #     r = float(self.cfg.hit_radius)
    #     is_first_substep = ((self._sim_step_counter - 1) % self.cfg.decimation) == 0
    #     if is_first_substep:
    #         self._newly_frozen_friend[:] = False
    #         self._newly_frozen_enemy[:]  = False

    #     # 得到步首位置/冻结状态
    #     fr_pos0 = self.fr_pos.clone()
    #     en_pos0 = self.enemy_pos.clone()
    #     fz0 = self.friend_frozen.clone()
    #     ez0 = self.enemy_frozen.clone()

    #     # ---------- 判断是否在步首命中 ----------
    #     active_pair0 = (~fz0).unsqueeze(2) & (~ez0).unsqueeze(1)  # [N,M,E]
    #     if active_pair0.any():
    #         diff0 = fr_pos0.unsqueeze(2) - en_pos0.unsqueeze(1)   # [N,M,E,3]
    #         dist0 = torch.linalg.norm(diff0, dim=-1)              # [N,M,E]
    #         hit_pair0 = (dist0 <= r) & active_pair0               # [N,M,E]
    #         # 敌机是否被命中（沿友机维度），不能用hit_pair0.any(dim=2)因为这样只知道友机打中了，但是会出现多友机命中同一敌机的情况，需要选最近的友机作为击中者
    #         newly_hitted_enemy = hit_pair0.any(dim=1)             # [N,E] 返回的是第N个环境的第E个敌机是否被命中。沿着友机维度M看，有没有命中该敌机，从上往下看，再从左往右。.any(dim=1)传入的 dim 是被消掉（被扫描）的那一轴；剩下的轴就是你“固定住”的索引。
    #         if newly_hitted_enemy.any():
    #             # print("hit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #             # —— 为每个“本步新冻敌机”选最近友机作为击中者 ——
    #             INF = torch.tensor(float("inf"), device=self.device, dtype=dist0.dtype)
    #             dist_masked0 = torch.where(hit_pair0, dist0, INF)     # [N,M,E] 存储的是命中对的距离，未命中对为 +inf
    #             hitter_idx   = dist_masked0.argmin(dim=1)             # [N,E]存的是第N个环境中敌机E被哪个友机打中的友机索引。对每个(n,e)，在友机维M上找最小距离对应的友机索引j*，即击中者，二维矩阵就是竖着那一列找最小的索引。如果有一列全是+inf（即该敌机未被任何友机命中），argmin会返回0
    #             # newly_hitted_enemy形状是[N, E]（环境×敌机的布尔掩码）。newly_hitted_enemy.nonzero(as_tuple=False) 返回形状[K, 2]的整型张量，每一行是一个(n, e)。再 .T 转置成 [2, K]
    #             env_idx, enemy_idx = newly_hitted_enemy.nonzero(as_tuple=False).T       # [K]，将第N个环境中被击中的第E个敌机的环境与敌机的索引拿出来。env_idx是这些新命中敌机的场景索引n列表，enemy_idx是对应敌机索引e列表（长度为K），一维。可以避免dist_masked0中未命中对的+inf，返回索引0的干扰
    #             friend_idx = hitter_idx[env_idx, enemy_idx]              # [K]，取出每个“新命中敌机”的击中者友机索引 j*，用上面拿出来的索引去取打中敌机的友机索引

    #             # —— 仅冻结击中者（友机侧）和被击中敌机（敌机侧） ——
    #             hit_friend_mask = torch.zeros_like(self.friend_frozen)    # [N,M] bool
    #             hit_friend_mask[env_idx, friend_idx] = True               # 标记第env_idx个环境中，第friend_idx个友机击中了敌机

    #             # —— 只把击中者记为“新冻友机”，并写捕获点 ——
    #             self._newly_frozen_friend |= hit_friend_mask               # [N,M] 用于记录当前步新冻友机，在奖励阶段用
    #             self._newly_frozen_enemy |= newly_hitted_enemy             # [N,E] 记录“本步新冻敌机”,在奖励阶段用

    #             # self.enemy_capture_pos[newly_hitted_enemy]      = en_pos0[newly_hitted_enemy]
    #             # self.friend_capture_pos[env_idx, friend_idx]    = en_pos0[newly_hitted_enemy]
    #             self.friend_frozen |= hit_friend_mask                     # 冻结击中者,self.friend_frozen用于全局记录冻结状态
    #             self.enemy_frozen  |= newly_hitted_enemy                  # 冻结被击中者

    #     fz = self.friend_frozen
    #     ez = self.enemy_frozen

    #     # ---------- 友机姿态/速度（冻结为0） ----------
    #     cos_th_now = torch.cos(self.theta).clamp_min(1e-6)                  # 当前俯仰角的余弦值
    #     Vm_eff = torch.where(fz, torch.zeros_like(self.Vm), self.Vm)        # 网络映射过来的友机的速度
    #     Vm_eps = Vm_eff.clamp_min(1e-6)
    #     theta_rate = self.g0 * (self._ny - cos_th_now) / Vm_eps
    #     psi_rate   = - self.g0 * self._nz / (Vm_eps * cos_th_now)
    #     theta_rate = torch.where(fz, torch.zeros_like(theta_rate), theta_rate)
    #     psi_rate   = torch.where(fz, torch.zeros_like(psi_rate),   psi_rate)

    #     THETA_RATE_LIMIT = 1.0
    #     PSI_RATE_LIMIT   = 1.0
    #     theta_rate = torch.clamp(theta_rate, -THETA_RATE_LIMIT, THETA_RATE_LIMIT)
    #     psi_rate   = torch.clamp(psi_rate,   -PSI_RATE_LIMIT,   PSI_RATE_LIMIT)

    #     self.theta = self.theta + theta_rate * dt                                               # 俯仰角 θt+1​=θt​+θ˙t​Δt
    #     self.psi_v = (self.psi_v + psi_rate * dt + math.pi) % (2.0 * math.pi) - math.pi         # 偏航角

    #     sin_th, cos_th = torch.sin(self.theta), torch.cos(self.theta)
    #     sin_ps, cos_ps = torch.sin(self.psi_v), torch.cos(self.psi_v)
    #     Vxm = Vm_eff * cos_th * cos_ps
    #     Vym = Vm_eff * sin_th
    #     Vzm = -Vm_eff * cos_th * sin_ps
    #     V_m = torch.stack([Vxm, Vym, Vzm], dim=-1)
    #     fr_vel_w_step = y_up_to_z_up(V_m)

    #     # 敌机速度（始终沿-axis_hat方向，以恒定速度前进，冻结为0）
    #     v_enemy_xy = -self._axis_hat_xy
    #     zeros_z = torch.zeros_like(v_enemy_xy[:, :1]) 
    #     v_move_3d = torch.cat([v_enemy_xy, zeros_z], dim=-1)
    #     v_move_expanded = v_move_3d.unsqueeze(1).expand(-1, self.E, -1)
    #     enemy_vel_step = v_move_expanded * float(self.cfg.enemy_speed)
    #     enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)

    #     # ---------- 推进 ----------
    #     fr_pos1 = fr_pos0 + fr_vel_w_step * dt
    #     en_pos1 = en_pos0 + enemy_vel_step * dt

    #     # ---------- 冻结 ----------
    #     if fz.any():
    #         fr_vel_w_step = torch.where(fz.unsqueeze(-1), torch.zeros_like(fr_vel_w_step), fr_vel_w_step)
    #         fr_pos1       = torch.where(fz.unsqueeze(-1), self.friend_capture_pos, fr_pos1)
    #     if ez.any():
    #         enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)
    #         en_pos1        = torch.where(ez.unsqueeze(-1), self.enemy_capture_pos, en_pos1)

    #     # ---------- 将积分后的速度写到全局变量中，并更新位置 ----------
    #     self.fr_vel_w  = fr_vel_w_step
    #     self.enemy_vel = enemy_vel_step
    #     self.fr_pos    = fr_pos1
    #     self.enemy_pos = en_pos1

    #     # ---------- 云台与可视化 ----------
    #     self._gimbal_control()
    #     self._refresh_enemy_cache()
    #     self._update_traj_vis()

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
        cap_pos = 0.0
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
        enemy_active     = (~self.enemy_frozen)                      # [N,E] bool
        enemy_active_any = self._enemy_active_any                    # [N]   bool
        vis_fe = self._gimbal_enemy_visible_mask()                   # [N, M, E]
        vis_ff = self._gimbal_friend_visible_mask()                  # [N, M, M]

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
        mission_success = self.enemy_frozen.all(dim=1, keepdim=True).float()    # [N,1]

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
        enemy_reach_goal_any = enemy_goal_any.float().unsqueeze(1) * friend_active.float() 

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
        r_each = r_each + mission_success * enemy_all_killed_reward_weight  # [N,M]
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

        # # ----------------------debug ----------------------
        # r_centroid     = centroid_weight * centroid_each                                 # [N,M]
        # r_v_to_c       = v_to_c_weight * vel_to_centroid_each                            # [N,M]
        # r_hit          = hit_weight * per_agent_hit                                      # [N,M]
        # r_gimbal_f     = - fb_weight * penalty_friend_each                               # [N,M]
        # r_gimbal_e     = ec_weight * enemy_count_each                                    # [N,M]
        # r_enemy_goal   = - enemy_reach_goal_weight * enemy_reach_goal_any                # [N,M]
        # r_all_killed   = mission_success * enemy_all_killed_reward_weight                # [N,M]
        # r_target_align = target_align_weight * target_align_each                         # [N,M]
        # r_overshoot    = - leak_penalty_weight * leak_each                               # [N,M]
        # r_fr_high      = - friend_too_high_penalty_weight * penalty_friend_high_each     # [N,M]
        # r_fr_low       = - friend_too_low_penalty_weight  * penalty_friend_low_each      # [N,M]

        # # 总 reward（和 r_each 一致）
        # reward = (
        #     r_centroid
        #     + r_v_to_c
        #     + r_hit
        #     + r_gimbal_f
        #     + r_gimbal_e
        #     + r_enemy_goal
        #     + r_all_killed
        #     + r_target_align
        #     + r_overshoot
        #     + r_fr_high
        #     + r_fr_low
        # )  # [N,M]

        # # --- episode 级累计统计 ---
        # if "total" not in self.episode_sums:
        #     base = torch.zeros_like(reward)
        #     self.episode_sums["total"]           = base.clone()
        #     self.episode_sums["centroid"]        = base.clone()
        #     self.episode_sums["vel_to_centroid"] = base.clone()
        #     self.episode_sums["hit"]             = base.clone()
        #     self.episode_sums["gimbal_f"]        = base.clone()
        #     self.episode_sums["gimbal_e"]        = base.clone()
        #     self.episode_sums["enemy_goal"]      = base.clone()
        #     self.episode_sums["all_killed"]      = base.clone()
        #     self.episode_sums["target_align"]    = base.clone()
        #     self.episode_sums["overshoot"]       = base.clone()
        #     self.episode_sums["friend_too_high"] = base.clone()
        #     self.episode_sums["friend_too_low"]  = base.clone()

        # self.episode_sums["total"]           += reward
        # self.episode_sums["centroid"]        += r_centroid
        # self.episode_sums["vel_to_centroid"] += r_v_to_c
        # self.episode_sums["hit"]             += r_hit
        # self.episode_sums["gimbal_f"]        += r_gimbal_f
        # self.episode_sums["gimbal_e"]        += r_gimbal_e
        # self.episode_sums["enemy_goal"]      += r_enemy_goal
        # self.episode_sums["all_killed"]      += r_all_killed
        # self.episode_sums["target_align"]    += r_target_align
        # self.episode_sums["overshoot"]       += r_overshoot
        # self.episode_sums["friend_too_high"] += r_fr_high
        # self.episode_sums["friend_too_low"]  += r_fr_low
        # # ----------------------debug----------------------


        # ========================== 1. 组装奖励字典 ==========================
        # 这里我们将“权重 * 原始项”直接存入字典，这就是 Agent 实际收到的奖励
        reward_terms = {
            "centroid":         centroid_weight * centroid_each,
            "vel_to_centroid":  v_to_c_weight * vel_to_centroid_each,
            "hit":              hit_weight * per_agent_hit,
            "gimbal_friend":    -fb_weight * penalty_friend_each,
            "gimbal_enemy":     ec_weight * enemy_count_each,
            "enemy_reach_goal": -enemy_reach_goal_weight * enemy_reach_goal_any,
            "all_killed":       enemy_all_killed_reward_weight * mission_success,
            "target_align":     target_align_weight * target_align_each,
            "overshoot":        -leak_penalty_weight * leak_each,
            "too_high":         -friend_too_high_penalty_weight * penalty_friend_high_each,
            "too_low":          -friend_too_low_penalty_weight * penalty_friend_low_each,
        }

        # ========================== 2. 计算总奖励 ==========================
        # 将所有分量叠加
        r_total = sum(reward_terms.values())

        # ========================== 3. 累积 Episode Sums ==========================
        # 懒初始化
        if "total" not in self.episode_sums:
             self.episode_sums["total"] = torch.zeros_like(r_total)
             for k, v in reward_terms.items():
                 # 处理一下维度广播的问题，确保 buffer 是 [N, M]
                 if v.ndim < r_total.ndim:
                     # 例如 mission_success 是 [N, 1]，需要广播到 [N, M] 才能累加
                     v_expanded = v.expand_as(r_total)
                     self.episode_sums[k] = torch.zeros_like(r_total)
                 else:
                     self.episode_sums[k] = torch.zeros_like(v)

        # 执行累加
        self.episode_sums["total"] += r_total
        for k, v in reward_terms.items():
            # 确保维度匹配
            if v.shape != self.episode_sums[k].shape:
                v = v.expand_as(self.episode_sums[k])
            self.episode_sums[k] += v

        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        N         = self.num_envs
        device    = self.device
        r2_goal = float(self.cfg.enemy_goal_radius) ** 2
        xy_max2 = float(self.cfg.enemy_cluster_ring_radius + 50.0) ** 2
        # ---------- 基本终止判据（逐env） ----------
        success_all_enemies = self.enemy_frozen.all(dim=1)                        # [N] 敌全灭（友军成功），all函数用于判断输入中个张量是否都是True
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
        if alive_mask.any():
            tol = float(getattr(self.cfg, "overshoot_tol", 2.0))
            idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)          # [n]
            friend_active = (~self.friend_frozen[idx])                    # [n,M]
            enemy_active  = (~self.enemy_frozen[idx])                     # [n,E]
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
                # enemy_frozen[env_ids]: [N_reset, E]，True 表示该敌机已被击落/冻结
                frozen = self.enemy_frozen[env_ids]              # bool，[N_reset, E]
                frozen_cnt = frozen.sum(dim=1)                   # [N_reset]
                total = float(self.E)
                rate = frozen_cnt.float() / total                # [N_reset]

                print("Interception rate per env (frozen enemies / total enemies):")
                for i_local, env_id in enumerate(env_ids.tolist()):
                    c = int(frozen_cnt[i_local].item())
                    r = rate[i_local].item()
                    print(f"  Env {env_id}: {c} / {int(total)} = {r:.3f}")
                print(
                    f"  Summary: mean={rate.mean().item():.3f}  "
                    f"min={rate.min().item():.3f}  max={rate.max().item():.3f}"
                )
                print("---------------------------------------------------------")

        # ========================== LOGGING TO TENSORBOARD ==========================
        # 将 episode_sums 中的数据写入 extras["log"]
        # 注意：extras 必须在 reset 时填充，这样 runner 才能在 done 的时候取到数据
        if not hasattr(self, "extras"):
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

        # 遍历所有累积的奖励分项
        if len(self.episode_sums) > 0:
            for k, v in self.episode_sums.items():
                # v 的形状是 [N, M] (Env x Agent)
                # 我们需要取 mean：先对 M 个智能体取平均，再对 N 个 Reset 的环境取平均
                # 或者：根据你的偏好，如果是 cooperative 任务，通常关心 Sum(M) 的 Mean(N)

                # 方式 A：平均每个智能体的平均奖励 (Average Reward per Agent) -> 推荐
                # metric_val = v[env_ids].mean() 
                
                # 方式 B：整个集群的总奖励 (Total Swarm Reward) 
                metric_val = v[env_ids].sum(dim=1).mean()

                self.extras["log"][f"Episode_Reward/{k}"] = metric_val

                # 清零该环境的累积器
                self.episode_sums[k][env_ids] = 0.0

            # 补充拦截率统计 (Interception Rate)
            if self.E > 0:
                frozen_cnt = self.enemy_frozen[env_ids].sum(dim=1).float() # [N_reset]
                rate = frozen_cnt / float(self.E)
                self.extras["log"]["Episode_Metric/Interception_Rate"] = rate.mean()

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

    # 取K个最近的敌机和友机的观测
    def _get_observations(self) -> dict[str, torch.Tensor]:
        N, M, E = self.num_envs, self.M, self.E
        dev, dtype = self.device, self.fr_pos.dtype
        eps = 1e-9

        K_target  = self.cfg.obs_k_target
        K_friends = self.cfg.obs_k_friends

        # ====================== 1. 友机相对观测 (Top-K) ======================
        if M > 1:
            pos_i = self.fr_pos.unsqueeze(2)   # [N,M,1,3]
            pos_j = self.fr_pos.unsqueeze(1)   # [N,1,M,3]
            dist_ij_raw = torch.linalg.norm(pos_j - pos_i, dim=-1)  # [N,M,M]

            # 屏蔽自己和冻结友机
            large = torch.full_like(dist_ij_raw, 1e6)
            eye = torch.eye(M, device=dev, dtype=torch.bool).unsqueeze(0)
            friend_alive = (~self.friend_frozen)
            both_alive = friend_alive.unsqueeze(1) & friend_alive.unsqueeze(2)
            valid_pair = (~eye) & both_alive
            dist_ij = torch.where(valid_pair, dist_ij_raw, large)

            # 排序,返回沿给定维度按值升序排序张数的索引
            sorted_idx_all = dist_ij.argsort(dim=-1)

            # 拿取前K个友机的索引
            valid_k_fr = min(M - 1, K_friends)
            top_k_idx = sorted_idx_all[..., :valid_k_fr] # [N, M, valid_k_fr]

            # Gather 位置和速度
            gather_idx = top_k_idx.unsqueeze(-1).expand(-1, -1, -1, 3)

            closest_pos = torch.gather(self.fr_pos.unsqueeze(1).expand(N, M, M, 3), 2, gather_idx)
            closest_vel = torch.gather(self.fr_vel_w.unsqueeze(1).expand(N, M, M, 3), 2, gather_idx)

            # 转为相对量
            rel_pos = closest_pos - self.fr_pos.unsqueeze(2)
            rel_vel = closest_vel - self.fr_vel_w.unsqueeze(2)

            # Padding 到固定长度
            out_pos = torch.zeros(N, M, K_friends, 3, device=dev, dtype=dtype)
            out_vel = torch.zeros(N, M, K_friends, 3, device=dev, dtype=dtype)

            if valid_k_fr > 0:
                out_pos[:, :, :valid_k_fr, :] = rel_pos
                out_vel[:, :, :valid_k_fr, :] = rel_vel

            topk_pos_flat = out_pos.reshape(N, M, -1)
            topk_vel_flat = out_vel.reshape(N, M, -1)
        else:
            topk_pos_flat = torch.zeros(N, M, K_friends * 3, device=dev, dtype=dtype)
            topk_vel_flat = torch.zeros(N, M, K_friends * 3, device=dev, dtype=dtype)

        # ====================== 2. 敌机观测 (Top-K) ======================
        # 输出维度：K_target * 4
        if E > 0:
            vis_fe = self._gimbal_enemy_visible_mask()  # [N, M, E]

            rel_all  = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)
            dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)
            dir_all  = rel_all / dist_all

            cam_dir = self._dir_from_yaw_pitch(self._gimbal_yaw, self._gimbal_pitch).unsqueeze(2)
            cos_ang = (cam_dir * dir_all).sum(dim=-1).clamp(-1.0+1e-6, 1.0-1e-6)
            angle   = torch.acos(cos_ang)

            large_angle = 100.0
            angle_masked = torch.where(vis_fe, angle, torch.tensor(large_angle, device=dev))
            sorted_indices = angle_masked.argsort(dim=-1, descending=False)

            valid_k = min(E, K_target)
            top_k_idx = sorted_indices[..., :valid_k]

            gather_idx = top_k_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            best_vecs = torch.gather(dir_all, 2, gather_idx)
            best_angles = torch.gather(angle_masked, 2, top_k_idx)

            is_locked = (best_angles < large_angle)
            final_vecs = best_vecs * is_locked.unsqueeze(-1).float()
            final_locks = is_locked.unsqueeze(-1).float()

            target_obs_container = torch.zeros(N, M, K_target, 4, device=dev, dtype=dtype)
            valid_part = torch.cat([final_vecs, final_locks], dim=-1)
            target_obs_container[:, :, :valid_k, :] = valid_part
            target_feat_flat = target_obs_container.reshape(N, M, -1)
        else:
            target_feat_flat = torch.zeros((N, M, K_target * 4), device=dev, dtype=dtype)

        # ====================== 3. 自身状态 & ID ======================
        self_pos_abs = self.fr_pos
        self_vel_abs = self.fr_vel_w

        cen = self._enemy_centroid
        rel_c = cen.unsqueeze(1) - self.fr_pos
        dist_c = torch.linalg.norm(rel_c, dim=-1, keepdim=True).clamp_min(eps)
        e_hat_c = rel_c / dist_c

        agent_id_feat = self._agent_id_onehot.expand(N, -1, -1).to(dtype=dtype)

        # ====================== 4. 拼接 ======================
        obs_each = torch.cat(
            [
                topk_pos_flat,           # K_friends * 3
                topk_vel_flat,           # K_friends * 3
                self_pos_abs,            # 3
                self_vel_abs,            # 3
                e_hat_c,                 # 3
                dist_c,                  # 1
                target_feat_flat,        # K_target * 4
                # agent_id_feat,           # M
            ],
            dim=-1,
        )

        obs_dict = {ag: obs_each[:, i, :] for i, ag in enumerate(self.possible_agents)}

        return obs_dict

    # 取K个最近敌机
    # def _get_observations(self) -> dict[str, torch.Tensor]:
        N, M, E = self.num_envs, self.M, self.E
        dev, dtype = self.device, self.fr_pos.dtype
        eps = 1e-9
        K_target = 10

        # ====================== 1. 友机相对观测 (保持不变) ======================
        pos_i = self.fr_pos.unsqueeze(2)   # [N,M,1,3]
        pos_j = self.fr_pos.unsqueeze(1)   # [N,1,M,3]
        dist_ij_raw = torch.linalg.norm(pos_j - pos_i, dim=-1)  # [N,M,M]

        # 把"自己"和"冻结友机"推到排序末尾
        large = torch.full_like(dist_ij_raw, 1e6)
        eye = torch.eye(M, device=dev, dtype=torch.bool).unsqueeze(0)
        friend_alive = (~self.friend_frozen)
        both_alive = friend_alive.unsqueeze(1) & friend_alive.unsqueeze(2)
        valid_pair = (~eye) & both_alive
        dist_ij = torch.where(valid_pair, dist_ij_raw, large)

        # 排序：近 -> 远
        sorted_idx = dist_ij.argsort(dim=-1)[:, :, :M-1]  # [N,M,M-1]

        # Gather 友机位置/速度
        other_pos_sorted = torch.gather(
            self.fr_pos.unsqueeze(1).expand(N, M, M, 3),
            2, sorted_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )
        other_vel_sorted = torch.gather(
            self.fr_vel_w.unsqueeze(1).expand(N, M, M, 3),
            2, sorted_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )

        # 转为相对量
        self_pos = self.fr_pos.unsqueeze(2)
        self_vel = self.fr_vel_w.unsqueeze(2)
        zeros_self = torch.zeros_like(self_pos)

        rel_pos_sorted = other_pos_sorted - self_pos
        rel_vel_sorted = other_vel_sorted - self_vel

        # 拼接友机信息 [N, M, 3*(M-1) * 2] -> [N, M, 6(M-1)]
        all_pos_sorted = torch.cat([zeros_self, rel_pos_sorted], dim=2).reshape(N, M, 3 * M)
        all_vel_sorted = torch.cat([zeros_self, rel_vel_sorted], dim=2).reshape(N, M, 3 * M)

        # ====================== 2. 多目标导引头观测 (Top-K) ======================
        # 目标：输出视野中心光轴附近 Top-K 个敌机的方向向量 + 锁定标志
        # 输出维度将变为：K_target * (3个方向 + 1个标志)
        if E > 0:
            # --- A. 基础几何计算 (不变) ---
            vis_fe = self._gimbal_enemy_visible_mask()  # [N, M, E]

            rel_all  = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2) # [N,M,E,3]
            dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)
            dir_all  = rel_all / dist_all  # [N,M,E,3]

            cam_dir = self._dir_from_yaw_pitch(self._gimbal_yaw, self._gimbal_pitch).unsqueeze(2) # [N,M,1,3]
            cos_ang = (cam_dir * dir_all).sum(dim=-1).clamp(-1.0+1e-6, 1.0-1e-6)
            angle   = torch.acos(cos_ang) # [N,M,E]

            # --- B. 筛选与排序 (Top-K Logic) ---
            # 1. 屏蔽：把视野外(不可见)的敌机角度设为极大值
            large_angle = 100.0
            angle_masked = torch.where(vis_fe, angle, torch.tensor(large_angle, device=dev))

            # 2. 排序：从小到大排序 (argsort)
            # sorted_indices: [N, M, E]
            sorted_indices = angle_masked.argsort(dim=-1, descending=False)

            # 3. 截取 Top-K
            # 注意：如果 E < K_target，则只取 E 个；通常 E >= K_target
            valid_k = min(E, K_target)
            top_k_idx = sorted_indices[..., :valid_k] # [N, M, valid_k]

            # 4. Gather 提取对应的 3D 向量
            # top_k_idx 扩展为 [N, M, valid_k, 3] 以适配 dir_all
            gather_idx = top_k_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            # best_vecs: [N, M, valid_k, 3]
            best_vecs = torch.gather(dir_all, 2, gather_idx)

            # 5. 提取对应的角度值，用于判断是否真的锁定
            # best_angles: [N, M, valid_k]
            best_angles = torch.gather(angle_masked, 2, top_k_idx)

            # 6. 生成锁定掩码 (Lock Mask)
            # 如果角度值仍然很大(large_angle)，说明这个位置其实是空的(视野内敌机不足K个)
            is_locked = (best_angles < (large_angle - 1.0)) # [N, M, valid_k] Bool

            # 7. 应用掩码：未锁定的位置向量置 0
            # [N, M, valid_k, 3]
            final_vecs = best_vecs * is_locked.unsqueeze(-1).float()
            final_locks = is_locked.unsqueeze(-1).float() # [N, M, valid_k, 1]

            # --- C. 构建固定维度的输出容器 ---
            # 为了防止 E < K_target 导致维度变化，或者 E 动态变化，这里使用固定容器
            # 容器形状：[N, M, K_target, 4] (3个方向 + 1个标志)
            target_obs_container = torch.zeros(N, M, K_target, 4, device=dev, dtype=dtype)

            # 拼接向量和标志: [N, M, valid_k, 4]
            valid_part = torch.cat([final_vecs, final_locks], dim=-1)

            # 填入容器 (前 valid_k 个位置)
            target_obs_container[:, :, :valid_k, :] = valid_part

            # --- D. 展平 ---
            # [N, M, K_target * 4]
            target_feat_flat = target_obs_container.reshape(N, M, -1)
        else:
            target_feat_flat = torch.zeros((N, M, K_target * 4), device=dev, dtype=dtype)

        # ====================== 3. 自身状态 & 战术引导 ======================
        self_pos_abs = self.fr_pos
        self_vel_abs = self.fr_vel_w

        # 敌团质心 (Search Guidance)：当 lock_feat=0 时，Agent 依赖这个飞
        cen = self._enemy_centroid
        rel_c = cen.unsqueeze(1) - self.fr_pos
        dist_c = torch.linalg.norm(rel_c, dim=-1, keepdim=True).clamp_min(eps)
        e_hat_c = rel_c / dist_c  # [N, M, 3]

        # Agent ID (One-hot)
        agent_id_feat = self._agent_id_onehot.expand(N, -1, -1).to(dtype=dtype)

        # ====================== 4. 拼接总观测 ======================
        obs_each = torch.cat(
            [
                all_pos_sorted,          # 3M
                all_vel_sorted,          # 3M
                self_pos_abs,            # 3
                self_vel_abs,            # 3
                e_hat_c,                 # 3
                dist_c,                  # 1
                target_feat_flat,        # K_target * 4
                agent_id_feat,           # M
            ],
            dim=-1,
        )

        obs_dict = {ag: obs_each[:, i, :] for i, ag in enumerate(self.possible_agents)}

        return obs_dict

    # 取一个最近敌机
    # def _get_observations(self) -> dict[str, torch.Tensor]:
    #     N, M, E = self.num_envs, self.M, self.E
    #     dev, dtype = self.device, self.fr_pos.dtype
    #     eps = 1e-9

    #     # ====================== 1. 友机相对观测 (保持不变) ======================
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

    #     # ====================== 3. 自身状态 & 战术引导 ======================
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
