from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import gymnasium as gym
from typing import Tuple, Optional, Literal

# Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# USD / Hydra
from pxr import UsdGeom, Gf, Vt


# ===============================
# Config
# ===============================
@configclass
class QuadcopterHierKineEnvCfg(DirectRLEnvCfg):
    """分层(高层RL + 低层控制）的纯运动学跟踪环境配置。

    高层以较低频率输出参考（加速度/期望速度/相对航点），
    低层将参考映射为世界系加速度并做积分推进。

    重要字段：
        hl_action_mode: 高层动作模式，可选 {"accel", "vel", "wp_rel"}。
        debug_vis_master: 可视化总开关;False 时不创建/更新任何可视化。
        debug_vis_target, debug_vis_los, debug_vis_env0_only: 可视化子开关。
    """

    # 视角
    viewer = ViewerCfg(eye=(6.0, -6.0, 8.0))

    # ---- 分层接口 ----
    hl_action_mode: Literal["accel", "vel", "wp_rel"] = "vel"
    hl_freq: float = 10.0
    ll_kp_v: float = 2.0
    ll_kd_v: float = 0.3
    wp_to_v_gain: float = 1.5

    # ---- 奖励权重 ----
    approach_weight = 1.0
    approach_clip = 0.5
    dist_weight = 0.05
    align_weight = 0.5
    smooth_action_weight = 0.01
    success_bonus = 5.0
    step_cost = 0.001

    # ---- 任务参数 ----
    follow_radius = 1.5
    success_pos_tol = 1.0
    success_hold_steps = 8
    lost_distance = 60.0
    workspace_radius = 100.0

    # ---- 运动学与限幅 ----
    v_max = 8.0                 # m/s
    a_max = 6.0                 # m/s^2
    keep_altitude = 1.0         # m
    target_speed = 2.0          # m/s (+X 匀速)
    wp_rel_max = 3.0            # m（仅 'wp_rel' 模式）

    # ---- 时序 ----
    episode_length_s = 30.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

    # ---- 空间 ----
    observation_space = 9       # 位置(3)+速度(3)+LOS(3)
    state_space = 0
    action_space = 3
    clip_action = 1.0

    # ---- 仿真 ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / physics_freq,
        render_interval=render_decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        debug_vis=True,
    )

    # 场景
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5, replicate_physics=True)

    # 仅显示用USD
    ROBOT_VIS_USD: str = "/home/wcw/swarm_rl/assets/X152b/x152b.usd"
    ROBOT_USD: str     = "/home/wcw/swarm_rl/assets/dji_fpv/v1/dji_fpv.usd"

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=ROBOT_USD, copy_from_source=False),
        actuators=None,
    )

    # ---- 可视化开关 ----
    debug_vis_master: bool = True       # 总开关
    debug_vis_target: bool = True       # 目标立方体
    debug_vis_los: bool = True          # LOS连线
    debug_vis_env0_only: bool = True    # 仅 env_0


# ===============================
# Environment
# ===============================
class QuadcopterHierKineEnv(DirectRLEnv):
    """分层纯运动学四旋翼环境(DirectRLEnv)"""

    cfg: QuadcopterHierKineEnvCfg

    # Visual prim paths
    PRIM_VIS_ROOT = "/World/Visuals"
    PRIM_TARGET   = f"{PRIM_VIS_ROOT}/Target"
    PRIM_LOS      = f"{PRIM_VIS_ROOT}/LOS"

    def __init__(self, cfg: QuadcopterHierKineEnvCfg, render_mode: Optional[str] = None, **kwargs):
        """构造环境并初始化关键状态、缓存与可视化开关。

        Args:
            cfg: 环境配置(QuadcopterHierKineEnvCfg)
            render_mode: 渲染模式，透传给父类。
            **kwargs: 其他透传参数。
        """
        super().__init__(cfg, render_mode, **kwargs)

        # === 状态（世界系） ===
        self.pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_w = torch.zeros(self.num_envs, 3, device=self.device)

        # 目标
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)

        # === 分层缓存 ===
        self.hl_decimation = max(1, int(self.cfg.physics_freq // self.cfg.hl_freq))
        self.step_counter = 0

        # 高层“参考”缓存
        self._hl_accel = torch.zeros(self.num_envs, 3, device=self.device)   # for 'accel'
        self._hl_v_des = torch.zeros(self.num_envs, 3, device=self.device)   # for 'vel' / 'wp_rel'
        self._wp_rel   = torch.zeros(self.num_envs, 3, device=self.device)   # for 'wp_rel'

        # 训练缓存与日志
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.success_window = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self.episode_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in ["approach", "dist", "align", "smooth", "success_bonus", "step_cost"]
        }

        # 运行时可视化总开关（可动态切换）
        self._vis_enabled = bool(self.cfg.debug_vis_master)

    def _setup_scene(self) -> None:
        """构建场景：机器人、地形、光源与（可选）可视化基元。"""
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        vis_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.ROBOT_VIS_USD, visible=True)
        for i in range(self.scene.cfg.num_envs):
            vis_path = f"/World/envs/env_{i}/Robot/RobotVis"
            vis_cfg.func(vis_path, vis_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        if self._vis_enabled:
            self._ensure_visual_prims()

    def _ensure_visual_prims(self) -> None:
        """在 USD Stage 上按需创建可视化基元（总开关/子开关生效）。"""
        if not self._vis_enabled:
            return

        stage = self.scene.stage
        if not stage.GetPrimAtPath(self.PRIM_VIS_ROOT):
            UsdGeom.Xform.Define(stage, self.PRIM_VIS_ROOT)

        if self.cfg.debug_vis_target and not stage.GetPrimAtPath(self.PRIM_TARGET):
            cube = UsdGeom.Cube.Define(stage, self.PRIM_TARGET)
            cube.CreateSizeAttr(0.30)
            cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.1, 0.1)]))

        if self.cfg.debug_vis_los and not stage.GetPrimAtPath(self.PRIM_LOS):
            curves = UsdGeom.BasisCurves.Define(stage, self.PRIM_LOS)
            curves.CreateTypeAttr("linear")
            curves.CreateCurveVertexCountsAttr([2])
            curves.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(0, 0, 0)]))
            widths_attr = curves.CreateWidthsAttr()
            widths_attr.Set(Vt.FloatArray([0.02, 0.02]))
            widths_attr.SetMetadata("interpolation", "vertex")
            color_attr = curves.CreateDisplayColorAttr()
            color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 0.0)]))
            color_attr.SetMetadata("interpolation", "constant")

    def _set_prim_pose(self, prim_path: str, p_xyz: Tuple[float, float, float],
                       q_wxyz: Tuple[float, float, float, float]) -> None:
        """设置 USD 基元的位姿（仅显示用途，不影响物理）。

        Args:
            prim_path: 基元路径。
            p_xyz: 平移 (x, y, z)。
            q_wxyz: 四元数 (w, x, y, z)。
        """
        stage = self.scene.stage
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        xform = UsdGeom.Xformable(prim)
        t_op = o_op = None
        for op in xform.GetOrderedXformOps():
            name = op.GetOpName()
            if name == "xformOp:translate:vis":
                t_op = op
            elif name == "xformOp:orient:vis":
                o_op = op
        if t_op is None:
            t_op = xform.AddTranslateOp(opSuffix="vis")
        if o_op is None:
            o_op = xform.AddOrientOp(opSuffix="vis")
        xform.SetXformOpOrder([o_op, t_op])
        t_op.Set(Gf.Vec3d(*map(float, p_xyz)))
        o_op.Set(Gf.Quatf(*map(float, q_wxyz)))

    def _update_visuals_env0(self) -> None:
        """按开关更新 env_0 的目标位置与 LOS 连线。"""
        if not self._vis_enabled:
            return
        idx = 0  # 仅演示 env_0；如需多 env 可扩展

        if self.cfg.debug_vis_target:
            p_tgt = self.target_pos_w[idx].detach().cpu().tolist()
            self._set_prim_pose(self.PRIM_TARGET, p_tgt, (1.0, 0.0, 0.0, 0.0))

        if self.cfg.debug_vis_los:
            los_prim = UsdGeom.BasisCurves.Get(self.scene.stage, self.PRIM_LOS)
            if los_prim:
                p_uav = self.pos_w[idx].detach().cpu().tolist()
                p_tgt = self.target_pos_w[idx].detach().cpu().tolist()
                pts = Vt.Vec3fArray([Gf.Vec3f(*p_uav), Gf.Vec3f(*p_tgt)])
                los_prim.GetPointsAttr().Set(pts)

    # -------------------- Hierarchical interface --------------------
    def _decode_high_level_action(self, actions: torch.Tensor) -> None:
        """将策略输出解码为高层参考(_hl_*),仅在高层步更新。

        行为:
            - accel: 直接映射到加速度参考 _hl_accel。
            - vel  : 映射到期望速度参考 _hl_v_des。
            - wp_rel: 映射到相对航点，再按比例转 v_des 并限幅。
        """
        if (self.step_counter % self.hl_decimation) != 0:
            return

        a = torch.nan_to_num(actions).clamp(-self.cfg.clip_action, self.cfg.clip_action)

        if self.cfg.hl_action_mode == "accel":
            self._hl_accel = self.cfg.a_max * a

        elif self.cfg.hl_action_mode == "vel":
            self._hl_v_des = self.cfg.v_max * a

        elif self.cfg.hl_action_mode == "wp_rel":
            self._wp_rel = self.cfg.wp_rel_max * a
            p_des = self.pos_w + self._wp_rel
            v_des = (p_des - self.pos_w) * self.cfg.wp_to_v_gain
            v_norm = torch.linalg.norm(v_des, dim=-1, keepdim=True).clamp_min(1e-9)
            self._hl_v_des = v_des * torch.clamp(self.cfg.v_max / v_norm, max=1.0)

        else:
            raise ValueError(f"Unknown hl_action_mode: {self.cfg.hl_action_mode}")

    def _low_level_controller(self) -> torch.Tensor:
        """将高层参考转为世界系加速度命令 a_cmd_w,并做幅值限幅。

        Returns:
            (N,3) 世界系加速度命令。
        """
        if self.cfg.hl_action_mode == "accel":
            a_cmd = self._hl_accel
        else:
            v_err = self._hl_v_des - self.vel_w
            a_cmd = self.cfg.ll_kp_v * v_err
            # 若需要，可改为差分近似 D 项；此处用零化处理避免放大噪声
            a_cmd = a_cmd - self.cfg.ll_kd_v * (self.vel_w * 0.0)

        a_norm = torch.linalg.norm(a_cmd, dim=-1, keepdim=True).clamp_min(1e-9)
        a_cmd = a_cmd * torch.clamp(self.cfg.a_max / a_norm, max=1.0)
        return a_cmd

    # -------------------- Gym hooks --------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """物理步前回调：按频率解码策略动作为高层参考。"""
        self._decode_high_level_action(actions)

    def _apply_action(self) -> None:
        """核心推进：目标匀速、低层控制、速度/位置积分、渲染与可视化。

        流程:
            1) 目标以 +X 匀速推进，高度保持。
            2) 低层控制将参考映射为 a_cmd_w。
            3) 对自身速度/位置做积分并限幅，高度保持。
            4) 写回 root pose/velocity 并设置朝向。
            5) 若可视化开启，按 decimation 更新 env_0 的可视要素。
        """
        dt = self.physics_dt

        # 1) 目标运动（+X 匀速）
        self.target_vel_w[:, 0] = self.cfg.target_speed
        self.target_vel_w[:, 1:] = 0.0
        self.target_pos_w += self.target_vel_w * dt
        self.target_pos_w[:, 2] = self.cfg.keep_altitude

        # 2) 低层控制 -> a_cmd_w
        a_cmd_w = self._low_level_controller()

        # 3) 积分（速度/位置，含限幅与高度保持）
        self.vel_w = self.vel_w + a_cmd_w * dt
        speed = torch.linalg.norm(self.vel_w, dim=-1, keepdim=True).clamp_min(1e-9)
        self.vel_w = self.vel_w * torch.clamp(self.cfg.v_max / speed, max=1.0)
        self.pos_w = self.pos_w + self.vel_w * dt
        self.pos_w[:, 2] = self.cfg.keep_altitude

        # 4) 写回渲染（朝向速度方向）
        yaw = torch.atan2(self.vel_w[:, 1], self.vel_w[:, 0])
        quat = torch.stack(
            [torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)],
            dim=-1
        ) # 构造绕 Z 轴的四元数 [w, x, y, z]，只用到 yaw，默认 pitch/roll 为 0
        root_pose = torch.cat([self.pos_w, quat], dim=-1) # 拼成根位姿张量：[x, y, z, q_w, q_x, q_y, q_z]
        root_vel  = torch.cat([self.vel_w, torch.zeros(self.num_envs, 3, device=self.device)], dim=-1) # 拼成根速度张量：[vx, vy, vz, wx, wy, wz]
        self.robot.write_root_pose_to_sim(root_pose, self.robot._ALL_INDICES)
        self.robot.write_root_velocity_to_sim(root_vel, self.robot._ALL_INDICES)

        # 5) 可视化（按开关/频率）
        if self._vis_enabled and (self.step_counter % max(1, int(self.cfg.render_decimation)) == 0):
            self._update_visuals_env0()

        # 6) 步计数
        self.step_counter += 1

    # -------------------- Termination --------------------
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算终止条件。

        Returns:
            out_of_bounds: (N,) 是否平面越界。
            time_out: (N,) 是否到达最大步数。
        """
        origins = self.terrain.env_origins
        dxy2 = torch.sum((self.pos_w[:, :2] - origins[:, :2]) ** 2, dim=-1)
        out_of_bounds = dxy2 > (self.cfg.workspace_radius ** 2)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_of_bounds, time_out

    # -------------------- Reward --------------------
    def _get_rewards(self) -> torch.Tensor:
        """计算一步的奖励并累计日志项。

        奖励项:
            - approach: 距离差分靠近（裁剪）。
            - dist: 距离惩罚。
            - align: 速度朝向与 LOS 对齐（随速度大小加权）。
            - smooth: 动作 L2 正则。
            - success_bonus: 连续命中窗口达标的奖励。
            - step_cost: 步时成本。
        """
        p_rel = self.target_pos_w - self.pos_w
        d = torch.linalg.norm(p_rel, dim=-1).clamp_min(1e-6)
        los = p_rel / d.unsqueeze(-1)

        # 1) 差分靠近
        delta_d = (self.prev_distance - d).clamp(-self.cfg.approach_clip, self.cfg.approach_clip)
        r_approach = self.cfg.approach_weight * delta_d

        # 2) 距离惩罚
        r_dist = -self.cfg.dist_weight * d

        # 3) 速度朝向对齐
        v_norm = torch.linalg.norm(self.vel_w, dim=-1).clamp_min(1e-6)
        v_hat = self.vel_w / v_norm.unsqueeze(-1)
        align = torch.sum(v_hat * los, dim=-1)
        speed_gain = torch.clamp(v_norm / self.cfg.v_max, max=1.0)
        r_align = self.cfg.align_weight * align * speed_gain

        # 4) 平滑/能耗
        r_smooth = -self.cfg.smooth_action_weight * torch.sum(
            torch.nan_to_num(self.prev_actions) ** 2, dim=-1
        )

        # 5) 成功窗口
        success_now = d < self.cfg.success_pos_tol
        self.success_window = torch.where(success_now, self.success_window + 1, torch.zeros_like(self.success_window))
        hit_success = self.success_window >= self.cfg.success_hold_steps
        r_success = torch.where(hit_success, torch.full_like(d, self.cfg.success_bonus), torch.zeros_like(d))
        self.success_window = torch.where(hit_success, torch.zeros_like(self.success_window), self.success_window)

        # 6) 步时
        r_step = -torch.full_like(d, self.cfg.step_cost)

        terms = {
            "approach": r_approach,
            "dist": r_dist,
            "align": r_align,
            "smooth": r_smooth,
            "success_bonus": r_success,
            "step_cost": r_step,
        }
        for k, v in terms.items():
            self.episode_sums[k] += v

        # 缓存
        self.prev_distance = d.detach()
        return torch.sum(torch.stack(list(terms.values())), dim=0)

    # -------------------- Observations --------------------
    def _get_observations(self) -> dict:
        """构造观测字典。

        Returns:
            dict:
              - "policy": (N,9) 位置(3)+速度(3)+LOS(3)
              - "odom":  (N,13) root pose/vel,便于外部可视化或记录
        """
        p_rel = self.target_pos_w - self.pos_w
        d = torch.linalg.norm(p_rel, dim=-1, keepdim=True).clamp_min(1e-6)
        los = p_rel / d
        obs = torch.cat([self.pos_w, self.vel_w, los], dim=-1)  # [N,9]

        quat_id = torch.tensor([1., 0., 0., 0.], device=self.device).repeat(self.num_envs, 1)
        root_state = torch.cat([self.pos_w, quat_id, self.vel_w, torch.zeros(self.num_envs, 3, device=self.device)], dim=-1)
        return {"policy": obs, "odom": root_state}

    # -------------------- Reset --------------------
    def _reset_idx(self, env_ids: Optional[torch.Tensor]) -> None:
        """按 env_ids 重置子环境;env_ids 为 None 或全量时重置全部。

        行为:
            - 记录并清零 episode_sums 对应 env 的统计到 self.extras["log"]。
            - 放置 UAV/目标 初始位姿与速度，重置分层/训练缓存。
            - 按需刷新一次可视化。
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # 日志
        extras = {f"Episode_Reward/{k}": torch.mean(self.episode_sums[k][env_ids]) for k in self.episode_sums}
        for k in self.episode_sums:
            self.episode_sums[k][env_ids] = 0.0
        self.extras["log"] = {**self.extras.get("log", {}), **extras}

        # 初始状态
        origins = self.terrain.env_origins[env_ids]
        self.pos_w[env_ids] = origins
        self.pos_w[env_ids, 2] = self.cfg.keep_altitude
        self.vel_w[env_ids] = 0.0

        # 目标
        self.target_pos_w[env_ids] = origins
        self.target_pos_w[env_ids, 0] += 2.0
        self.target_pos_w[env_ids, 2] = self.cfg.keep_altitude
        self.target_vel_w[env_ids] = 0.0
        self.target_vel_w[env_ids, 0] = self.cfg.target_speed

        # 分层缓存
        self._hl_accel[env_ids] = 0.0
        self._hl_v_des[env_ids] = 0.0
        self._wp_rel[env_ids]   = 0.0
        self.step_counter = 0

        # 训练缓存
        self.prev_actions[env_ids] = 0.0
        rel = self.target_pos_w - self.pos_w
        self.prev_distance[env_ids] = torch.linalg.norm(rel[env_ids], dim=-1)
        self.success_window[env_ids] = 0

        # 渲染
        yaw = torch.zeros(len(env_ids), device=self.device)
        quat = torch.stack([torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)], dim=-1)
        root_pose = torch.cat([self.pos_w[env_ids], quat], dim=-1)
        root_vel  = torch.cat([self.vel_w[env_ids], torch.zeros(len(env_ids), 3, device=self.device)], dim=-1)
        self.robot.write_root_pose_to_sim(root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids)

        # 可视化一次
        if self._vis_enabled:
            self._ensure_visual_prims()
            if env_ids[0].item() == 0:  # 简单处理 env_0
                self._update_visuals_env0()

    # -------------------- Runtime visual toggle --------------------
    def set_visuals_enabled(self, enabled: bool) -> None:
        """运行时切换可视化总开关；开启时会确保基元已创建。"""
        self._vis_enabled = bool(enabled)
        if self._vis_enabled:
            self._ensure_visual_prims()


# ===============================
# Gym registration
# ===============================
from config import agents
gym.register(
    id="FAST-Tracking",
    entry_point=QuadcopterHierKineEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterHierKineEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:quadcopter_sb3_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:quadcopter_tracking_skrl_ppo_cfg.yaml",
    },
)
