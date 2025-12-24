from __future__ import annotations

import math
import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import CUBOID_MARKER_CFG, GIMBAL_RAY_MARKER_CFG
import isaaclab.sim as sim_utils


def _quat_wxyz_from_z_to_dir(d: torch.Tensor) -> torch.Tensor:
    """把局部 z 轴旋到方向 d(世界系单位向量)，返回 (w,x,y,z)"""
    d = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-8)
    z = torch.tensor([0.0, 0.0, 1.0], device=d.device, dtype=d.dtype).expand_as(d)
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


def _dir_from_yaw_pitch(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """从yaw和pitch计算方向向量（z-up坐标系）
    
    Args:
        yaw: 偏航角 [...,]
        pitch: 俯仰角 [...,]
    
    Returns:
        方向向量 [..., 3]
    """
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    return torch.stack([cp * cy, cp * sy, sp], dim=-1)


class VisualizationHelper:
    """可视化辅助类，用于管理环境中的各种可视化功能"""
    
    def __init__(self, env):
        """初始化可视化辅助类
        
        Args:
            env: 环境实例，需要提供以下属性和方法：
                - cfg: 配置对象
                - device: 设备
                - num_envs: 环境数量
                - M: 友机数量
                - E: 敌机数量
                - friend_frozen: [N, M] 友机冻结状态
                - fr_pos: [N, M, 3] 友机位置
                - _gimbal_yaw: [N, M] 云台偏航角
                - _gimbal_pitch: [N, M] 云台俯仰角
                - _dir_from_yaw_pitch: 方法，从yaw/pitch计算方向
                - _traj_buf: [N, M, K, 3] 轨迹缓冲区
                - _traj_len: [N, M] 轨迹长度
                - _sim_step_counter: 仿真步计数器
                - enemy_pos: [N, E, 3] 敌机位置
                - _enemy_exists_mask: [N, E] 敌机存在掩码
                - _dbg_vis_fe: [N, M, E] 可见性掩码
                - _dbg_est_pos_world: [N, E, 3] 估计位置
        """
        self.env = env
        self._gimbal_fov_ray_marker = None
        self._bearing_ray_markers = []
        self._bearing_est_marker = None
        self._traj_markers = []
        self._claim_markers = []
    
    @torch.no_grad()
    def update_gimbal_fov_vis(self):
        """用圆柱射线绘制云台四棱锥 FOV 线框(4 条侧边 + 远平面矩形）"""
        if not getattr(self.env.cfg, "gimbal_vis_enable", False):
            return

        # === 确保 marker ===
        if not hasattr(self, "_gimbal_fov_ray_marker") or self._gimbal_fov_ray_marker is None:
            self._gimbal_fov_ray_marker = VisualizationMarkers(GIMBAL_RAY_MARKER_CFG)
            self._gimbal_fov_ray_marker.set_visibility(True)
            # 如果环境也有这个属性，同步引用
            if hasattr(self.env, "_gimbal_fov_ray_marker"):
                self.env._gimbal_fov_ray_marker = self._gimbal_fov_ray_marker

        dev = self.env.device
        Ndraw = int(min(self.env.num_envs, getattr(self.env.cfg, "gimbal_vis_max_envs", 4)))
        if Ndraw <= 0:
            return

        # === 参数 ===
        half_h = 0.5 * math.radians(float(self.env.cfg.gimbal_fov_h_deg))   # 水平半角
        half_v = 0.5 * math.radians(float(self.env.cfg.gimbal_fov_v_deg))   # 垂直半角
        R      = float(getattr(self.env.cfg, "gimbal_effective_range", 40.0))

        all_mid = []
        all_quat = []
        all_scale = []

        for ei in range(Ndraw):
            active = (~self.env.friend_frozen[ei])
            if not torch.any(active):
                continue

            # 顶点与角度
            P = self.env.fr_pos[ei][active]          # [S,3]
            Y = self.env._gimbal_yaw[ei][active]     # [S]
            T = self.env._gimbal_pitch[ei][active]   # [S]
            S = P.shape[0]
            if S == 0:
                continue

            # 四个角方向 (±half_h, ±half_v) → 远平面四角
            yaws   = torch.stack([Y - half_h, Y - half_h, Y + half_h, Y + half_h], dim=1)  # [S,4]
            pitchs = torch.stack([T - half_v, T + half_v, T - half_v, T + half_v], dim=1)  # [S,4]
            dirs4  = _dir_from_yaw_pitch(yaws, pitchs)                                 # [S,4,3]
            corners = P[:, None, :] + R * dirs4                                             # [S,4,3]

            # 可选：中心轴线
            if getattr(self.env.cfg, "gimbal_axis_vis_enable", True):
                dir_c = _dir_from_yaw_pitch(Y, T)         # [S,3]
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

    def _ensure_bearing_markers(self):
        """确保bearing标记已创建"""
        if self._bearing_ray_markers:
            return
        # 给不同友机配几种区分度高的颜色
        palette = [
            (1.00, 0.00, 0.00),  # 红
            (0.00, 1.00, 0.00),  # 绿
            (0.00, 0.00, 1.00),  # 蓝
            (1.00, 1.00, 0.00),  # 黄
            (1.00, 0.00, 1.00),  # 品红
            (0.00, 1.00, 1.00),  # 青
            (1.00, 0.50, 0.00),  # 橙
            (0.50, 0.00, 1.00),  # 紫
        ]
        for i in range(self.env.M):
            cfg = GIMBAL_RAY_MARKER_CFG.copy()
            cfg.prim_path = f"/Visuals/Bearing/Friend{i}"
            # 修改颜色
            for k in cfg.markers:
                cfg.markers[k].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=palette[i % len(palette)]
                )
            mk = VisualizationMarkers(cfg)
            mk.set_visibility(True)
            self._bearing_ray_markers.append(mk)

    @torch.no_grad()
    def update_bearing_vis(self):
        """可视化前几个 env、前几个友机的 bearing 射线和估计点（调试用）"""
        if not getattr(self.env.cfg, "bearing_vis_enable", False):
            return
        if not hasattr(self.env, "_dbg_bearings"):
            return

        self._ensure_bearing_markers()

        if self._bearing_est_marker is None:
            # 修改：将估计点改为方块（CUBOID_MARKER_CFG）
            cfg = CUBOID_MARKER_CFG.copy()
            cfg.prim_path = "/Visuals/Bearing/Estimates"
            cfg.markers["cuboid"].size = (0.5, 0.5, 0.5)  # 稍微设置大一点方便观察
            self._bearing_est_marker = VisualizationMarkers(cfg)
            self._bearing_est_marker.set_visibility(True)

        dev = self.env.device
        dtype = self.env.fr_pos.dtype

        # 实时计算当前 bearing 方向，确保射线紧跟友机移动
        rel_all = self.env.enemy_pos.unsqueeze(1) - self.env.fr_pos.unsqueeze(2)  # [N,M,E,3]
        dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(1e-9)
        curr_bearings = rel_all / dist_all
        
        vis_fe = self.env._dbg_vis_fe      # [N,M,E] 可见性仍使用最后一次观测的缓存
        est_pos_world = self.env._dbg_est_pos_world  # [N,E,3] 估计位置使用最后一次滤波结果

        Ndraw = int(min(self.env.num_envs, getattr(self.env.cfg, "bearing_vis_max_envs", 1)))
        Mf = int(getattr(self.env.cfg, "bearing_vis_num_friends", 2))
        Me = int(getattr(self.env.cfg, "bearing_vis_num_enemies", 3))
        L = float(getattr(self.env.cfg, "bearing_vis_length", 120.0))

        all_pts = []

        # 遍历每个友机
        for f_idx in range(self.env.M):
            f_mid = []
            f_quat = []
            f_scale = []
            
            # 如果该友机不在可视化范围内，或者已冻结，则清空其可视化
            is_visible_friend = (f_idx < Mf)
            
            if is_visible_friend:
                for ei in range(Ndraw):
                    if self.env.friend_frozen[ei, f_idx]:
                        continue
                    
                    # 敌机子集
                    exists = self.env._enemy_exists_mask[ei]
                    enemy_idx = torch.arange(self.env.E, device=dev)[exists][:Me]
                    
                    for e in enemy_idx:
                        if not vis_fe[ei, f_idx, e]:
                            continue
                        origin = self.env.fr_pos[ei, f_idx]        # [3]
                        dir_fe = curr_bearings[ei, f_idx, e]   # 使用实时计算的方向
                        dir_norm = dir_fe / dir_fe.norm().clamp_min(1e-8)
                        tip = origin + L * dir_norm
                        mid = origin + 0.5 * (tip - origin)
                        quat = _quat_wxyz_from_z_to_dir(dir_norm.unsqueeze(0)).squeeze(0)
                        scale = torch.tensor([1.0, 1.0, L], device=dev, dtype=dtype)

                        f_mid.append(mid.unsqueeze(0))
                        f_quat.append(quat.unsqueeze(0))
                        f_scale.append(scale.unsqueeze(0))

            if f_mid:
                self._bearing_ray_markers[f_idx].set_visibility(True)
                self._bearing_ray_markers[f_idx].visualize(
                    translations=torch.cat(f_mid, dim=0),
                    orientations=torch.cat(f_quat, dim=0),
                    scales=torch.cat(f_scale, dim=0)
                )
            else:
                # 隐藏该友机的射线，避免 ValueError
                self._bearing_ray_markers[f_idx].set_visibility(False)

        # 估计点可视化
        for ei in range(Ndraw):
            exists = self.env._enemy_exists_mask[ei]
            enemy_idx = torch.arange(self.env.E, device=dev)[exists][:Me]
            if enemy_idx.numel() > 0:
                pts = est_pos_world[ei, enemy_idx]  # [Me,3]
                all_pts.append(pts)

        if all_pts:
            pts_cat = torch.cat(all_pts, dim=0)
            if pts_cat.shape[0] > 0:
                self._bearing_est_marker.set_visibility(True)
                self._bearing_est_marker.visualize(translations=pts_cat)
            else:
                self._bearing_est_marker.set_visibility(False)
        elif self._bearing_est_marker is not None:
            self._bearing_est_marker.set_visibility(False)

        # ========== Claim 可视化：显示每个友机声明的 track 位置 ==========
        if hasattr(self.env, "intent_claimed_track"):
            self._ensure_claim_markers()
            Ndraw = int(min(self.env.num_envs, getattr(self.env.cfg, "bearing_vis_max_envs", 1)))
            for mi in range(self.env.M):
                pts_list = []
                for ei in range(Ndraw):
                    # 跳过冻结的友机
                    if self.env.friend_frozen[ei, mi]:
                        continue
                    # claimed slot (pseudo-track slot) for this friend in this env
                    try:
                        slot = int(self.env.intent_claimed_track[ei, mi].item())
                    except Exception:
                        slot = int(self.env.intent_claimed_track[ei, mi])
                    if slot < 0:
                        continue
                    # ensure slot exists in global track_pos and is alive
                    if slot < self.env.track_pos.shape[1] and self.env.track_alive[ei, slot]:
                        pos = self.env.track_pos[ei, slot]  # [3]
                        pts_list.append(pos.unsqueeze(0))
                if pts_list:
                    self._claim_markers[mi].set_visibility(True)
                    self._claim_markers[mi].visualize(translations=torch.cat(pts_list, dim=0))
                else:
                    # 隐藏该友机的声明标记
                    self._claim_markers[mi].set_visibility(False)
    def _ensure_traj_markers(self):
        """确保轨迹标记已创建"""
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
        for i in range(self.env.M):
            cfg = CUBOID_MARKER_CFG.copy()
            cfg.prim_path = f"/Visuals/Traj/Friend{i}"
            cfg.markers["cuboid"].size = tuple(getattr(self.env.cfg, "traj_marker_size", (0.05, 0.05, 0.05)))
            cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=palette[i % len(palette)]
            )
            mk = VisualizationMarkers(cfg)
            mk.set_visibility(True)
            self._traj_markers.append(mk)

    def _ensure_claim_markers(self):
        """确保 claim 标记已创建（显示哪个friend声明了哪个 track）"""
        if self._claim_markers:
            return
        palette = [
            (1.00, 0.00, 0.00),  # 红
            (0.00, 1.00, 0.00),  # 绿
            (0.00, 0.00, 1.00),  # 蓝
            (1.00, 1.00, 0.00),  # 黄
            (1.00, 0.00, 1.00),  # 品红
            (0.00, 1.00, 1.00),  # 青
            (1.00, 0.50, 0.00),  # 橙
            (0.50, 0.00, 1.00),  # 紫
        ]
        for i in range(self.env.M):
            cfg = CUBOID_MARKER_CFG.copy()
            cfg.prim_path = f"/Visuals/Claim/Friend{i}"
            # claim marker slightly larger
            cfg.markers["cuboid"].size = (0.6, 0.6, 0.6)
            cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=palette[i % len(palette)]
            )
            mk = VisualizationMarkers(cfg)
            mk.set_visibility(True)
            self._claim_markers.append(mk)

    @torch.no_grad()
    def update_traj_vis(self):
        """记录并绘制友方轨迹（仅前 K 个 env；按步抽样）。"""
        if not getattr(self.env.cfg, "traj_vis_enable", False):
            return
        # 步抽样
        stride = int(getattr(self.env.cfg, "traj_vis_every_n_steps", 1))
        if stride > 1 and ((self.env._sim_step_counter - 1) % stride != 0):
            return

        N_draw = int(min(self.env.num_envs, getattr(self.env.cfg, "traj_vis_max_envs", 4)))
        if N_draw <= 0:
            return
        self._ensure_traj_markers()

        K = int(getattr(self.env.cfg, "traj_vis_len", 200))

        # 1) 先把当前点写入循环缓冲
        for ei in range(N_draw):
            for mi in range(self.env.M):
                w = int(self.env._traj_len[ei, mi].item() % K)
                self.env._traj_buf[ei, mi, w, :] = self.env.fr_pos[ei, mi]
                self.env._traj_len[ei, mi] += 1

        # 2) 为每个友机收集其在前 N_draw 个 env 的历史点并下发 marker
        for mi in range(self.env.M):
            pts_list = []
            for ei in range(N_draw):
                n = int(self.env._traj_len[ei, mi].item())
                if n == 0:
                    continue
                L = min(n, K)
                buf = self.env._traj_buf[ei, mi, :K, :]  # [K,3]
                if n <= K:
                    pts = buf[:L, :]
                else:
                    # 循环缓冲展开为时间顺序（oldest -> newest）
                    start = n % K
                    pts = torch.cat([buf[start:K, :], buf[0:start, :]], dim=0)
                pts_list.append(pts)
            if pts_list:
                self._traj_markers[mi].visualize(translations=torch.cat(pts_list, dim=0))

