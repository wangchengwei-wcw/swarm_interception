import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class BearingOnlyEstimator:

    def __init__(
        self,
        min_observers: int = 2,
        max_observers: int = 5,
        estimation_method: str = "triangulation",
        uncertainty_threshold: float = 0.5,
    ):
        """
        初始化估计器
        
        Args:
            min_observers: 进行估计所需的最少观测者数量
            max_observers: 参与估计的最大观测者数量（超过则选择最优组合）
            estimation_method: 估计方法 ("triangulation" 或 "least_squares")
            uncertainty_threshold: 不确定性阈值，超过此值认为估计不可靠
        """
        self.min_observers = min_observers
        self.max_observers = max_observers
        self.estimation_method = estimation_method
        self.uncertainty_threshold = uncertainty_threshold
    
    def estimate_target_position(
        self,
        observer_positions: torch.Tensor,  # [N, M, 3] 或 [M, 3] 观测者位置
        bearings: torch.Tensor,            # [N, M, 3] 或 [M, 3] 指向目标的单位向量
        observer_mask: Optional[torch.Tensor] = None,  # [N, M] 或 [M] 哪些观测者有效
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        估计目标位置
        
        Args:
            observer_positions: 观测者位置 [N, M, 3] 或 [M, 3]
            bearings: 指向目标的单位向量 [N, M, 3] 或 [M, 3]
            observer_mask: 哪些观测者有效 [N, M] 或 [M]，True 表示有效
        
        Returns:
            estimated_pos: 估计的目标位置 [N, 3] 或 [3]
            uncertainty: 估计不确定性（0-1，越小越可靠）[N] 或 scalar
            num_observers: 参与估计的观测者数量 [N] 或 scalar
        """
        # 处理维度
        if observer_positions.dim() == 2:
            observer_positions = observer_positions.unsqueeze(0)
            bearings = bearings.unsqueeze(0)
            if observer_mask is not None:
                observer_mask = observer_mask.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        N, M, _ = observer_positions.shape
        
        # 默认所有观测者都有效
        if observer_mask is None:
            observer_mask = torch.ones(N, M, dtype=torch.bool, device=observer_positions.device)
        
        # 检查 bearing 是否为零向量（无效观测）
        bearing_norm = torch.linalg.norm(bearings, dim=-1)
        valid_bearing = (bearing_norm > 1e-6) & observer_mask
        
        # 向量化处理：对每个环境选择前 max_observers 个有效观测者
        # 计算每个环境的有效观测者数量
        num_valid_per_env = valid_bearing.sum(dim=-1)  # [N]
        
        # 为每个环境选择前 max_observers 个有效观测者
        # 使用 argsort 找到有效观测者的索引
        valid_bearing_int = valid_bearing.int()  # [N, M]
        # 创建排序键：有效观测者为1，无效为0，然后按索引排序
        # 使用索引作为次要排序键，确保相同有效性的观测者按索引排序
        sort_key = valid_bearing_int * (M + 1) - torch.arange(M, device=observer_positions.device).unsqueeze(0).float()
        sorted_indices = sort_key.argsort(dim=-1, descending=True)  # [N, M]
        
        # 选择前 max_observers 个，但确保索引唯一
        # 对于每个环境，选择前 max_observers 个不同的有效观测者
        selected_indices_list = []
        for n in range(N):
            env_indices = []
            seen = set()
            for idx in sorted_indices[n]:
                if idx.item() not in seen and valid_bearing[n, idx]:
                    env_indices.append(idx.item())
                    seen.add(idx.item())
                    if len(env_indices) >= self.max_observers:
                        break
            # 如果有效观测者不足，用-1填充（后续会被过滤）
            while len(env_indices) < self.max_observers:
                env_indices.append(-1)
            selected_indices_list.append(env_indices)
        
        selected_indices = torch.tensor(selected_indices_list, device=observer_positions.device, dtype=torch.long)  # [N, max_observers]
        
        # 提取选中的观测者数据
        # 处理填充的-1索引：对于无效索引，使用0索引（后续会被掩码过滤）
        selected_indices_safe = torch.clamp(selected_indices, 0, M - 1)  # [N, max_observers]
        gather_idx = selected_indices_safe.unsqueeze(-1).expand(-1, -1, 3)  # [N, max_observers, 3]
        obs_pos_selected = torch.gather(observer_positions, 1, gather_idx)  # [N, max_observers, 3]
        obs_bearings_selected = torch.gather(bearings, 1, gather_idx)  # [N, max_observers, 3]
        
        # 创建有效性掩码：标记哪些观测者真正有效（索引有效且bearing有效）
        valid_mask_selected = (selected_indices >= 0) & torch.gather(valid_bearing, 1, selected_indices_safe)  # [N, max_observers]
        num_valid_selected = valid_mask_selected.sum(dim=-1)  # [N]
        
        # 检查哪些环境有足够的观测者
        has_enough_observers = num_valid_selected >= self.min_observers  # [N]
        
        # 批量执行估计（向量化版本）
        if self.estimation_method == "triangulation":
            est_pos, uncertainty = self._triangulation_estimate_batch(
                obs_pos_selected, obs_bearings_selected, valid_mask_selected, has_enough_observers
            )
        elif self.estimation_method == "least_squares":
            est_pos, uncertainty = self._least_squares_estimate_batch(
                obs_pos_selected, obs_bearings_selected, valid_mask_selected, has_enough_observers
            )
        else:
            raise ValueError(f"Unknown estimation method: {self.estimation_method}")
        
        # 对于观测者不足的环境，设置为无效估计
        estimated_pos = torch.where(has_enough_observers.unsqueeze(-1), est_pos, torch.zeros_like(est_pos))
        uncertainty = torch.where(has_enough_observers, uncertainty, torch.ones_like(uncertainty))
        num_observers = torch.where(has_enough_observers, num_valid_selected, torch.zeros_like(num_valid_selected))
        
        if squeeze_output:
            estimated_pos = estimated_pos.squeeze(0)
            uncertainty = uncertainty.squeeze(0)
            num_observers = num_observers.squeeze(0)
        
        return estimated_pos, uncertainty, num_observers
    
    def _triangulation_estimate(
        self,
        observer_positions: torch.Tensor,  # [K, 3]
        bearings: torch.Tensor,            # [K, 3]
        max_range: float = 100.0,          # clamp very far solutions (meters)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        K = observer_positions.shape[0]
        device = observer_positions.device
        dtype = observer_positions.dtype
        
        if K == 2:
            # 两个观测者的三角定位
            p1, p2 = observer_positions[0], observer_positions[1]
            d1, d2 = bearings[0], bearings[1]
            
            # 验证 bearing 向量是否为单位向量
            d1_norm = torch.linalg.norm(d1)
            d2_norm = torch.linalg.norm(d2)
            
            if d1_norm < 1e-6 or d2_norm < 1e-6:
                # bearing 向量无效
                return torch.zeros(3, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
            
            # 归一化 bearing 向量（防止数值误差）
            d1 = d1 / d1_norm
            d2 = d2 / d2_norm
            
            # 计算两条射线的最近点（最小二乘解）
            # 射线1: p1 + t1 * d1
            # 射线2: p2 + t2 * d2
            # 目标：min ||(p1 + t1*d1) - (p2 + t2*d2)||^2
            # 即：min ||t1*d1 - t2*d2 - (p2 - p1)||^2
            
            # 构建线性系统：A * [t1; t2] = b
            # A = [d1, -d2] (3x2矩阵)
            # b = p2 - p1
            A = torch.stack([d1, -d2], dim=1)  # [3, 2]
            b = p2 - p1  # [3]
            
            # 最小二乘解：[t1; t2] = (A^T * A)^(-1) * A^T * b
            ATA = A.T @ A  # [2, 2]
            ATb = A.T @ b  # [2]
            
            # 检查矩阵是否可逆（两条射线不能平行）
            det = ATA[0, 0] * ATA[1, 1] - ATA[0, 1] * ATA[1, 0]
            
            if abs(det) < 1e-6:
                # 两条射线几乎平行，无法定位
                return torch.zeros(3, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
            
            # 求解 [t1; t2]
            ATA_inv = torch.inverse(ATA)
            t_vec = ATA_inv @ ATb  # [2]
            t1, t2 = t_vec[0], t_vec[1]
            
            # 估计位置：两条射线的中点（更稳定）
            pos1 = p1 + t1 * d1
            pos2 = p2 + t2 * d2
            est_pos = (pos1 + pos2) / 2.0
            
            # 不确定性：两条射线的距离（距离越大，不确定性越高）
            ray_distance = torch.linalg.norm(pos1 - pos2)
            uncertainty = torch.clamp(ray_distance / 100.0, 0.0, 1.0)

            # 远距离解直接截断并视为无效
            if torch.linalg.norm(est_pos) > max_range:
                return torch.zeros(3, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)

            return est_pos, uncertainty
        
        else:
            # K > 2：使用所有两两组合的平均值
            all_estimates = []
            all_uncertainties = []
            filtered_reasons = {"too_close": 0, "bad_angle": 0}
            
            for i in range(K):
                for j in range(i + 1, K):
                    p1, p2 = observer_positions[i], observer_positions[j]
                    d1, d2 = bearings[i], bearings[j]
                    
                    # 检查两个观测者之间的距离（太近的观测者对估计不利）
                    obs_distance = torch.linalg.norm(p2 - p1)
                    if obs_distance < 1e-2:  # 观测者太近（放宽到1cm），跳过
                        filtered_reasons["too_close"] += 1
                        continue
                    
                    # 检查两个 bearing 向量的夹角（太小的夹角对估计不利）
                    cos_angle = torch.dot(d1, d2)
                    angle = torch.acos(torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6))
                    min_angle = 0.01  # 最小夹角（弧度），约0.57度（放宽条件）
                    if angle < min_angle or angle > (math.pi - min_angle):  # 几乎平行或反平行
                        filtered_reasons["bad_angle"] += 1
                        continue
                    
                    # 调用两观测者版本
                    est_pos, uncertainty = self._triangulation_estimate(
                        torch.stack([p1, p2]), torch.stack([d1, d2]), max_range=max_range
                    )
                    
                    # 只保留有效估计，并且检查估计位置是否合理
                    if uncertainty >= 1.0:
                        continue
                    
                    if torch.isnan(est_pos).any():
                        continue
                    
                    # 检查估计位置是否在合理范围内（放宽距离限制）
                    max_reasonable_dist = 10000.0  # 最大合理距离（米），放宽到10km
                    dist_to_obs1 = torch.linalg.norm(est_pos - p1)
                    dist_to_obs2 = torch.linalg.norm(est_pos - p2)
                    # 同时检查距离不能太小（至少1米）
                    min_reasonable_dist = 1.0
                    if not (dist_to_obs1 < max_reasonable_dist and dist_to_obs2 < max_reasonable_dist and
                            dist_to_obs1 > min_reasonable_dist and dist_to_obs2 > min_reasonable_dist):
                        continue
                    
                    all_estimates.append(est_pos)
                    all_uncertainties.append(uncertainty)
            
            if len(all_estimates) == 0:
                return torch.zeros(3, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
            
            # 加权平均（不确定性小的权重更大）
            all_estimates = torch.stack(all_estimates)  # [num_pairs, 3]
            all_uncertainties = torch.stack(all_uncertainties)  # [num_pairs]
            
            # 使用更稳健的加权策略：优先使用不确定性较小的估计
            # 放宽不确定性阈值，允许更多估计参与
            max_uncertainty = 0.9  # 最大可接受的不确定性（放宽到0.9）
            good_estimates = all_uncertainties < max_uncertainty
            if good_estimates.sum() > 0:
                good_est = all_estimates[good_estimates]
                good_unc = all_uncertainties[good_estimates]
                weights = 1.0 / (good_unc + 1e-6)
                weights = weights / weights.sum()
                est_pos = (good_est * weights.unsqueeze(-1)).sum(dim=0)
                avg_uncertainty = good_unc.mean()
            else:
                # 如果没有好的估计，使用所有估计的平均值（即使不确定性较大）
                weights = 1.0 / (all_uncertainties + 1e-6)
                weights = weights / weights.sum()
                est_pos = (all_estimates * weights.unsqueeze(-1)).sum(dim=0)
                avg_uncertainty = all_uncertainties.mean()

            # clamp if too far
            if torch.linalg.norm(est_pos) > max_range:
                return torch.zeros(3, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)

            return est_pos, avg_uncertainty
    
    def _triangulation_estimate_batch(
        self,
        observer_positions: torch.Tensor,  # [N, K, 3]
        bearings: torch.Tensor,            # [N, K, 3]
        valid_mask: torch.Tensor,          # [N, K] 哪些观测者有效
        has_enough_observers: torch.Tensor,  # [N] 哪些环境有足够观测者
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, K, _ = observer_positions.shape
        device = observer_positions.device
        dtype = observer_positions.dtype
        
        # 初始化输出
        estimated_positions = torch.zeros(N, 3, device=device, dtype=dtype)
        uncertainties = torch.ones(N, device=device, dtype=dtype)
        
        # 对每个环境分别处理，避免复杂的批量维度问题
        for n in range(N):
            if not has_enough_observers[n]:
                continue
            
            # 提取当前环境的有效观测者
            valid_obs = valid_mask[n]  # [K]
            valid_indices = torch.where(valid_obs)[0]  # [num_valid]
            num_valid = valid_indices.shape[0]
            
            if num_valid < self.min_observers:
                continue
            
            # 提取有效观测者的位置和 bearing
            obs_pos_valid = observer_positions[n, valid_indices]  # [num_valid, 3]
            bearings_valid = bearings[n, valid_indices]  # [num_valid, 3]
            
            # 检查 bearing 向量是否为零向量（无效观测）
            bearing_norms = torch.linalg.norm(bearings_valid, dim=-1)  # [num_valid]
            valid_bearing_mask = bearing_norms > 1e-6  # [num_valid]
            
            if valid_bearing_mask.sum() < self.min_observers:
                # 有效bearing向量不足，跳过
                continue
            
            # 只保留有效的bearing向量
            valid_bearing_indices = torch.where(valid_bearing_mask)[0]
            obs_pos_final = obs_pos_valid[valid_bearing_indices]  # [num_final, 3]
            bearings_final = bearings_valid[valid_bearing_indices]  # [num_final, 3]
            
            # 使用单目标版本的估计方法（处理 K > 2 的情况）
            est_pos, uncertainty = self._triangulation_estimate(obs_pos_final, bearings_final)
            
            # 如果估计有效，保存结果
            if uncertainty < 1.0 and not torch.isnan(est_pos).any():
                estimated_positions[n] = est_pos
                uncertainties[n] = uncertainty
        
        return estimated_positions, uncertainties
    
    def _least_squares_estimate(
        self,
        observer_positions: torch.Tensor,  # [K, 3]
        bearings: torch.Tensor,            # [K, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用最小二乘法估计目标位置（适用于 K >= 2）
        
        目标：找到点 x，使得 x 到所有射线的距离平方和最小
        射线 i: p_i + t_i * d_i
        
        Args:
            observer_positions: 观测者位置 [K, 3]
            bearings: 指向目标的单位向量 [K, 3]
        
        Returns:
            estimated_position: 估计位置 [3]
            uncertainty: 不确定性 (0-1)
        """
        K = observer_positions.shape[0]
        device = observer_positions.device
        dtype = observer_positions.dtype
        
        if K < 2:
            return torch.zeros(3, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
        
        # 构建最小二乘问题
        # 对于每个观测者 i，射线为 p_i + t_i * d_i
        # 目标点 x 到射线 i 的距离：||(x - p_i) - ((x - p_i) · d_i) * d_i||
        # 最小化 sum_i ||(x - p_i) - ((x - p_i) · d_i) * d_i||^2
        
        # 简化：使用迭代优化或解析解
        # 这里使用简化的解析解：假设目标在观测者质心附近
        
        # 初始化：使用三角定位的结果（如果K=2）或观测者质心
        if K == 2:
            # 先用三角定位得到初始估计
            x_init, _ = self._triangulation_estimate(observer_positions, bearings)
        else:
            # 使用前两个观测者的三角定位结果作为初始值
            x_init, _ = self._triangulation_estimate(
                observer_positions[:2], bearings[:2]
            )
            # 如果三角定位失败，使用观测者质心
            if torch.isnan(x_init).any() or (x_init.norm() > 1e6):
                x_init = observer_positions.mean(dim=0)
        
        # 使用梯度下降优化
        x = x_init.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=0.5)
        
        for _ in range(50):  # 增加迭代次数以提高精度
            optimizer.zero_grad()
            
            # 计算到每条射线的距离
            residuals = []
            for i in range(K):
                p_i = observer_positions[i]
                d_i = bearings[i]
                
                # 点 x 到射线 p_i + t * d_i 的距离
                vec_to_x = x - p_i
                proj = torch.dot(vec_to_x, d_i) * d_i
                residual = vec_to_x - proj
                residuals.append(residual)
            
            # 损失：所有距离的平方和
            loss = sum(r.norm() ** 2 for r in residuals)
            loss.backward()
            optimizer.step()
        
        est_pos = x.detach()
        
        # 计算不确定性：估计位置到各射线的平均距离
        distances = []
        for i in range(K):
            p_i = observer_positions[i]
            d_i = bearings[i]
            vec_to_x = est_pos - p_i
            proj = torch.dot(vec_to_x, d_i) * d_i
            residual = vec_to_x - proj
            distances.append(residual.norm())
        
        avg_distance = torch.stack(distances).mean()
        uncertainty = torch.clamp(avg_distance / 50.0, 0.0, 1.0)  # 归一化
        
        return est_pos, uncertainty
    
    def _least_squares_estimate_batch(
        self,
        observer_positions: torch.Tensor,  # [N, K, 3]
        bearings: torch.Tensor,            # [N, K, 3]
        valid_mask: torch.Tensor,          # [N, K] 哪些观测者有效
        has_enough_observers: torch.Tensor,  # [N] 哪些环境有足够观测者
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量最小二乘估计（简化版：使用三角定位作为基础）
        
        为了性能，这里使用三角定位方法（前两个有效观测者）
        如果需要更精确的最小二乘，可以使用迭代优化，但会慢很多
        """
        # 简化：直接使用三角定位（对于批量处理，三角定位已经足够好）
        return self._triangulation_estimate_batch(
            observer_positions, bearings, valid_mask, has_enough_observers
        )


class MultiTargetBearingEstimator:
    
    def __init__(
        self,
        base_estimator: Optional[BearingOnlyEstimator] = None,
        association_method: str = "clustering",
    ):
        """
        初始化多目标估计器
        
        Args:
            base_estimator: 基础单目标估计器
            association_method: 数据关联方法 ("clustering" 或 "greedy")
        """
        self.base_estimator = base_estimator or BearingOnlyEstimator()
        self.association_method = association_method
    
    def estimate_multiple_targets(
        self,
        observer_positions: torch.Tensor,  # [N, M, 3] 观测者位置
        all_bearings: torch.Tensor,        # [N, M, E, 3] 每个观测者对每个目标的 bearing
        visibility_mask: torch.Tensor,     # [N, M, E] 哪些观测者能看到哪些目标
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        估计多个目标的位置
        
        Args:
            observer_positions: 观测者位置 [N, M, 3]
            all_bearings: 所有 bearing [N, M, E, 3]
            visibility_mask: 可见性掩码 [N, M, E]，True 表示可见
        
        Returns:
            estimated_positions: 估计的目标位置 [N, E, 3]
            uncertainties: 估计不确定性 [N, E]
            num_observers_per_target: 每个目标的观测者数量 [N, E]
        """
        N, M, E, _ = all_bearings.shape
        device = observer_positions.device
        dtype = observer_positions.dtype
        
        # 向量化处理：对每个目标-环境组合进行估计
        # 重塑为 [N*E, M, 3] 以便批量处理
        # 对于每个目标，我们需要所有M个友机的位置
        # observer_positions: [N, M, 3] -> 扩展为 [N, E, M, 3] -> reshape 为 [N*E, M, 3]
        # 使用 repeat 确保数据被正确复制
        observer_pos_expanded = observer_positions.unsqueeze(1).repeat(1, E, 1, 1)  # [N, E, M, 3]
        observer_pos_reshaped = observer_pos_expanded.reshape(N * E, M, 3)  # [N*E, M, 3]
        all_bearings_reshaped = all_bearings.reshape(N * E, M, 3)  # [N*E, M, 3]
        visibility_reshaped = visibility_mask.reshape(N * E, M)  # [N*E, M]
        
        # 批量估计所有目标
        est_pos_flat, uncertainty_flat, num_obs_flat = self.base_estimator.estimate_target_position(
            observer_pos_reshaped,  # [N*E, M, 3]
            all_bearings_reshaped,   # [N*E, M, 3]
            visibility_reshaped      # [N*E, M]
        )
        
        # 重塑回 [N, E, ...]
        estimated_positions = est_pos_flat.reshape(N, E, 3)  # [N, E, 3]
        uncertainties = uncertainty_flat.reshape(N, E)  # [N, E]
        num_observers = num_obs_flat.reshape(N, E)  # [N, E]
        
        return estimated_positions, uncertainties, num_observers


# ====================== 便捷函数 ======================

def estimate_target_from_bearings(
    observer_positions: torch.Tensor,
    bearings: torch.Tensor,
    observer_mask: Optional[torch.Tensor] = None,
    method: str = "triangulation",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    便捷函数：从 bearing 估计目标位置
    
    Args:
        observer_positions: 观测者位置 [N, M, 3] 或 [M, 3]
        bearings: 指向目标的单位向量 [N, M, 3] 或 [M, 3]
        observer_mask: 哪些观测者有效 [N, M] 或 [M]
        method: 估计方法 ("triangulation" 或 "least_squares")
    
    Returns:
        estimated_pos: 估计位置 [N, 3] 或 [3]
        uncertainty: 不确定性 [N] 或 scalar
        num_observers: 观测者数量 [N] 或 scalar
    """
    estimator = BearingOnlyEstimator(estimation_method=method)
    return estimator.estimate_target_position(observer_positions, bearings, observer_mask)


def estimate_enemy_positions_from_gimbal_bearings(
    friend_positions: torch.Tensor,      # [N, M, 3] 友机位置
    enemy_bearings: torch.Tensor,        # [N, M, E, 3] 每个友机到每个敌机的单位向量（导引头观测）
    visibility_mask: torch.Tensor,       # [N, M, E] 哪些友机能看到哪些敌机（导引头FOV内）
    method: str = "triangulation",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    estimator = MultiTargetBearingEstimator(
        base_estimator=BearingOnlyEstimator(estimation_method=method)
    )
    
    return estimator.estimate_multiple_targets(
        friend_positions, enemy_bearings, visibility_mask
    )


# ====================== 协作几何 + 简单 CV-EKF（世界系） ======================


class CooperativeCVEKFEstimator:
    """
    对每个敌机维护一个 3D 常速（CV）EKF，观测来自多机几何三角定位。

    - 先用 BearingOnlyEstimator 做几何位置估计（世界系）。
    - 再用 CV-EKF 做时间滤波和平滑。
    - 状态: [x, y, z, vx, vy, vz] （世界坐标系）
    """

    def __init__(
        self,
        num_envs: int,
        num_targets: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        base_estimator: Optional[BearingOnlyEstimator] = None,
        process_pos_std: float = 1.0,
        process_vel_std: float = 1.0,
        meas_noise_base: float = 5.0,
        init_cov: float = 1e3,
    ):
        self.num_envs = num_envs
        self.num_targets = num_targets
        self.device = device
        self.dtype = dtype

        self.base_estimator = base_estimator or BearingOnlyEstimator(estimation_method="triangulation")
        self.process_pos_std = float(process_pos_std)
        self.process_vel_std = float(process_vel_std)
        self.meas_noise_base = float(meas_noise_base)

        # 状态与协方差
        self.state = torch.zeros(num_envs, num_targets, 6, device=device, dtype=dtype)
        P0 = torch.eye(6, device=device, dtype=dtype) * init_cov
        self.P = P0.view(1, 1, 6, 6).repeat(num_envs, num_targets, 1, 1)

    def reset(self, env_ids: Optional[torch.Tensor] = None, init_pos: Optional[torch.Tensor] = None):
        """
        仅重置指定 env 的滤波器；init_pos: [len(env_ids), E, 3] 可选，填初值位置，速度清零。
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        self.state[env_ids] = 0.0
        if init_pos is not None:
            # init_pos shape: [len, E, 3]
            self.state[env_ids, :, 0:3] = init_pos
            self.state[env_ids, :, 3:6] = 0.0

        P0 = torch.eye(6, device=self.device, dtype=self.dtype) * 1e3
        self.P[env_ids] = P0

    def step(
        self,
        friend_positions: torch.Tensor,   # [N, M, 3]
        enemy_bearings: torch.Tensor,     # [N, M, E, 3]
        visibility_mask: torch.Tensor,    # [N, M, E]
        dt: float,
        enemy_centroid: Optional[torch.Tensor] = None,  # [N, 3] 可选：用于过滤虚假解
        centroid_filter_radius: float = 5.0,            # 离质心太远则丢弃
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：estimated_positions [N, E, 3], uncertainties [N, E], num_obs [N, E]
        """
        N, M, E, _ = enemy_bearings.shape
        dev, dtype = friend_positions.device, friend_positions.dtype

        # ---------- 观测：几何三角定位 ----------
        est_pos_geo, uncertainties_geo, num_obs_geo = estimate_enemy_positions_from_gimbal_bearings(
            friend_positions, enemy_bearings, visibility_mask, method="triangulation"
        )  # [N,E,3], [N,E], [N,E]

        # clamp overly far geo results to avoid polluting EKF
        geo_norm_all = torch.linalg.norm(est_pos_geo, dim=-1, keepdim=True)  # [N,E,1]
        far_mask = geo_norm_all > 300.0
        est_pos_geo = torch.where(far_mask, torch.zeros_like(est_pos_geo), est_pos_geo)
        uncertainties_geo = torch.where(far_mask.squeeze(-1), torch.ones_like(uncertainties_geo), uncertainties_geo)

        # ---------- 基于几何结果的有效性筛选 ----------
        geo_norm = torch.linalg.norm(est_pos_geo, dim=-1)  # [N,E]
        valid_geo = (
            (num_obs_geo >= 2)
            & (uncertainties_geo < 0.9)
            & torch.isfinite(geo_norm)
            & (geo_norm < 1e4)
        )

        # 新增：如果提供了敌机质心，过滤掉离质心“太远”的点（用户要求）
        if enemy_centroid is not None and valid_geo.any():
            dist_to_cen = torch.linalg.norm(est_pos_geo - enemy_centroid.unsqueeze(1), dim=-1)
            # 如果离团中心太远，则认为是虚假交点，直接丢弃
            too_far_from_centroid = dist_to_cen > centroid_filter_radius
            valid_geo = valid_geo & (~too_far_from_centroid)

        # ---------- EKF 预测 ----------
        # CV 模型: x_k+1 = F x_k + w,  F 为 6x6 常速状态转移矩阵
        # F = [[1,0,0,dt,0,0],
        #      [0,1,0,0,dt,0],
        #      [0,0,1,0,0,dt],
        #      [0,0,0,1,0,0 ],
        #      [0,0,0,0,1,0 ],
        #      [0,0,0,0,0,1 ]]
        F6 = torch.eye(6, device=dev, dtype=dtype)
        F6[0, 3] = dt
        F6[1, 4] = dt
        F6[2, 5] = dt
        # 形状对齐到 [N,E,6,6]
        while F6.dim() < self.P.dim():
            F6 = F6.unsqueeze(0).unsqueeze(0)

        # 过程噪声协方差 Q，位置和速度分别给定标准差
        q_pos = (self.process_pos_std * dt) ** 2
        q_vel = (self.process_vel_std * dt) ** 2
        Q6 = torch.zeros(6, 6, device=dev, dtype=dtype)
        Q6[0, 0] = q_pos
        Q6[1, 1] = q_pos
        Q6[2, 2] = q_pos
        Q6[3, 3] = q_vel
        Q6[4, 4] = q_vel
        Q6[5, 5] = q_vel
        while Q6.dim() < self.P.dim():
            Q6 = Q6.unsqueeze(0).unsqueeze(0)

        # 预测状态与协方差
        state_pred = torch.matmul(F6, self.state.unsqueeze(-1)).squeeze(-1)     # [N,E,6]
        P_pred = torch.matmul(F6, torch.matmul(self.P, F6.transpose(-1, -2))) + Q6  # [N,E,6,6]

        # ---------- EKF 更新 ----------
        H = torch.zeros(N, E, 3, 6, device=dev, dtype=dtype)
        H[..., 0, 0] = 1.0
        H[..., 1, 1] = 1.0
        H[..., 2, 2] = 1.0
        H_t = H.transpose(-1, -2)  # [N,E,6,3]

        # 测量噪声：根据几何不确定性和观测者数量调整
        # base^2 * (1 + uncertainty) / max(1, num_obs)
        denom = num_obs_geo.clamp_min(1.0)
        meas_std = self.meas_noise_base * (1.0 + uncertainties_geo) / denom
        meas_var = (meas_std.unsqueeze(-1))**2  # [N,E,1]
        R = torch.zeros(N, E, 3, 3, device=dev, dtype=dtype)
        R[..., 0, 0] = meas_var.squeeze(-1)
        R[..., 1, 1] = meas_var.squeeze(-1)
        R[..., 2, 2] = meas_var.squeeze(-1)

        # 有效观测条件（加入几何有效性约束）
        valid_meas = valid_geo

        # 计算卡尔曼增益
        S = torch.matmul(H, torch.matmul(P_pred, H_t)) + R            # [N,E,3,3]
        S_inv = torch.linalg.pinv(S)
        K = torch.matmul(P_pred, torch.matmul(H_t, S_inv))            # [N,E,6,3]

        # 无效观测时使用预测位置，避免将异常位置注入滤波器
        z = torch.where(valid_meas.unsqueeze(-1), est_pos_geo, state_pred[..., 0:3])  # [N,E,3]
        z_pred = torch.matmul(H, state_pred.unsqueeze(-1)).squeeze(-1)  # [N,E,3]
        innov = z - z_pred                                           # [N,E,3]

        state_upd = state_pred + torch.matmul(K, innov.unsqueeze(-1)).squeeze(-1)
        I6 = torch.eye(6, device=dev, dtype=dtype)
        while I6.dim() < self.P.dim():
            I6 = I6.unsqueeze(0).unsqueeze(0)
        P_upd = torch.matmul(I6 - torch.matmul(K, H), P_pred)

        mask = valid_meas.unsqueeze(-1)
        maskP = valid_meas.unsqueeze(-1).unsqueeze(-1)
        self.state = torch.where(mask, state_upd, state_pred)
        self.P = torch.where(maskP, P_upd, P_pred)

        # 输出
        est_pos = self.state[..., 0:3]            # [N,E,3]
        return est_pos, uncertainties_geo, num_obs_geo


    {
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

    #     # ====================== 2. 敌机观测 (协作几何 + CV-EKF) ======================
    #     # 先用几何三角定位估计敌机世界系位置，再用 CV-EKF 做时间平滑，最后按距离分配
    #     # 输出维度：K_target * 9 (估计相对位置3 + 不确定性1 + “观测次数”1 + 估计距离1 + 估计方向3)
    #     if E > 0:
    #         vis_fe = self._gimbal_enemy_visible_mask()  # [N, M, E] 可见性掩码

    #         # 计算 bearing（单位向量）：从友机指向敌机的方向
    #         # 注意：只对可见的敌机计算 bearing，不可见的设为零向量
    #         rel_all  = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)  # [N, M, E, 3]
    #         dist_all = torch.linalg.norm(rel_all, dim=-1, keepdim=True).clamp_min(eps)
    #         bearings = rel_all / dist_all  # [N, M, E, 3] 单位向量（bearing）
            
    #         # 关键：只保留可见敌机的 bearing，不可见的设为零向量
    #         bearings = torch.where(vis_fe.unsqueeze(-1), bearings, torch.zeros_like(bearings))

    #         # ---- 协作几何 + CV-EKF 滤波 ----
    #         dt_obs = float(self.physics_dt) * float(self.cfg.decimation)
    #         # 引入敌方质心进行过滤
    #         est_pos_world, uncertainties_local, num_obs_local = self.enemy_filter.step(
    #             self.fr_pos, bearings, vis_fe, dt_obs, 
    #             enemy_centroid=self._enemy_centroid,
    #             centroid_filter_radius=40.0
    #         )  # est_pos_world: [N,E,3]

    #         # 调试：可选打印
    #         # print("self.enemy_pos", self.enemy_pos)
    #         # print("est_pos_world", est_pos_world)

    #         # 缓存用于可视化
    #         self._dbg_bearings = bearings
    #         self._dbg_est_pos_world = est_pos_world
    #         self._dbg_vis_fe = vis_fe

    #         # 对每个友机，计算估计敌机位置相对于自己的位置（一对一隐式分配）
    #         est_rel_all = est_pos_world.unsqueeze(1) - self.fr_pos.unsqueeze(2)  # [N, M, E, 3]
    #         est_dist_all = torch.linalg.norm(est_rel_all, dim=-1)           # [N, M, E]

    #         # 对于当前不可见的敌机，或者不确定性过大的估计，设置大距离避免选中
    #         large_dist = torch.full_like(est_dist_all, 1e6)
    #         invalid_est = (~vis_fe) | (uncertainties_local.unsqueeze(1) > 10.0)
    #         est_dist_all = torch.where(invalid_est, large_dist, est_dist_all)

    #         # 一对一隐式分配：每个友机选择最近的估计敌机
    #         # 按距离排序，选择 Top-K 个最近的估计敌机
    #         sorted_indices = est_dist_all.argsort(dim=-1, descending=False)  # [N, M, E]
    #         valid_k = min(E, K_target)
    #         top_k_idx = sorted_indices[..., :valid_k]  # [N, M, valid_k]

    #         # Gather 估计位置、不确定性、观测者数量等信息
    #         gather_idx_3d = top_k_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    #         gather_idx_1d = top_k_idx

    #         est_rel_selected = torch.gather(est_rel_all, 2, gather_idx_3d)   # [N, M, valid_k, 3]
    #         est_dist_selected = torch.gather(est_dist_all, 2, gather_idx_1d)  # [N, M, valid_k]

    #         # uncertainties_local, num_obs_local: [N,E]
    #         uncertainties_expanded = uncertainties_local.unsqueeze(1).expand(-1, M, -1)  # [N,M,E]
    #         num_obs_expanded = num_obs_local.unsqueeze(1).expand(-1, M, -1)              # [N,M,E]
    #         uncertainties_selected = torch.gather(uncertainties_expanded, 2, gather_idx_1d)  # [N, M, valid_k]
    #         num_obs_selected = torch.gather(num_obs_expanded, 2, gather_idx_1d)              # [N, M, valid_k]

    #         # 计算估计位置的方向（单位向量）
    #         est_dir_selected = est_rel_selected / (est_dist_selected.unsqueeze(-1).clamp_min(eps))  # [N, M, valid_k, 3]

    #         # 标记有效估计（距离不是无穷大）
    #         is_valid = (est_dist_selected < 1e5).unsqueeze(-1).float()  # [N, M, valid_k, 1]

    #         # 拼接观测特征：[估计相对位置(3) + 不确定性(1) + 观测者数量(1) + 估计距离(1) + 估计方向(3)] = 9维
    #         target_features = torch.cat([
    #             est_rel_selected,              # [N, M, valid_k, 3] 估计相对位置
    #             uncertainties_selected.unsqueeze(-1),  # [N, M, valid_k, 1] 不确定性
    #             num_obs_selected.unsqueeze(-1),        # [N, M, valid_k, 1] 观测者数量
    #             est_dist_selected.unsqueeze(-1),       # [N, M, valid_k, 1] 估计距离
    #             est_dir_selected,                      # [N, M, valid_k, 3] 估计方向
    #         ], dim=-1)  # [N, M, valid_k, 9]

    #         # 应用有效性掩码
    #         target_features = target_features * is_valid

    #         # Padding 到固定长度 K_target
    #         target_obs_container = torch.zeros(N, M, K_target, 9, device=dev, dtype=dtype)
    #         if valid_k > 0:
    #             target_obs_container[:, :, :valid_k, :] = target_features
    #         target_feat_flat = target_obs_container.reshape(N, M, -1)  # [N, M, K_target * 9]
    #     else:
    #         target_feat_flat = torch.zeros((N, M, K_target * 9), device=dev, dtype=dtype)

    #     # ====================== 3. 自身状态 & ID ======================
    #     self_pos_abs = self.fr_pos
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
    #             e_hat_c,                 # 3
    #             dist_c,                  # 1
    #             target_feat_flat,        # K_target * 4
    #             # agent_id_feat,           # M
    #         ],
    #         dim=-1,
    #     )

    #     obs_dict = {ag: obs_each[:, i, :] for i, ag in enumerate(self.possible_agents)}

    #     return obs_dict
    }