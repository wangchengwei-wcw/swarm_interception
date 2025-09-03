# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FASTInterceptSwarmPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100 #每个环境中每次采样的步数
    max_iterations = 100000 #最大迭代次数
    save_interval = 10 #每隔多少迭代存一次 checkpoint
    experiment_name = ""
    clip_actions = 1.0 #将策略输出裁剪到[-1, 1]之间
    empirical_normalization = True #使用经验统计（当前 batch）对优势/回报等做归一化，打开后能提升训练稳定性
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, #高斯策略的初始标准差
        actor_hidden_dims=[256, 256, 128, 64], #Actor 的多层感知机宽度
        critic_hidden_dims=[256, 256, 128, 64], #Critic 的多层感知机宽度
        activation="elu", #激活函数
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, #值函数损失在总损失中的系数
        use_clipped_value_loss=True,
        clip_param=0.2, #策略比率裁剪阈值，小 → 更新更保守更稳；大 → 学得快但易崩。
        entropy_coef=0.0005, #熵正则系数
        num_learning_epochs=5, #每次迭代对同一批数据重复训练的轮数，大 → 学得快但过拟合风险高；小 → 学得慢但稳
        num_mini_batches=25, #将本迭代 batch 切成多少个 mini-batch
        learning_rate=1.0e-5, #学习率
        schedule="fixed", #学习率调度
        gamma=0.99, #折扣因子
        lam=0.95, #GAE(λ) 的 λ
        desired_kl=0.016, #目标 KL（步长控制器）
        max_grad_norm=1.0, #梯度裁剪上限
    )

