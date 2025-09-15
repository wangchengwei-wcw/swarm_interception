#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize friendly/enemy markers for FAST-Quadcopter-Bodyrate with the new kinematic model.

Usage:
  python3 visualize_enemy.py --task FAST-Quadcopter-Bodyrate --num_envs 1 --verbosity DEBUG
  python3 visualize_enemy.py --task FAST-Quadcopter-Bodyrate --num_envs 1 --set_vm 2.2
  python3 visualize_enemy.py --enemy_speed 2.0 --enemy_spawn_radius 15
"""

import os, sys, argparse, logging, traceback

# ----------------- 1) 路径引导：确保能 import 你的 envs 包 -----------------
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANDIDATE_DIRS = [
    PROJ_ROOT,                              # 项目根
    os.path.join(PROJ_ROOT, "envs"),        # 以防需要
    os.path.join(PROJ_ROOT, "envs", "lib"), # 你的 isaaclab 源码位置（若有）
    os.path.join(PROJ_ROOT, "envs", "lib", "IsaacLab"),
    os.path.join(PROJ_ROOT, "envs", "lib", "IsaacLab", "source"),
]
for d in CANDIDATE_DIRS:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

# ----------------- 2) AppLauncher（必须先起 SimulationApp 再 import isaaclab 相关） -----------------
try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError:
    from omni.isaac.lab.app import AppLauncher  # 兼容命名
# CLI
parser = argparse.ArgumentParser(description="Enemy/Friendly visualization for FAST-Quadcopter-Bodyrate.")
parser.add_argument("--task", type=str, default="FAST-Quadcopter-Bodyrate")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--verbosity", type=str, default="INFO",
                    choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"])
# 敌机临时覆盖
parser.add_argument("--enemy_speed", type=float, default=None)
parser.add_argument("--enemy_spawn_radius", type=float, default=None)
parser.add_argument("--enemy_height_min", type=float, default=None)
parser.add_argument("--enemy_height_max", type=float, default=None)
# 友方 Vm 接口（若给则直接设为常数）
parser.add_argument("--set_vm", type=float, default=None)
# 追加 AppLauncher 通用参数
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Logging
LEVEL = dict(TRACE=logging.DEBUG, DEBUG=logging.DEBUG, INFO=logging.INFO,
             SUCCESS=logging.INFO, WARNING=logging.WARNING, ERROR=logging.ERROR,
             CRITICAL=logging.CRITICAL)[args.verbosity]
logging.basicConfig(level=LEVEL, format="[%(levelname)s] %(message)s")
log = logging.getLogger("enemy_vis")

# 启动 SimulationApp
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ----------------- 3) SimulationApp 起后，再 import 依赖它的模块 -----------------
import gymnasium as gym
import torch

# 触发环境注册（很重要）
try:
    import envs.Loitering_Munition_interception_single as env_mod
except Exception as e:
    log.error("导入 envs.Loitering_Munition_interception_single 失败：%s", e)
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

# ----------------- 4) 构建 cfg，并覆盖参数 -----------------
def build_cfg(num_envs: int):
    # 直接实例化你文件里的 cfg 类
    cfg = env_mod.QuadcopterBodyrateEnvCfg()
    # 环境数量
    cfg.scene.num_envs = int(num_envs)
    # 确保渲染频率关联（默认已在类里计算，这里不强制改）
    # 临时覆盖敌机参数
    if args.enemy_speed is not None:
        cfg.enemy_speed = float(args.enemy_speed)
    if args.enemy_spawn_radius is not None:
        cfg.enemy_spawn_radius = float(args.enemy_spawn_radius)
    if args.enemy_height_min is not None:
        cfg.enemy_height_min = float(args.enemy_height_min)
    if args.enemy_height_max is not None:
        cfg.enemy_height_max = float(args.enemy_height_max)
    # 打开调试可视化
    cfg.debug_vis = True
    cfg.debug_vis_goal = False
    cfg.debug_vis_enemy = True
    return cfg

# ----------------- 5) 创建 env -----------------
def main():
    env_cfg = build_cfg(args.num_envs)
    try:
        env = gym.make(args.task, cfg=env_cfg)
    except Exception as e:
        log.error("gym.make(%s) 失败：%s", args.task, e)
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)

    # reset 一下以生成与可视化
    obs, _ = env.reset()
    log.info("可视化启动: num_envs=%d, enemy_speed=%.3f, spawn_R=%.2f, h=[%.1f, %.1f], Vm=[%.2f, %.2f]",
             env.unwrapped.num_envs,
             getattr(env_cfg, "enemy_speed", 1.5),
             getattr(env_cfg, "enemy_spawn_radius", 12.0),
             getattr(env_cfg, "enemy_height_min", 1.0),
             getattr(env_cfg, "enemy_height_max", 3.0),
             getattr(env_cfg, "Vm_min", 1.0),
             getattr(env_cfg, "Vm_max", 3.0))

    # 如指定了 set_vm，就把所有 env 设为这个 Vm
    if args.set_vm is not None:
        try:
            env.unwrapped.set_friendly_vm(float(args.set_vm))
            log.info("已将友方 Vm 设置为常数：%.3f", float(args.set_vm))
        except Exception as e:
            log.error("设置 Vm 失败：%s", e)

    # 构造零动作（shape 对齐环境的 action_space 维度）
    # 构造零动作
    try:
        act_dim = int(env.unwrapped.cfg.action_space)
    except Exception:
        act_dim = 3
    zero_actions = torch.zeros((env.unwrapped.num_envs, act_dim), device=env.unwrapped.device)

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                obs, rew, terminated, truncated, info = env.step(zero_actions)
                if (terminated | truncated).any():
                    try:
                        env.unwrapped.reset_done()
                    except AttributeError:
                        env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        simulation_app.close()

    zero_actions = torch.zeros((env.unwrapped.num_envs, act_dim), device=env.unwrapped.device)

    # 主循环
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                obs, rew, terminated, truncated, info = env.step(zero_actions)
                # 到步末尾自动 reset（便于长时间观测）
                if (terminated | truncated).any():
                    env.reset_done()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error("主循环异常：%s", e)
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
