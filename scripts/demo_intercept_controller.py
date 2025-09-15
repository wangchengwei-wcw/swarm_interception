#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo: simple intercept controller (稳态版)
用法示例：
  python demo_intercept_controller.py --headless --steps 1500 --num_envs 1 --vm_set 2.5 --k_yaw 0.8 --k_pitch 0.8 --act_beta 0.35
"""

import os, sys, math, argparse, traceback

# --------- 1) 修正 PYTHONPATH，让能导入 envs/* ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --------- 2) 先启动 SimulationApp，再 import 依赖 ----------
try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError:
    from omni.isaac.lab.app import AppLauncher  # 兼容旧命名

parser = argparse.ArgumentParser(description="Demo: simple intercept controller (stable)")
parser.add_argument("--task", type=str, default="FAST-Quadcopter-Bodyrate")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--vm_set", type=float, default=None, help="固定友方速度(在 [Vm_min, Vm_max] 内)")
parser.add_argument("--k_yaw", type=float, default=0.8, help="平面指向控制增益 -> nz（建议先小于1）")
parser.add_argument("--k_pitch", type=float, default=0.8, help="爬升角指向控制增益 -> ny（建议先小于1）")
parser.add_argument("--act_beta", type=float, default=0.35, help="动作一阶滤波系数: act = beta*raw + (1-beta)*prev")
# 由 AppLauncher 自动添加 --headless / --device 等
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("[DEMO] Starting SimulationApp ...", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[DEMO] SimulationApp started.", flush=True)

# --------- 3) 启动后才能 import gym/任务解析/你的环境 ----------
import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg

print("[DEMO] Importing env module to trigger gym.register ...", flush=True)
import envs.Loitering_Munition_interception_single   # 只需导入一次，内部会 gym.register
print("[DEMO] Env module imported.", flush=True)

# --------- 4) 解析 env cfg 并创建环境（兼容无 disable_fabric 的情况）----------
use_fabric = True if not hasattr(args, "disable_fabric") else (not args.disable_fabric)
print(f"[DEMO] parse_env_cfg(use_fabric={use_fabric}, device={args.device}, num_envs={args.num_envs})", flush=True)
env_cfg = parse_env_cfg(
    args.task,
    device=args.device,
    num_envs=args.num_envs,
    use_fabric=use_fabric,
)
env = gym.make(args.task, cfg=env_cfg)
obs, info = env.reset()
uw = env.unwrapped  # 直接访问内部状态以做 debug

# 固定速度（可选）：调用你导出的接口
if args.vm_set is not None:
    try:
        uw.set_friendly_vm(float(args.vm_set))
        print(f"[DEMO] set_friendly_vm({args.vm_set}) 已调用", flush=True)
    except Exception as e:
        print(f"[WARN] set_friendly_vm 调用失败：{e}", flush=True)

# --------- 5) 工具函数 ----------
def wrap_to_pi(a: torch.Tensor):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

# --------- 6) 稳定版“朝向”控制器：tanh 限幅 + 1g 配平 ----------
def policy(obs_tensor: torch.Tensor) -> torch.Tensor:
    # 用 env 内部的真状态算控制（demo 专用）
    v = uw.fr_vel_w                      # [N,3]
    rel = uw.enemy_pos - uw.fr_pos       # [N,3]

    # 方位角误差
    psi_now  = torch.atan2(v[:, 1], v[:, 0])
    psi_goal = torch.atan2(rel[:, 1], rel[:, 0])
    dpsi = wrap_to_pi(psi_goal - psi_now)

    # 爬升角误差
    vxy = torch.linalg.norm(v[:, :2], dim=1).clamp(1e-6)
    gamma_now  = torch.atan2(v[:, 2], vxy)
    rxy = torch.linalg.norm(rel[:, :2], dim=1).clamp(1e-6)
    gamma_goal = torch.atan2(rel[:, 2], rxy)
    dgamma = torch.clamp(gamma_goal - gamma_now, -math.pi/2, math.pi/2)

    # 使用 tanh 限幅，避免高频抖动；ny 加 1g 配平
    nz_cmd_g = args.k_yaw   * torch.tanh(dpsi)
    ny_cmd_g = 1.0 + args.k_pitch * torch.tanh(dgamma)

    # 速度命令：建议固定一个不小的速度，降低低速区姿态切换
    if args.vm_set is not None:
        vm = float(args.vm_set)
    else:
        vm = max(0.5 * (uw.cfg.Vm_min + uw.cfg.Vm_max), 0.8 * uw.cfg.Vm_max)

    # 归一化到 [-1,1]
    ny_norm = torch.clamp(ny_cmd_g / uw.cfg.ny_max_g, -1.0, 1.0)
    nz_norm = torch.clamp(nz_cmd_g / uw.cfg.nz_max_g, -1.0, 1.0)
    vm_norm = (torch.tensor(vm, device=uw.device) - uw.cfg.Vm_min) / (uw.cfg.Vm_max - uw.cfg.Vm_min) * 2.0 - 1.0
    vm_norm = torch.clamp(vm_norm, -1.0, 1.0)

    return torch.stack([ny_norm, nz_norm, vm_norm.expand_as(ny_norm)], dim=-1)  # [N,3]

# --------- 7) 主循环（带动作一阶滤波 + 回合同步） ----------
prev_act = None  # 一阶滤波状态
beta = float(args.act_beta)
beta = min(max(beta, 0.0), 1.0)

try:
    print(f"[DEMO] Running {args.steps} steps "
          f"(headless={args.headless}, num_envs={args.num_envs})", flush=True)
    for t in range(args.steps):
        with torch.inference_mode():
            raw = policy(obs["policy"])  # [N,3]

            # 一阶低通：act = beta*raw + (1-beta)*prev
            if prev_act is None:
                act = raw
            else:
                act = beta * raw + (1.0 - beta) * prev_act

            obs, rew, terminated, truncated, info = env.step(act)

            # 回合结束的 env，把 prev_act 对应通道重置为 raw，避免跨回合继承
            if isinstance(terminated, torch.Tensor):
                reset_mask = (terminated | truncated)
                if reset_mask.any():
                    if prev_act is None:
                        prev_act = act.clone()
                    prev_act[reset_mask] = raw[reset_mask]

            prev_act = act  # 更新滤波状态

        # 调试打印
        if t % 20 == 0:
            dist0 = torch.linalg.norm(uw.enemy_pos[0] - uw.fr_pos[0]).item()
            vmag0 = torch.linalg.norm(uw.fr_vel_w[0]).item()
            succ0 = dist0 < uw.cfg.success_distance_threshold
            print(
                f"[t={t:04d}] dist0={dist0:.3f}  success={succ0}  "
                f"|v|={vmag0:.2f}  Vm={uw.Vm[0].item():.2f}  "
                f"ny_nz=({act[0,0].item():+.2f},{act[0,1].item():+.2f})",
                flush=True,
            )

        # （可选）若需要看到每次 reset 的原因：
        if bool(terminated.any()) or bool(truncated.any()):
            term = getattr(uw, "extras", {}).get("termination", {})
            print(f"[t={t:04d}] terminated={bool(terminated.any())} truncated={bool(truncated.any())}  reasons={term}",
                  flush=True)

except KeyboardInterrupt:
    print("\n[DEMO] Interrupted by user.", flush=True)
except Exception as e:
    print("[DEMO][ERROR] Exception in main loop:", e, flush=True)
    traceback.print_exc()
finally:
    try:
        env.close()
    except Exception:
        pass
    simulation_app.close()
    print("[DEMO] Closed.", flush=True)
