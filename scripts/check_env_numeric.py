#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, math, traceback, functools
import argparse
import torch
import gymnasium as gym

# ---- 强制行缓冲输出 ----
print = functools.partial(print, flush=True)

# ---- 路径注入：确保能 import envs.* 与 isaaclab ----
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAB_ROOT = os.path.join(PROJ_ROOT, "envs", "lib", "IsaacLab")
for d in [PROJ_ROOT, os.path.join(LAB_ROOT, "source"), os.path.join(LAB_ROOT, "exts")]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Headless numeric smoke test for FAST-Quadcopter-Bodyrate")
parser.add_argument("--task", type=str, default="FAST-Quadcopter-Bodyrate")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--vm_set", type=float, default=None, help="固定友方 Vm（将 clamp 到 [Vm_min, Vm_max]）")
parser.add_argument("--print_every", type=int, default=50)
# 交给 AppLauncher 增加 headless/renderer 等
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("[TEST] Starting SimulationApp ...")
app = AppLauncher(args).app
print("[TEST] SimulationApp started.")

# 依赖 SimulationApp 的包放在后面 import
from isaaclab_tasks.utils import parse_env_cfg

# 触发 gym.register（很关键）
print("[TEST] Importing env module to trigger gym.register ...")
import envs.Loitering_Munition_interception_single  # noqa: F401
print("[TEST] Env module imported.")

def make_env():
    print("[TEST] Parsing env cfg ...")
    # 新版 AppLauncher 没有 disable_fabric，默认启用 fabric
    use_fabric = True
    if hasattr(args, "disable_fabric"):
        use_fabric = not args.disable_fabric
    elif hasattr(args, "enable_fabric"):
        use_fabric = args.enable_fabric
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=use_fabric,
    )
    print("[TEST] Creating gym env ...")
    env = gym.make(args.task, cfg=env_cfg)
    print("[TEST] Env created.")
    return env


def main():
    try:
        env = make_env()
        print("[TEST] Resetting env ...")
        obs, info = env.reset()
        print("[TEST] Env reset OK.")

        env_u = env.unwrapped
        N = env_u.num_envs
        obs_dim = int(env_u.cfg.observation_space)
        act_dim = int(getattr(env_u.cfg, "action_space", 3))

        if args.vm_set is not None and hasattr(env_u, "set_friendly_vm"):
            vm_min = float(env_u.cfg.Vm_min); vm_max = float(env_u.cfg.Vm_max)
            vm_cmd = float(max(vm_min, min(vm_max, args.vm_set)))
            env_u.set_friendly_vm(vm_cmd)
            print(f"[TEST] Vm fixed to {vm_cmd:.3f} (clamped in [{vm_min:.2f},{vm_max:.2f}])")

        actions = torch.zeros((N, act_dim), device=env_u.device)

        success_cnt = 0
        dist_min, dist_max = float("inf"), 0.0

        print(f"[TEST] obs first keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
        if isinstance(obs, dict) and "policy" in obs:
            print(f"[TEST] obs['policy'].shape at reset = {tuple(obs['policy'].shape)} (expect: [{N}, {obs_dim}])")

        for step in range(1, args.steps + 1):
            with torch.inference_mode():
                obs, rew, terminated, truncated, info = env.step(actions)

            # 形状与数值检查
            if not (isinstance(obs, dict) and "policy" in obs):
                raise RuntimeError("obs 不是 dict 或缺少 'policy' 键")
            o = obs["policy"]
            if step == 1 and o.shape[1] != obs_dim:
                raise RuntimeError(f"观测维度不一致: got {o.shape[1]} vs cfg {obs_dim}")
            if not torch.isfinite(o).all():
                raise RuntimeError("观测中出现 NaN/Inf")

            # 读内部状态做数值监控
            fr = env_u.fr_pos
            en = env_u.enemy_pos
            dist = torch.linalg.norm(en - fr, dim=1)
            dmean = float(dist.mean().item())
            dmin = float(dist.min().item()); dmax = float(dist.max().item())
            dist_min = min(dist_min, dmin); dist_max = max(dist_max, dmax)

            succ = (dist < env_u.cfg.success_distance_threshold)
            success_cnt += int(succ.sum().item())

            if step % args.print_every == 0 or step in (1, args.steps):
                closing0 = float((-(en[0] - fr[0]) @ (env_u.enemy_vel[0] - env_u.fr_vel_w[0])
                                  / dist[0].clamp_min(1e-6)).item())
                print(f"[STEP {step:4d}] "
                      f"dist_mean={dmean:7.3f}  dist[min,max]=[{dmin:6.3f},{dmax:6.3f}]  "
                      f"closing(env0)={closing0:7.3f}  success_in_batch={int(succ.sum().item())}")

        print("\n========== SUMMARY ==========")
        print(f"Total steps: {args.steps}")
        print(f"Obs dims: {tuple(obs['policy'].shape)}  (N={N})")
        print(f"Distance min/max over run: [{dist_min:.3f}, {dist_max:.3f}]")
        print(f"Success radius: {env_u.cfg.success_distance_threshold:.3f} m")
        print(f"Success count (all envs * steps): {success_cnt}")
        print("All finite checks: OK")
        print("=============================\n")

    except Exception as e:
        print("[TEST] ERROR:", e)
        traceback.print_exc()
    finally:
        try:
            env.close()
        except Exception:
            pass
        app.close()
        print("[TEST] Closed cleanly.")

if __name__ == "__main__":
    main()
