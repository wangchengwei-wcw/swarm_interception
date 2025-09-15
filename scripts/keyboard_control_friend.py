import argparse
import os
import sys
from isaaclab.app import AppLauncher
import gymnasium as gym
from isaaclab.devices import Se3Keyboard
from isaaclab_tasks.utils import parse_env_cfg
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for quadcopter environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument(
    "--task",
    type=str,
    default="FAST-Quadcopter-Bodyrate",  # Example task, use your specific task
    help="Name of the task. Optional includes: FAST-Quadcopter-Waypoint; FAST-Quadcopter-Vel; FAST-Quadcopter-Acc;",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--velocity", type=float, default=4.0, help="Velocity of teleoperation.")
parser.add_argument(
    "--verbosity", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="Verbosity level of the custom logger."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Create environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Create environment
env_cfg = parse_env_cfg(args_cli.task, device="cuda", num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
env_cfg.debug_vis_goal = False
env = gym.make(args_cli.task, cfg=env_cfg)

# Create controller for keyboard input
teleop_interface = Se3Keyboard()

# Reset environment
env.reset()
teleop_interface.reset()

# Friend position initialization (static until keyboard control)
fr_pos = torch.zeros(1, 3)  # Initial position at the origin (x, y, z)
fr_vel = torch.zeros(1, 3)  # Initial velocity (x, y, z)
dt = env.unwrapped.step_dt

def visualize_images_live(images):
    N = images.shape[0]
    channels = images.shape[-1]
    if channels == 1:
        images = np.repeat(images, 3, axis=-1)
        images = np.where(np.isinf(images), np.nan, images)
    elif channels == 4:
        images = images[..., :3]

    height, width = images.shape[1], images.shape[2]
    cols = int(math.ceil(math.sqrt(N)))
    rows = int(math.ceil(N / cols))

    canvas = np.zeros((rows * height, cols * width, 3))

    for idx in range(N):
        row = idx // cols
        col = idx % cols
        canvas[row * height : (row * height) + height, col * width : (col * width) + width, :] = images[idx]

    if not hasattr(visualize_images_live, "img_plot"):
        visualize_images_live.fig = plt.figure()
        visualize_images_live.fig.canvas.manager.set_window_title("Images")
        visualize_images_live.img_plot = plt.imshow(canvas)
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        visualize_images_live.img_plot.set_data(canvas)

    plt.draw()
    plt.pause(0.001)

def main():
    global fr_pos, fr_vel
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            # Get keyboard command
            delta_pose, _ = teleop_interface.advance()

            # Update friendly position based on key inputs
            if delta_pose[0] > 0:
                fr_pos[0, 0] += args_cli.velocity * dt  # Move forward (x axis)
            elif delta_pose[0] < 0:
                fr_pos[0, 0] -= args_cli.velocity * dt  # Move backward (x axis)

            if delta_pose[1] > 0:
                fr_pos[0, 1] += args_cli.velocity * dt  # Move right (y axis)
            elif delta_pose[1] < 0:
                fr_pos[0, 1] -= args_cli.velocity * dt  # Move left (y axis)

            if delta_pose[4] > 0:
                fr_pos[0, 2] += args_cli.velocity * dt  # Move up (z axis)
            elif delta_pose[4] < 0:
                fr_pos[0, 2] -= args_cli.velocity * dt  # Move down (z axis)

            # Optionally visualize the updated friendly position
            print(f"Friendly position: {fr_pos}")

            # Apply action (update position)
            env.unwrapped.fr_pos = fr_pos

            # Step the environment with the updated action
            obs, _, reset_terminated, reset_time_outs, _ = env.step(None)  # No specific actions, just control update

            if args_cli.task in ["FAST-RGB-Waypoint", "FAST-Depth-Waypoint"]:
                visualize_images_live(obs["image"].cpu().numpy())

    # Close the simulation
    env.close()

if __name__ == "__main__":
    main()

    simulation_app.close()
