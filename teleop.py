"""Script to run a keyboard teleoperation with quacopter environments"""

import argparse

from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Keyboard teleoperation for quadcopter environments."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=8, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task in [
    "FAST-Quadcopter-RGB-Camera-Direct-v0",
    "FAST-Quadcopter-Depth-Camera-Direct-v0",
]:
    args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings

import env, camera_env, swarm_env
from isaaclab.devices import Se3Keyboard
from isaaclab_tasks.utils import parse_env_cfg


def delta_pose_to_action(delta_pose: np.ndarray) -> np.ndarray:
    action = np.array([-1.0, 0.0, 0.0, 0.0])

    if delta_pose[4] > 0:
        action[0] = 0.1 * args_cli.sensitivity

    if delta_pose[0] > 0:
        action[2] = 0.001 * args_cli.sensitivity
    elif delta_pose[0] < 0:
        action[2] = -0.001 * args_cli.sensitivity

    if delta_pose[1] > 0:
        action[1] = -0.001 * args_cli.sensitivity
    elif delta_pose[1] < 0:
        action[1] = 0.001 * args_cli.sensitivity

    if delta_pose[2] > 0:
        action[3] = 0.01 * args_cli.sensitivity
    elif delta_pose[2] < 0:
        action[3] = -0.01 * args_cli.sensitivity

    return action


def visualize_images_live(tensor):
    # Tensor shape can be (i, height, width) or (i, height, width, channels)
    i = tensor.shape[0]

    if len(tensor.shape) == 3:
        # Case when the tensor has no channels dimension (grayscale images)
        tensor = np.expand_dims(
            tensor, -1
        )  # Add a channels dimension, becomes (i, height, width, 1)

    # Determine if the images are grayscale or color
    channels = tensor.shape[-1]

    if channels == 1:
        # Convert grayscale images to 3 channels by repeating the single channel
        tensor = np.repeat(tensor, 3, axis=-1)
        tensor = np.where(np.isinf(tensor), np.nan, tensor)
    elif channels == 4:
        # Use only the first 3 channels as RGB, ignore the 4th channel (perhaps alpha)
        tensor = tensor[..., :3]
    else:
        warnings.warn(f"Unexpected channel number {channels}.", UserWarning)

    # Get the height and width from the first image (all images have the same size)
    height, width = tensor.shape[1], tensor.shape[2]

    # Calculate the grid size
    cols = int(math.ceil(math.sqrt(i)))
    rows = int(math.ceil(i / cols))

    # Create an empty canvas to hold the images
    canvas = np.zeros((rows * height, cols * width, 3))

    for idx in range(i):
        row = idx // cols
        col = idx % cols
        # Place the image in the grid cell
        canvas[
            row * height : (row * height) + height,
            col * width : (col * width) + width,
            :,
        ] = tensor[idx]

    # Display the canvas
    if not hasattr(visualize_images_live, "img_plot"):
        # Create the plot for the first time
        visualize_images_live.fig = plt.figure()
        visualize_images_live.fig.canvas.manager.set_window_title("Images")
        visualize_images_live.img_plot = plt.imshow(canvas)
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        # Update the existing plot
        visualize_images_live.img_plot.set_data(canvas)

    plt.draw()
    plt.pause(0.001)  # Pause to allow the figure to update


def main():
    # Create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Create controller
    teleop_interface = Se3Keyboard()
    # Add teleoperation key for env reset
    teleop_interface.add_callback("R", env.reset)

    # Reset environment
    env.reset()
    teleop_interface.reset()

    # Simulate environment
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            # Get keyboard command
            delta_pose, _ = teleop_interface.advance()
            action = delta_pose_to_action(delta_pose)
            action = action.astype("float32")
            # Convert to torch
            actions = torch.tensor(action, device=env.unwrapped.device).repeat(
                env.unwrapped.num_envs, 1
            )
            if args_cli.task == "FAST-Quadcopter-Swarm-Direct-v0":
                actions = {drone: actions for drone in env_cfg.possible_agents}

            # Apply actions
            obs, _, _, _, _ = env.step(actions)

            if args_cli.task in [
                "FAST-Quadcopter-RGB-Camera-Direct-v0",
                "FAST-Quadcopter-Depth-Camera-Direct-v0",
            ]:
                visualize_images_live(obs["policy"].cpu().numpy())

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
