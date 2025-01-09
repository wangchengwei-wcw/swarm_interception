import torch


@torch.jit.script
def quat_to_ang_between_z_body_and_z_world(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    z_body_x = 2 * (x * z + y * w)
    z_body_y = 2 * (y * z - x * w)
    z_body_z = 1 - 2 * (x**2 + y**2)
    z_mag = torch.sqrt(z_body_x**2 + z_body_y**2 + z_body_z**2)

    ang = torch.acos(z_body_z / z_mag)

    return ang
