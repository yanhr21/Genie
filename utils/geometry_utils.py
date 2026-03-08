import torch
import torch.nn.functional as F
from einops import rearrange


def resize_traj_and_ray(traj_n_ray, mem_size, future_size, height, width):
    '''
    traj_n_ray: bv c t h w
    '''
    orig_t = traj_n_ray.shape[2]
    target_t = int(mem_size) + int(future_size)
    if orig_t < target_t:
        raise ValueError(
            f"traj_n_ray has insufficient timesteps: orig_t={orig_t}, required={target_t}"
        )

    mem = traj_n_ray[:, :, :mem_size]
    mem = rearrange(mem, 'bv c t h w -> (bv t) c h w')
    mem = F.interpolate(mem, (height, width), mode='bilinear')
    mem = rearrange(mem, '(bv t) c h w -> bv c t h w', t=mem_size)

    future = traj_n_ray[:, :, mem_size:]  # bv c t h w
    future = F.interpolate(future, (future_size, height, width), mode='trilinear')

    out = torch.cat([mem, future], dim=2)
    return out
