import torch
from typing import Optional
from fireants.interpolator import fireants_interpolator
from torch import distributed as dist

def get_affine_rescaling_ringsampler(min_img_coords: torch.Tensor, max_img_coords: torch.Tensor, image_size: Optional[torch.Tensor] = None, align_corners: bool = True) -> torch.Tensor:
    '''
    `min_img_coords` and `max_img_coords` are the minimum and maximum coordinates of the "points" in the image (no matter what `align_corners` is)
    they will be mapped to whatever the sharded image size is 

    image coords: (X, Y, Z)
    '''
    mx, my, mz = min_img_coords
    Mx, My, Mz = max_img_coords
    # coords are stored in ZYX order
    if image_size is None:
        assert align_corners, "align_corners must be True if image_size is not provided"
    else:
        sZ, sY, sX = image_size[-3:]
    current_device = torch.cuda.current_device()
    if align_corners:
        # map (-1, -1, -1) to (mx, my, mz) and (1, 1, 1) to (Mx, My, Mz)
        _mx, _my, _mz = -1.0, -1.0, -1.0
        _Mx, _My, _Mz = 1.0, 1.0, 1.0
    else:
        _mx, _my, _mz = -1.0 + 1.0/sX, -1.0 + 1.0/sY, -1.0 + 1.0/sZ
        _Mx, _My, _Mz = 1.0 - 1.0/sX, 1.0 - 1.0/sY, 1.0 - 1.0/sZ
    # map affine
    ax = (_Mx - _mx)/(Mx - mx)
    ay = (_My - _my)/(My - my)
    az = (_Mz - _mz)/(Mz - mz)
    tx = (_mx*Mx - _Mx*mx)/(Mx - mx)
    ty = (_my*My - _My*my)/(My - my)
    tz = (_mz*Mz - _Mz*mz)/(Mz - mz)
    affine = torch.eye(4, 4, device=current_device)
    affine[0, 0] = ax
    affine[1, 1] = ay
    affine[2, 2] = az
    affine[0, 3] = tx
    affine[1, 3] = ty
    affine[2, 3] = tz
    return affine[None]  # (1, 4, 4)

def make_homogenous(affine: torch.Tensor) -> torch.Tensor:
    '''
    Make the affine matrix homogenous.
    If the affine matrix is not homogenous, add a row of zeros and a 1 in the last column.
    If the affine matrix is homogenous, return it as is.
    '''
    B, D1, D2 = affine.shape
    if D1 + 1 == D2:
        row = torch.zeros((B, 1, D2), device=affine.device, dtype=affine.dtype)
        row[:, :, -1] = 1
        affine_h = torch.cat([affine, row], dim=1)
        return affine_h
    elif D1 == D2:
        return affine
    else:
        raise ValueError('Invalid affine shape: {}'.format(affine.shape))


def distributed_grid_sampler_3d(
    image: torch.Tensor,
    min_img_coords: torch.Tensor,
    max_img_coords: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    grid: Optional[torch.Tensor] = None,   # displacement
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    min_coords: Optional[tuple] = None,
    max_coords: Optional[tuple] = None,
    out_shape: tuple = None,
    is_displacement: bool = True
) -> torch.Tensor:
    '''

    '''
    # get rank and world size
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert is_displacement, "is_displacement must be True"

    # Get base image
    image_size = image.shape[-3:]
    scale_factor = get_affine_rescaling_ringsampler(min_img_coords, max_img_coords, image_size, align_corners)
    # scale_affine
    scaled_affine = scale_factor @ make_homogenous(affine) if affine is not None else scale_factor
    scaled_affine = scaled_affine[:, :3, :]  # (B, 3, 4)
    scaled_disp = torch.einsum('bij,b...j->b...i', scaled_affine[:, :3, :3], grid)
    # sample image
    ret_image = fireants_interpolator(image, affine=scaled_affine, grid=scaled_disp, mode=mode, padding_mode='zeros', align_corners=align_corners, is_displacement=is_displacement)

    # fetch from the previous rank
    prev_rank = (rank - 1 + world_size) % world_size
    next_rank = (rank + 1) % world_size

    # prepare variables to send 
    send_sharded_image = image
    send_sharded_size = torch.tensor(list(image.shape), device=image.device)
    send_sharded_min_coords = min_img_coords
    send_sharded_max_coords = max_img_coords

    # prepare variables to receive
    recv_sharded_min_coords = torch.zeros_like(min_img_coords)
    recv_sharded_max_coords = torch.zeros_like(max_img_coords)
    recv_sharded_size = torch.zeros_like(send_sharded_size)

    for ring_id in range(world_size - 1):
        # send asynchronously to next rank
        reqs = []
        reqs.append(dist.isend(send_sharded_min_coords, dst=next_rank))
        reqs.append(dist.isend(send_sharded_max_coords, dst=next_rank))
        reqs.append(dist.isend(send_sharded_size, dst=next_rank))
        reqs.append(dist.isend(send_sharded_image, dst=next_rank))
        
        # receive from previous rank
        reqs = []
        dist.recv(recv_sharded_min_coords, src=prev_rank)
        dist.recv(recv_sharded_max_coords, src=prev_rank)
        dist.recv(recv_sharded_size, src=prev_rank)
        # convert to list and create tensor
        recv_sharded_image = torch.zeros(recv_sharded_size.to(torch.long).tolist(), device=image.device, dtype=image.dtype)
        dist.recv(recv_sharded_image, src=prev_rank)

        ### sample the image
        scale_factor = get_affine_rescaling_ringsampler(recv_sharded_min_coords, recv_sharded_max_coords, recv_sharded_size, align_corners)
        # scale_affine
        scaled_affine = scale_factor @ make_homogenous(affine) if affine is not None else scale_factor
        scaled_affine = scaled_affine[:, :3, :]  # (B, 3, 4)
        scaled_disp = torch.einsum('bij,b...j->b...i', scaled_affine[:, :3, :3], grid)
        # sample image
        ret_image = ret_image + fireants_interpolator(recv_sharded_image, affine=scaled_affine, grid=scaled_disp, mode=mode, padding_mode='zeros', align_corners=align_corners, is_displacement=is_displacement)

        # update new variables to send
        send_sharded_min_coords = recv_sharded_min_coords
        send_sharded_max_coords = recv_sharded_max_coords
        send_sharded_size = recv_sharded_size
        send_sharded_image = recv_sharded_image

        # wait for all sent requests to complete (acts as a barrier)
        [r.wait() for r in reqs]

    return ret_image
        