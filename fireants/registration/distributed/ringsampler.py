import torch
from typing import Optional
from fireants.interpolator import fireants_interpolator
from fireants.registration.distributed import parallel_state
import logging
logger = logging.getLogger(__name__)

def get_affine_rescaling_ringsampler(min_img_coords: torch.Tensor, max_img_coords: torch.Tensor, image_size: Optional[torch.Tensor] = None, align_corners: bool = True) -> torch.Tensor:
    '''
    We need to compute ---- s @ (Ax + u)
    This function computes s
    

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


@torch.no_grad
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
    is_displacement: bool = True
) -> torch.Tensor:
    '''
    Forward pass of the distributed grid sampler

    Distributed grid sampler for 3D images. This function performs grid sampling across multiple GPUs in a ring-like fashion.
    Each GPU holds a portion of the full image and samples from it using the provided grid coordinates.
    The results are aggregated across all GPUs in the grid parallel group.

    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, D, H, W)
        min_img_coords (torch.Tensor): Minimum coordinates of the image in torch space
        max_img_coords (torch.Tensor): Maximum coordinates of the image in torch space  
        affine (torch.Tensor, optional): Affine transformation matrix of shape (B, 3, 4)
        grid (torch.Tensor, optional): Displacement field of shape (B, D, H, W, 3)
        mode (str): Interpolation mode, one of 'bilinear' or 'nearest'
        padding_mode (str): Padding mode, one of 'zeros', 'border', or 'reflection'
        align_corners (bool): Whether to align corners when sampling
        min_coords (tuple, optional): Minimum coordinates of the output grid
        max_coords (tuple, optional): Maximum coordinates of the output grid
        out_shape (tuple, optional): Shape of the output grid
        is_displacement (bool): Whether the grid represents displacements (must be True)

    Returns:
        torch.Tensor: Sampled image tensor of shape (B, C, D, H, W)

    '''
    # get rank and world size
    rank = parallel_state.get_parallel_state().get_rank()
    gp_group = parallel_state.get_parallel_state().get_current_gp_group()
    gp_size = len(gp_group)
    local_rank = gp_group.index(rank)

    assert is_displacement, "is_displacement must be True"
    assert mode in ['bilinear'], "only bilinear mode is supported"
    assert min_coords is not None and max_coords is not None, "min_coords and max_coords must be provided"

    # Get base image
    image_size = image.shape[-3:]
    scale_factor = get_affine_rescaling_ringsampler(min_img_coords, max_img_coords, image_size, align_corners)
    # scale_affine
    scaled_affine = scale_factor @ make_homogenous(affine) if affine is not None else scale_factor
    scaled_affine = scaled_affine[:, :3, :]  # (B, 3, 4)

    scaled_disp = torch.einsum('bij,b...j->b...i', scale_factor[:, :3, :3], grid).contiguous()
    # sample image
    ret_image = fireants_interpolator(image, affine=scaled_affine, grid=scaled_disp, mode=mode, padding_mode='zeros', min_coords=min_coords, max_coords=max_coords, align_corners=align_corners, is_displacement=is_displacement)

    # fetch from the previous rank
    # prev_rank = gp_group[local_rank - 1]
    # next_rank = gp_group[(local_rank + 1) % gp_size]

    # prepare variables to send 
    send_sharded_image = image
    send_sharded_size = torch.tensor(list(image.shape), device=image.device)
    send_sharded_min_coords = min_img_coords
    send_sharded_max_coords = max_img_coords
    # prepare variables to receive
    recv_sharded_min_coords = torch.zeros_like(min_img_coords)
    recv_sharded_max_coords = torch.zeros_like(max_img_coords)
    recv_sharded_size = torch.zeros_like(send_sharded_size)

    print(f"affine = {scale_factor @ make_homogenous(affine)}, rank = {rank}")

    for ring_id in range(1, gp_size):
        # get the previous and next rank according to ring id size
        prev_rank = gp_group[(local_rank - ring_id + gp_size) % gp_size]
        next_rank = gp_group[(local_rank + ring_id) % gp_size]
        # need to use batch isend irecv ops
        parallel_state.ring_collect_op(send_tensors=[send_sharded_min_coords, send_sharded_max_coords, send_sharded_size], recv_tensors=[recv_sharded_min_coords, recv_sharded_max_coords, recv_sharded_size], src=prev_rank, dst=next_rank)

        # Create image tensor and post its receive
        recv_sharded_image = torch.zeros(recv_sharded_size.to(torch.long).tolist(), device=image.device, dtype=image.dtype)

        parallel_state.ring_collect_op(send_tensors=[send_sharded_image], recv_tensors=[recv_sharded_image], src=prev_rank, dst=next_rank)

        ### sample the image
        scale_factor = get_affine_rescaling_ringsampler(recv_sharded_min_coords, recv_sharded_max_coords, recv_sharded_size, align_corners)
        # scale_affine
        scaled_affine = scale_factor @ make_homogenous(affine) if affine is not None else scale_factor
        scaled_affine = scaled_affine[:, :3, :].contiguous()  # (B, 3, 4)

        scaled_disp = torch.einsum('bij,b...j->b...i', scale_factor[:, :3, :3], grid).contiguous()
        # sample image
        ret_image = ret_image + fireants_interpolator(recv_sharded_image, affine=scaled_affine, grid=scaled_disp, mode=mode, padding_mode='zeros', min_coords=min_coords, max_coords=max_coords, align_corners=align_corners, is_displacement=is_displacement)

    return ret_image
        
# ***********************************************************
# *************  Distributed Grid Sampler  ******************
# ***********************************************************

class RingSampler3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, min_img_coords, max_img_coords, affine, grid, mode, padding_mode, align_corners, min_coords, max_coords, is_displacement):
        # 
        ctx.save_for_backward(image, min_img_coords, max_img_coords, affine, grid, mode, padding_mode, align_corners, min_coords, max_coords, is_displacement)
        return distributed_grid_sampler_3d(image, min_img_coords, max_img_coords, affine, grid, mode, padding_mode, align_corners, min_coords, max_coords, is_displacement)
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None, None, None, None, None, None, None


ring_sampler_3d_fn = RingSampler3D.apply