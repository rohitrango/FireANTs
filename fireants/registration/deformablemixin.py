# Copyright (c) 2025 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


## Class to inherit common functions to Greedy and SyN

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
import numpy as np
from typing import List, Optional, Union, Callable
from tqdm import tqdm
import SimpleITK as sitk
import logging 
logger = logging.getLogger(__name__)
import os

from fireants.interpolator import fireants_interpolator
from fireants.utils.globals import MIN_IMG_SIZE
from fireants.io.image import BatchedImages
from fireants.registration.abstract import AbstractRegistration
from fireants.registration.deformation.svf import StationaryVelocity
from fireants.registration.deformation.compositive import CompositiveWarp
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample
from fireants.registration.distributed.utils import gather_and_concat, refactor_grid_to_image_stats, calculate_bbox_from_gather_stats, async_send_tensor_ack, async_recv_tensor_ack
from fireants.registration.distributed import parallel_state

class DeformableMixin:
    """Mixin class providing common functionality for deformable registration classes.

    This mixin implements shared methods used by both GreedyRegistration and SyNRegistration
    classes, particularly for saving deformation fields in a format compatible with ANTs
    (Advanced Normalization Tools) and other widely used registration tools.

    The mixin assumes the parent class has:

    - opt_size: Number of registration pairs
    - dims: Number of spatial dimensions
    - fixed_images: BatchedImages object containing fixed images
    - moving_images: BatchedImages object containing moving images
    - get_warped_coordinates(): Method to get transformed coordinates
    """

    @torch.no_grad()
    def save_as_ants_transforms(reg, filenames: Union[str, List[str]]):
        """Save deformation fields in ANTs-compatible format.

        Converts the learned deformation fields to displacement fields in physical space
        and saves them in a format that can be used by ANTs registration tools.
        The displacement fields are saved as multi-component images where each component
        represents the displacement along one spatial dimension.

        Args:
            filenames (Union[str, List[str]]): Path(s) where the transform(s) should be saved.
                If a single string is provided for multiple transforms, it will be treated
                as the first filename. For multiple transforms, provide a list of filenames
                matching the number of transforms.

        Raises:
            AssertionError: If number of filenames doesn't match number of transforms (opt_size)

        !!! caution "Physical Space Coordinates"
            The saved transforms are in physical space coordinates, not normalized [-1,1] space.
            The displacement fields are saved with the same orientation and spacing as the 
            fixed images.
        """
        if "distributed" in reg.__class__.__name__.lower():
            logger.info("Rerouting to distributed transform")
            return reg.save_as_ants_transforms_distributed(filenames)

        if isinstance(filenames, str):
            filenames = [filenames]
        assert len(filenames) == reg.opt_size, "Number of filenames should match the number of warps"
        # get the warp field
        fixed_image: BatchedImages = reg.fixed_images
        moving_image: BatchedImages = reg.moving_images

        # get the moved coordinates and initial grid in pytorch space
        # moved_coords = reg.get_warped_coordinates(fixed_image, moving_image)   # [B, H, W, [D], dim]
        # get affine and displacement parts
        # init grid
        # init_grid = F.affine_grid(torch.eye(reg.dims, reg.dims+1, device=moved_coords.device)[None], \
        #                             fixed_image.shape, align_corners=True)
        # this is now moved displacements
        moving_t2p = moving_image.get_torch2phy()
        fixed_t2p = fixed_image.get_torch2phy()

        # get params
        moved_params = reg.get_warp_parameters(fixed_image, moving_image)   # contains affine and grid
        moved_coords = moved_params['grid']
        affine = moved_params['affine']

        # convert to ants format
        moved_coords = torch.einsum('bij, b...j->b...i', moving_t2p[:, :reg.dims, :reg.dims], moved_coords).contiguous()  # convert to moving space

        # create ants affine
        row = torch.zeros((affine.shape[0], 1, reg.dims+1), device=affine.device, dtype=affine.dtype)
        row[:, 0, -1] = 1
        affine = torch.cat([affine, row], dim=1)  # [b, dim+1, dim+1]
        affine = ((moving_t2p @ affine - fixed_t2p)[:, :reg.dims]).contiguous()

        # create moved_disp
        moved_disp = fireants_interpolator.affine_warp(affine=affine, grid=moved_coords)

        # save 
        for i in range(reg.opt_size):
            moved_disp = moved_coords[i].detach().cpu().numpy()  # [H, W, D, 3]
            savefile = filenames[i]
            # get itk image
            if len(fixed_image.images) < i:     # this image is probably broadcasted then
                itk_data = fixed_image.images[0].itk_image
            else:
                itk_data = fixed_image.images[i].itk_image
            # copy itk data
            warp = sitk.GetImageFromArray(moved_disp)
            warp.CopyInformation(itk_data)
            sitk.WriteImage(warp, savefile)

    @torch.no_grad()
    def save_as_ants_transforms_distributed(reg, filenames: Union[str, List[str]]):
        '''
        Save the warp field in ANTs compatible format for distributed registration.
        This function is called by `save_as_ants_transforms` if the registration is distributed.
        
        Args:
            filenames (Union[str, List[str]]): Path(s) where the transform(s) should be saved.
                If a single string is provided for multiple transforms, it will be treated
                as the first filename. For multiple transforms, provide a list of filenames
                matching the number of transforms.
        
        Raises:
            AssertionError: If number of filenames doesn't match number of transforms (opt_size)
        ''' 
        if "distributed" not in reg.__class__.__name__.lower():
            raise ValueError("This function should only be called by distributed registration")

        if isinstance(filenames, str):
            filenames = [filenames]
        assert len(filenames) == reg.opt_size, "Number of filenames should match the number of warps"
        logger.info(f"Saving {reg.opt_size} transforms to {filenames}")
        # get the warp field
        fixed_image: BatchedImages = reg.fixed_images
        moving_image: BatchedImages = reg.moving_images 
        moved_params = reg.get_warp_parameters(fixed_image, moving_image)   # contains affine and grid
    
        # the idea is that we have Ax + u in torch space,
        # we want Am . (Ax + u) - Af . x  (coordinates translated to moving frame - identity at fixed frame)
        # this is equal to (Am . A - Af) x + Am . u

        # metadata
        moving_t2p = moving_image.get_torch2phy()
        fixed_t2p = fixed_image.get_torch2phy()

        logger.info(f"device: {parallel_state.get_device()}, moving_t2p: {moving_t2p.device}, fixed_t2p: {fixed_t2p.device}, moved_params: {moved_params['grid'].device}")

        # gather grid
        grid = moved_params['grid']   # [B,N,dim]
        affine = moved_params['affine']  # [B, dim, dim+1]
        grid = torch.einsum('bij, b...j->b...i', moving_t2p[:, :reg.dims, :reg.dims], grid).contiguous()  # convert to moving space

        # create ants affine
        row = torch.zeros((affine.shape[0], 1, reg.dims+1), device=affine.device, dtype=affine.dtype)
        row[:, 0, -1] = 1
        affine = torch.cat([affine, row], dim=1)  # [b, dim+1, dim+1]
        affine = ((moving_t2p @ affine - fixed_t2p)[:, :reg.dims]).contiguous()   # this is synchronized

        # synchronize grid stats
        logger.info(f"Gathering grid stats")
        _, grid_gather_stats = gather_and_concat(grid, reg.rank, True, reg.dim_to_shard-1, gather_stats_only=True)
        grid_gather_stats = refactor_grid_to_image_stats(grid_gather_stats) 

        logger.info(f"Calculating bbox")
        min_coords, max_coords = calculate_bbox_from_gather_stats(grid_gather_stats, reg.rank, reg.dims)
        # get updated grid
        logger.info(f"Warping grid")
        grid = fireants_interpolator.affine_warp(affine=affine, grid=grid, min_coords=min_coords, max_coords=max_coords)
        logger.info(f"Grid shape: {grid.shape}")

        # save grid in disk and reload into cpu and concat (too big to fit in GPU)

        for i in range(reg.opt_size):
            grid_i = grid[i].detach()  # [H, W, D, dim]
            additional_grids = [grid_i.cpu()]
            # get grid parallel ids
            rank = parallel_state.get_parallel_state().get_rank()
            grid_parallel_group = parallel_state.get_parallel_state().get_current_gp_group()
            grid_parallel_group_obj = parallel_state.get_parallel_state().gp_group_obj
            master_rank = grid_parallel_group[0]
            is_master = (rank == grid_parallel_group[0])
            logger.info(f"Rank {rank} is master: {is_master}")
            
            # for others, we will send or receive the grid
            for pollrank in grid_parallel_group[1:]:
                # fetch
                if is_master:
                    print(f"Rank {rank} is fetching from {pollrank}")
                    tensor, req, sz = async_recv_tensor_ack(grid_i, pollrank, rank, False) # grid_i is passed for its shape length
                    req.wait()
                    tensor = tensor.cpu()
                    additional_grids.append(tensor)
                else:
                    # im the pollrank, 
                    if rank == pollrank:
                        print(f"Rank {rank} is sending to {master_rank}")
                        grid_i = grid_i.to(parallel_state.get_device())
                        async_send_tensor_ack(grid_i, master_rank, rank, False)
                # distributed
                torch.distributed.barrier(group=grid_parallel_group_obj)
            
            if is_master:
                # gather all grids
                grid_i = torch.cat(additional_grids, dim=reg.dim_to_shard).cpu().numpy()
                # save grid
                savefile = filenames[i]
                # get itk data
                if len(fixed_image.images) < i:     # this image is probably broadcasted then
                    itk_data = fixed_image.images[0].itk_image
                else:
                    itk_data = fixed_image.images[i].itk_image
                print("Writing to SimpleITK image... ")
                # copy itk data
                warp = sitk.GetImageFromArray(grid_i)
                warp.CopyInformation(itk_data)
                sitk.WriteImage(warp, savefile)
                print(f"Saved grid {i} to {savefile}")


        # for i in range(reg.opt_size):
        #     # get grid
        #     # TODO: this is bad but i dont know how to do it better for now
        #     # nccl backend only allows gpu to gpu comm and i want to avoid it to not trigger OOM
        #     grid_i = grid[i].detach().cpu().numpy()  # [H, W, D, dim]
        #     tmp_savefile = filenames[i] + f"_sharded_{reg.rank}_of_{reg.world_size}.npz"
        #     np.savez(tmp_savefile, grid=grid_i)
        #     print(f"Saved sharded grid {i}:{reg.rank}/{reg.world_size} to {tmp_savefile}")
        #     torch.distributed.barrier()

        #     if reg.rank == reg.master_rank:
        #         print("Concatenating grid... ")
        #         # gather all files
        #         grid = []
        #         for j in tqdm(range(reg.world_size), total=reg.world_size):
        #             tmp_file = filenames[i] + f"_sharded_{j}_of_{reg.world_size}.npz"
        #             grid.append(np.load(tmp_file)["grid"])
        #             os.remove(tmp_file)
        #         # grid = torch.concat(grid, dim=reg.dim_to_shard) 
        #         grid = np.concatenate(grid, axis=reg.dim_to_shard)
        #         # save this
        #         savefile = filenames[i]
        #         # get itk image
