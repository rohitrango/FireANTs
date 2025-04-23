import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
import numpy as np
from typing import List, Optional, Union, Callable
from tqdm import tqdm
import SimpleITK as sitk
import fireants_fused_ops as ffo  # required

torch.backends.cudnn.benchmark = True

# distributed
import torch.distributed as dist
import os

from fireants.utils.globals import MIN_IMG_SHARDED_SIZE as MIN_IMG_SIZE
from fireants.io.image import BatchedImages, FakeBatchedImages
from fireants.registration.abstract import AbstractRegistration
from fireants.registration.deformation.compositivedistributed import CompositiveDistributedWarp
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample
from fireants.utils.warputils import compositive_warp_inverse
from fireants.interpolator import fireants_interpolator
## Deformable utils 
from fireants.registration.deformablemixin import DeformableMixin
from fireants.registration.distributed.utils import *

# logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DistributedGreedyRegistration(AbstractRegistration, DeformableMixin):
    """Greedy deformable registration class for non-linear image alignment.
    
    Args:
        scales (List[int]): Downsampling factors for multi-resolution optimization.
            Must be in descending order (e.g. [4,2,1]).
        iterations (List[float]): Number of iterations to perform at each scale.
            Must be same length as scales.
        fixed_images (BatchedImages): Fixed/reference images to register to.
        moving_images (BatchedImages): Moving images to be registered.
        loss_type (str, optional): Similarity metric to use. Defaults to "cc".
        deformation_type (str, optional): Type of deformation model - 'geodesic' or 'compositive'. 
            Defaults to 'compositive'.
        optimizer (str, optional): Optimization algorithm - 'SGD' or 'Adam'. Defaults to 'Adam'.
        optimizer_params (dict, optional): Additional parameters for optimizer. Defaults to {}.
        optimizer_lr (float, optional): Learning rate for optimizer. Defaults to 0.5.
        integrator_n (Union[str, int], optional): Number of integration steps for geodesic shooting.
            Only used if deformation_type='geodesic'. Defaults to 7.
        mi_kernel_type (str, optional): Kernel type for MI loss. Defaults to 'b-spline'.
        cc_kernel_type (str, optional): Kernel type for CC loss. Defaults to 'rectangular'.
        cc_kernel_size (int, optional): Kernel size for CC loss. Defaults to 3.
        smooth_warp_sigma (float, optional): Gaussian smoothing sigma for warp field. Defaults to 0.5.
        smooth_grad_sigma (float, optional): Gaussian smoothing sigma for gradient field. Defaults to 1.0.
        loss_params (dict, optional): Additional parameters for loss function. Defaults to {}.
        reduction (str, optional): Loss reduction method - 'mean' or 'sum'. Defaults to 'sum'.
        tolerance (float, optional): Convergence tolerance. Defaults to 1e-6.
        max_tolerance_iters (int, optional): Max iterations for convergence check. Defaults to 10.
        init_affine (Optional[torch.Tensor], optional): Initial affine transformation. Defaults to None.
        warp_reg (Optional[Union[Callable, nn.Module]], optional): Regularization on warp field. Defaults to None.
        displacement_reg (Optional[Union[Callable, nn.Module]], optional): Regularization on displacement field. 
            Defaults to None.
        blur (bool, optional): Whether to blur images during downsampling. Defaults to True.
        custom_loss (nn.Module, optional): Custom loss module. Defaults to None.

    Attributes:
        warp: Deformation model (CompositiveDistributedWarp)
            * CompositiveDistributedWarp: Stores the compositive warp field representation of a dense diffeomorphic transform. This class is also used to store free-form deformations.
        affine (torch.Tensor): Initial affine transformation matrix
            * Shape: [N, D, D+1]
            * If init_affine is not provided, it is initialized to identity transform
        smooth_warp_sigma (float): Smoothing sigma for warp field

    """
    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                deformation_type: str = 'compositive',
                optimizer: str = 'Adam', optimizer_params: dict = {},
                optimizer_lr: float = 0.5, 
                integrator_n: Union[str, int] = 7,
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                cc_kernel_size: int = 3,
                smooth_warp_sigma: float = 0.5,
                smooth_grad_sigma: float = 1.0,
                loss_params: dict = {},
                reduction: str = 'mean',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, 
                init_affine: Optional[torch.Tensor] = None,
                warp_reg: Optional[Union[Callable, nn.Module]] = None,
                displacement_reg: Optional[Union[Callable, nn.Module]] = None,
                freeform: bool = False,
                custom_loss: nn.Module = None, **kwargs) -> None:
        # first move the images to cpu
        fixed_images.set_device('cpu')
        moving_images.set_device('cpu')
        # simple 1d world topology
        self.rank = rank = int(os.environ['RANK'])
        self.world_size = world_size = int(os.environ['WORLD_SIZE'])
        self.master_rank = master_rank = int(os.environ.get("MASTER_RANK", 0))
        logger.info(f"Running Distributed Greedy on rank {self.rank} out of {self.world_size} with master_rank {self.master_rank}")
        # initialize abstract registration
        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images, 
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, 
                         loss_params=loss_params,
                         cc_kernel_size=cc_kernel_size, reduction=reduction,
                         tolerance=tolerance, max_tolerance_iters=max_tolerance_iters, **kwargs)
        # change device to current rank
        self.device = torch.device(f'cuda:{self.rank}')
        torch.cuda.set_device(self.device)
        self.dims = fixed_images.dims
        self.reduction = reduction
        # find sharding strategy (spatial dim to shard)
        self.dim_to_shard = get_dim_to_shard(self.dims, world_size, fixed_images.shape, moving_images.shape)
        # shard the image now
        self.fixed_images._shard_dim(self.dim_to_shard, rank, world_size)
        self.moving_images._shard_dim(self.dim_to_shard, rank, world_size)
        self.fixed_images.set_device(self.device)
        self.moving_images.set_device(self.device)

        # get extra padding
        try:
            self.image_padding = self.loss_fn.get_image_padding()
        except:
            raise ValueError('Loss function must have get_image_padding method to support distributed mode')

        # specify regularizations
        self.warp_reg = warp_reg
        self.displacement_reg = displacement_reg
        self.deformation_type = deformation_type
        # specify deformation type
        if deformation_type == 'geodesic':
            raise ValueError('Geodesic deformation is not supported in distributed mode')
        elif deformation_type == 'compositive':
            warp = CompositiveDistributedWarp(self.fixed_images, self.moving_images, optimizer=optimizer,\
                optimizer_lr=optimizer_lr, \
                optimizer_params=optimizer_params, \
                dtype=self.dtype,
                rank=rank, world_size=world_size, master_rank=master_rank, \
                dim_to_shard=self.dim_to_shard, \
                smoothing_grad_sigma=smooth_grad_sigma, smoothing_warp_sigma=smooth_warp_sigma, init_scale=scales[0], freeform=freeform)
        else:
            raise ValueError('Invalid deformation type: {}'.format(deformation_type))
        self.warp = warp
        # initialize affine
        if init_affine is None:
            init_affine = torch.eye(self.dims+1, device=fixed_images.device).unsqueeze(0).repeat(self.opt_size, 1, 1)  # [N, D+1, D+1]
        B, D1, D2 = init_affine.shape
        # affine can be [N, D, D+1] or [N, D+1, D+1]
        if D1 == self.dims+1 and D2 == self.dims+1:
            self.affine = init_affine.detach().to(self.dtype)
        elif D1 == self.dims and D2 == self.dims+1:
            # attach row to affine
            row = torch.zeros(self.opt_size, 1, self.dims+1, device=fixed_images.device)
            row[:, 0, -1] = 1.0
            self.affine = torch.cat([init_affine.detach(), row], dim=1).to(self.dtype)
        else:
            raise ValueError('Invalid initial affine shape: {}'.format(init_affine.shape))

    def get_inverse_warp_parameters(self, fixed_images: Union[BatchedImages, FakeBatchedImages], \
                                             moving_images: Union[BatchedImages, FakeBatchedImages], \
                                             smooth_warp_sigma: float = 0, smooth_grad_sigma: float = 0,
                                             shape=None, displacement=False):
        ''' Get inverse warped coordinates for the moving image.

        This method is useful to analyse the effect of how the moving coordinates (fixed images) are transformed
        '''
        raise NotImplementedError
        moving_arrays = moving_images()
        if shape is None:
            shape = moving_images.shape
        else:
            shape = [moving_arrays.shape[0], 1] + list(shape) 

        # TODO: Change this eventually
        warp = self.warp.get_warp().detach().clone()
        # warp = warp + F.affine_grid(torch.eye(self.dims, self.dims+1, device=warp.device)[None], [1, 1] + list(warp.shape[1:-1]), align_corners=True)
        # get affine warp from displacement map
        # warp = fireants_interpolator.affine_warp(None, warp, align_corners=True)
        warp_inv = compositive_warp_inverse(moving_images, warp, displacement=True) #, smooth_warp_sigma=smooth_warp_sigma, smooth_grad_sigma=smooth_grad_sigma)
        # resample if needed
        mode = "bilinear" if self.dims == 2 else "trilinear"
        if tuple(warp_inv.shape[1:-1]) != tuple(shape[2:]):
            warp_inv = F.interpolate(warp_inv.permute(*self.warp.permute_vtoimg), size=shape[2:], mode=mode, align_corners=True).permute(*self.warp.permute_imgtov)
        
        # get affine transform
        fixed_t2p = fixed_images.get_torch2phy()
        moving_p2t = moving_images.get_phy2torch()
        # save initial affine transform to initialize grid 
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))
        affine_map_inv  = torch.linalg.inv(affine_map_init)
        # get A^-1 * v[y]
        if self.dims == 2:
            warp_inv = torch.einsum('bhwx,byx->bhwy', warp_inv, affine_map_inv[:, :-1, :-1])
        elif self.dims == 3:
            warp_inv = torch.einsum('bhwdx,byx->bhwdy', warp_inv, affine_map_inv[:, :-1, :-1])
        else:
            raise ValueError('Invalid number of dimensions: {}'.format(self.dims))
        #### grid = A^-1 y - A^-1 b = apply A^-1 to regular grid
        return {
            'affine': affine_map_inv[:, :-1].contiguous(),
            'grid': warp_inv,
        }

    def get_warp_parameters(self, fixed_images: Union[BatchedImages, FakeBatchedImages], \
                                     moving_images: Union[BatchedImages, FakeBatchedImages], \
                                     shape=None, displacement=False):
        """Get transformed coordinates for warping the moving image.

        Computes the coordinate transformation from fixed to moving image space
        using the current deformation parameters and optional initial affine transform.

        Args:
            fixed_images (BatchedImages): Fixed reference images
            moving_images (BatchedImages): Moving images to be transformed
            shape (Optional[tuple]): Output shape for coordinate grid.
                Defaults to fixed image shape.
            displacement (bool, optional): Whether to return displacement field instead of
                transformed coordinates. Defaults to False.

        Returns:
            torch.Tensor: If displacement=False, transformed coordinates in normalized [-1,1] space
                Shape: [N, H, W, [D], dims]
                If displacement=True, displacement field, displacements in normalized [-1,1] space
                Shape: [N, H, W, [D], dims]
        """
        raise NotImplementedError

        fixed_arrays = fixed_images() 
        if shape is None:
            shape = fixed_images.shape
        else:
            shape = [fixed_arrays.shape[0], 1] + list(shape) 

        fixed_t2p = fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = moving_images.get_phy2torch().to(self.dtype)
        # save initial affine transform to initialize grid 
        affine_map_init = (torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]).contiguous()
        # set affine coordinates
        warp_field = self.warp.get_warp()

        # resize the warp field if needed
        mode = "bilinear" if self.dims == 2 else "trilinear"
        if tuple(warp_field.shape[1:-1]) != tuple(shape[2:]):
            # interpolate this
            warp_field = F.interpolate(warp_field.permute(*self.warp.permute_vtoimg), size=shape[2:], mode=mode, align_corners=True).permute(*self.warp.permute_imgtov)

        # move these coordinates, and return them
        return {
            'affine': affine_map_init,
            'grid': warp_field,
        }

    def optimize(self, save_transformed=False):
        """Optimize the deformation parameters.

        Performs multi-resolution optimization of the deformation field
        using the configured similarity metric and optimizer. The deformation
        field is optionally smoothed at each iteration.

        Args:
            save_transformed (bool, optional): Whether to save transformed images
                at each scale. Defaults to False.

        Returns:
            Optional[List[torch.Tensor]]: If save_transformed=True, returns list of
                transformed images at each scale. Otherwise returns None.
        """
        fixed_t2p = self.fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = self.moving_images.get_phy2torch().to(self.dtype)
        # get shape of sharded image
        fixed_sharded_size = self.fixed_images.shape[2:]
        moving_sharded_size = self.moving_images.shape[2:]
        # save initial affine transform to initialize grid 
        affine_map_init = (torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]).contiguous().to(self.dtype)
        # get sharded arrays
        fixed_sharded_array = self.fixed_images()
        moving_sharded_array = self.moving_images()

        # to save transformed images
        transformed_images = []
        # multi-scale optimization
        for scale, iters in zip(self.scales, self.iterations):
            self.convergence_monitor.reset()
            # resized image sizes
            # size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_size]
            # moving_size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in moving_size]
            # resized sharded image sizes
            size_sharded_down = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_sharded_size]
            moving_size_sharded_down = [max(int(s / scale), MIN_IMG_SIZE) for s in moving_sharded_size]

            #### Set size for warp field (and let it figure out whether to shard or not)
            self.warp.set_size(size_sharded_down)

            # get sharded state
            # is_state_sharded = self.warp.is_state_sharded
            is_state_sharded = True

            if scale > 1:
                # blur image
                sigmas = 0.5 * torch.tensor([sz/szdown for sz, szdown in zip(fixed_sharded_size, size_sharded_down)], device=self.device, dtype=fixed_sharded_array.dtype)
                gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
                # get "downsampled" image
                fixed_image_down = downsample(fixed_sharded_array, size=size_sharded_down, mode=self.fixed_images.interpolate_mode, gaussians=gaussians)
                moving_image_blur = downsample(moving_sharded_array, size=moving_size_sharded_down, mode=self.moving_images.interpolate_mode, gaussians=gaussians)
            else:
                fixed_image_down = fixed_sharded_array
                moving_image_blur = moving_sharded_array
            
            # allconcat the moving images
            moving_image_blur, moving_gather_stats = gather_and_concat(moving_image_blur, self.rank, self.world_size, self.master_rank, is_state_sharded, self.dim_to_shard)

            # if the state is not sharded, gather the fixed image into rank 0 (default behavior of `gather_and_concat`)
            # else, just add some padding given by the number 
            # if not is_state_sharded:
            #     fixed_image_down, fixed_gather_stats = gather_and_concat(fixed_image_down, self.rank, self.world_size, self.master_rank, is_state_sharded, self.dim_to_shard)
            # else:
            fixed_image_down = add_distributed_padding(fixed_image_down, self.rank, self.world_size, self.image_padding, self.dim_to_shard)
            # Get coordinates to transform
            pbar = tqdm(range(iters)) if self.progress_bar else range(iters)
            # reduce 
            if self.reduction == 'mean':
                scale_factor = 1
            else:
                scale_factor = np.prod(fixed_image_down.shape)
        
            # TODO: from stats, save the min/max coordinates 
            min_coords_moving, max_coords_moving = calculate_bbox_from_gather_stats(moving_gather_stats, self.rank, self.world_size, self.dims)
            print(f"Rank: {self.rank}, min_coords_moving: {min_coords_moving}, max_coords_moving: {max_coords_moving}")

            # run different subroutines depending on whether state is sharded or not
            if is_state_sharded:
                # each rank needs to maintain its own moved image
                for i in pbar:
                    self.warp.set_zero_grad()
                    warp_field = self.warp.get_warp()  # [sharded state]
                    # get coordinates to interpolate
                    moved_image = fireants_interpolator(moving_image_blur, affine=affine_map_init, grid=warp_field, mode='bilinear', align_corners=True, is_displacement=True, min_coords=min_coords_moving, max_coords=max_coords_moving)
                    # pad it 
                    moved_image = add_distributed_padding(moved_image, self.rank, self.world_size, self.image_padding, self.dim_to_shard)
                    loss = self.loss_fn(moved_image, fixed_image_down)
                    # apply regularization on the warp field
                    if self.displacement_reg is not None:
                        loss = loss + self.displacement_reg(warp_field)
                    if self.warp_reg is not None:
                        # internally should use the fireants interpolator to avoid additional memory allocation
                        moved_coords = self.get_warped_coordinates(self.fixed_images, self.moving_images)  
                        loss = loss + self.warp_reg(moved_coords)
                    loss.backward()
                    if self.progress_bar and rank == self.master_rank:
                        pbar.set_description("rank:{}/{}, scale: {}, iter: {}/{}, loss: {:4f}".format(self.rank, self.world_size, scale, i, iters, loss.item()/scale_factor))
                    # optimize the velocity field
                    self.warp.step()
                    # allgather loss to synchronously break
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                    # check for convergence
                    if self.convergence_monitor.converged(loss.item()):
                        break
            else:
                # sharding is not done, problem size is small enough to do on a single gpu
                # run on rank 0
                raise NotImplementedError("Non sharded version not supported for simplicity")
                moved_image = None
                if rank == self.master_rank:
                    for i in pbar:
                        self.warp.set_zero_grad()
                        warp_field = self.warp.get_warp()  # [N, HWD, 3]
                        # move the image
                        moved_image = fireants_interpolator(moving_image_blur, affine=affine_map_init, grid=warp_field, mode='bilinear', align_corners=True, is_displacement=True)
                        loss = self.loss_fn(moved_image, fixed_image_down)
                        # apply regularization on the warp field
                        if self.displacement_reg is not None:
                            loss = loss + self.displacement_reg(warp_field)
                        if self.warp_reg is not None:
                            # internally should use the fireants interpolator to avoid additional memory allocation
                            moved_coords = self.get_warped_coordinates(self.fixed_images, self.moving_images)  
                            loss = loss + self.warp_reg(moved_coords)
                        loss.backward()
                        if self.progress_bar:
                            pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, loss.item()/scale_factor))
                        # optimize the velocity field
                        self.warp.step()
                        # check for convergence
                        if self.convergence_monitor.converged(loss.item()):
                            break

            # save transformed image
            if save_transformed:
                transformed_images.append(moved_image.detach() if isinstance(moved_image, torch.Tensor) else moved_image)
        
        print(f"Rank {rank} finished optimization")
        # cleanup optimization state
        self.warp.optimizer.cleanup()

        if save_transformed:
            return transformed_images
    
    def get_moved_image_fullres(self, ):
        '''
        Get moved image at full resolution 
        '''
        fixed_t2p = self.fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = self.moving_images.get_phy2torch().to(self.dtype)
        # get shape of sharded image
        fixed_sharded_size = self.fixed_images.shape[2:]
        moving_sharded_size = self.moving_images.shape[2:]
        # save initial affine transform to initialize grid 
        affine_map_init = (torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]).contiguous().to(self.dtype)
        # get sharded arrays
        fixed_sharded_array = self.fixed_images()
        moving_sharded_array = self.moving_images()
        # gather moving image
        moving_image_blur, moving_gather_stats = gather_and_concat(moving_sharded_array, self.rank, self.world_size, self.master_rank, True, self.dim_to_shard)
        # Get stats for min and max coords
        min_coords_moving, max_coords_moving = calculate_bbox_from_gather_stats(moving_gather_stats, self.rank, self.world_size, self.dims)
        # get coordinates to interpolate
        moved_image = fireants_interpolator(moving_image_blur, affine=affine_map_init, grid=self.warp.get_warp(), mode='bilinear', align_corners=True, is_displacement=True, min_coords=min_coords_moving, max_coords=max_coords_moving)
        # gather it up again
        del moving_image_blur
        moved_image, _ = gather_and_concat(moved_image, self.rank, self.world_size, self.master_rank, True, self.dim_to_shard)
        return moved_image


if __name__ == '__main__':
    from fireants.io.image import Image
    from time import time
    from torch import distributed as dist
    import os
    from pathlib import Path

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # torch.cuda.memory._record_memory_history()
    # data_path = os.environ['DATAPATH_R']
    # img1 = Image.load_file(f'{data_path}/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t1.nii.gz', device='cpu')
    # img2 = Image.load_file(f'{data_path}/BRATS2021/training/BraTS2021_00597/BraTS2021_00597_t1.nii.gz', device='cpu')

    path = os.environ['DATA_PATH2']
    path = Path(f'{path}/fMOST/subject/')
    img1 = Image.load_file(str(path / "192333_red_mm_SLA.nii.gz"), device='cpu')
    img2 = Image.load_file(str(path / "191820_red_mm_SLA.nii.gz"), device='cpu')
    # clamp
    # img1.array = torch.clamp(img1.array, 20, 150) - 20
    # img2.array = torch.clamp(img2.array, 20, 150) - 20
    # batchify
    fixed = BatchedImages([img1, ])
    moving = BatchedImages([img2, ])
    reg = DistributedGreedyRegistration(scales=[12, 8, 4, 2, 1], iterations=[200, 200, 200, 100, 50], 
                                cc_kernel_size=9,
                                smooth_grad_sigma=0,
                                smooth_warp_sigma=0,
                                # loss_params={'use_ants_gradient': True},
                                fixed_images=fixed, moving_images=moving, 
                                optimizer='Adam', optimizer_lr=0.5, loss_type='fusedcc')
    reg.optimize()
    print(f"Optimized from rank {rank}")

    # get moved image
    print("Gathering moved image")
    moved_image = reg.get_moved_image_fullres().to(torch.uint8)
    if rank == 0:
        print("Saving moved image")
        moved_image = FakeBatchedImages(moved_image, fixed, ignore_size_match=True)
        moved_image.write_image("fmost_moved_image.nii.gz")

    dist.destroy_process_group()

    # torch.cuda.memory._dump_snapshot(f"pkls/greedy_dist_{rank}.pickle")
