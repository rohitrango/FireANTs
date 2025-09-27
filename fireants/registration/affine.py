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


from fireants.registration.abstract import AbstractRegistration
from typing import List, Optional, Union
import torch
from torch import nn
from fireants.io.image import BatchedImages, FakeBatchedImages
from torch.optim import SGD, Adam
from torch.nn import functional as F
from fireants.utils.globals import MIN_IMG_SIZE
from tqdm import tqdm
import numpy as np
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample
from fireants.utils.util import check_and_raise_cond, check_correct_ext, any_extension, augment_filenames, savetxt
# from scipy.io import savemat
from fireants.utils.util import save_itk_affine as savemat
from fireants.utils.globals import PERMITTED_ANTS_TXT_EXT, PERMITTED_ANTS_MAT_EXT
import logging
from fireants.interpolator import fireants_interpolator
logger = logging.getLogger(__name__)


class AffineRegistration(AbstractRegistration):
    """Affine registration class for linear image alignment.

    This class implements affine registration between fixed and moving images using
    gradient descent optimization. Affine transformations are linear transformations that 
    include translation, rotation, scaling, and shearing operations.

    Note about initialization and optimization:
     - All initializations assume the format y = Ax + t (rigid, affine, moments)
     - However, optimization works better with the format y = A(x-c) + c + t'  (where c is the center of the image)
     - therefore, we need to compute t' = t - c + Ac as the learnable parameter if `around_center=True`

    Args:
        scales (List[float]): Downsampling factors for multi-resolution optimization.
            Must be in descending order (e.g. [4,2,1]).
        iterations (List[int]): Number of iterations to perform at each scale.
            Must be same length as scales.
        fixed_images (BatchedImages): Fixed/reference images to register to.
        moving_images (BatchedImages): Moving images to be registered.
        loss_type (str, optional): Similarity metric to use. Defaults to "cc".
        optimizer (str, optional): Optimization algorithm - 'SGD' or 'Adam'. Defaults to 'SGD'.
        optimizer_params (dict, optional): Additional parameters for optimizer. Defaults to {}.
        loss_params (dict, optional): Additional parameters for loss function. Defaults to {}.
        optimizer_lr (float, optional): Learning rate for optimizer. Defaults to 0.1.
        mi_kernel_type (str, optional): Kernel type for MI loss. Defaults to 'gaussian'
        cc_kernel_type (str, optional): Kernel type for CC loss. Defaults to 'rectangular'.
        cc_kernel_size (int, optional): Kernel size for CC loss. Defaults to 3.
        tolerance (float, optional): Convergence tolerance. Defaults to 1e-6.
        max_tolerance_iters (int, optional): Max iterations for convergence. Defaults to 10.
        init_rigid (Optional[torch.Tensor], optional): Initial affine matrix. Defaults to None.
        custom_loss (nn.Module, optional): Custom loss module. Defaults to None.
        blur (bool, optional): Whether to blur images during downsampling. Defaults to True.
        around_center (bool, optional): Whether to apply affine around the center of the image. Defaults to True.

    Attributes:
        affine (nn.Parameter): Learnable affine transformation matrix [N, D, D+1]
        row (torch.Tensor): Bottom row for homogeneous coordinates [N, 1, D+1]
        optimizer: SGD or Adam optimizer instance
    """

    def __init__(self, scales: List[float], iterations: List[int], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                optimizer: str = 'Adam', optimizer_params: dict = {},
                loss_params: dict = {},
                optimizer_lr: float = 3e-2,
                mi_kernel_type: str = 'gaussian', cc_kernel_type: str = 'rectangular',
                cc_kernel_size: int = 3,
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, 
                around_center: bool = True,
                init_rigid: Optional[torch.Tensor] = None,
                custom_loss: nn.Module = None,
                blur: bool = True,
                **kwargs
                ) -> None:

        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images, 
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, loss_params=loss_params,
                         cc_kernel_size=cc_kernel_size, tolerance=tolerance, max_tolerance_iters=max_tolerance_iters, **kwargs)
        device = self.device
        dims = self.dims
        self.blur = blur
        # first three params are so(n) variables, last three are translation
        if init_rigid is not None:
            B, D1, D2 = init_rigid.shape
            if D1 == dims and D2 == dims+1:
                affine = init_rigid
            elif D1 == dims+1 and D2 == dims+1:
                affine = init_rigid[:, :-1, :] + 0
            else:
                raise ValueError(f"init_rigid must have shape [N, {dims}, {dims+1}] or [N, {dims+1}, {dims+1}], got {init_rigid.shape}")
        else:
            affine = torch.eye(dims, dims+1).unsqueeze(0).repeat(self.opt_size, 1, 1).to(device)  # [N, D, D+1]
        
        affine = affine.to(self.dtype)

        # whether to apply affine around the center of the image
        self.around_center = around_center 
        self.center = self.fixed_images.get_torch2phy()[:, :self.dims, -1].contiguous()  # the center of the fixed image is (0, 0, ..., 0) in torch space => A0 + t = t in physical space
        self.center = self.center.to(device).to(self.dtype)
        if self.around_center:
            # we recalibrate the t' parameter from t 
            transl = affine[:, :self.dims, -1]
            transl = transl - self.center + (affine[:, :self.dims, :self.dims] @ self.center[..., None]).squeeze(-1)
            affine[:, :self.dims, -1] = transl
            affine = affine.detach().contiguous()

        self.affine = nn.Parameter(affine.to(device).to(self.dtype))  # [N, D]
        self.row = torch.zeros((self.opt_size, 1, dims+1), device=device, dtype=self.dtype)   # keep this to append to affine matrix
        self.row[:, 0, -1] = 1.0
        # optimizer
        if optimizer == 'SGD':
            self.optimizer = SGD([self.affine], lr=optimizer_lr, **optimizer_params)
        elif optimizer == 'Adam':
            self.optimizer = Adam([self.affine], lr=optimizer_lr, **optimizer_params)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")
    
    def get_inverse_warp_parameters(self, fixed_images: Union[BatchedImages, FakeBatchedImages], moving_images: Union[BatchedImages, FakeBatchedImages], shape=None):
        raise NotImplementedError("Inverse warped coordinates not implemented for affine registration")

    def save_as_ants_transforms(self, filenames: Union[str, List[str]]):
        ''' 
        Save the registration as ANTs transforms (.mat file)
        '''
        if isinstance(filenames, str):
            filenames = [filenames]

        affine = self.get_affine_matrix(homogenous=False)
        n = affine.shape[0]
        check_and_raise_cond(len(filenames)==1 or len(filenames)==n, "Number of filenames must match the number of transforms")
        check_and_raise_cond(check_correct_ext(filenames, PERMITTED_ANTS_TXT_EXT + PERMITTED_ANTS_MAT_EXT), "File extension must be one of {}".format(PERMITTED_ANTS_TXT_EXT + PERMITTED_ANTS_MAT_EXT))
        filenames = augment_filenames(filenames, n, PERMITTED_ANTS_TXT_EXT + PERMITTED_ANTS_MAT_EXT)

        for i in range(affine.shape[0]):
            mat = affine[i].detach().cpu().numpy().astype(np.float32)
            A = mat[:self.dims, :self.dims]
            t = mat[:self.dims, -1]
            if any_extension(filenames[i], PERMITTED_ANTS_MAT_EXT):
                dims = self.dims
                savemat(filenames[i], {f'AffineTransform_float_{dims}_{dims}': mat, 'fixed': np.zeros((self.dims, 1)).astype(np.float32)})
            else:
                savetxt(filenames[i], A, t)
            logger.info(f"Saved transform to {filenames[i]}")

    def get_affine_matrix(self, homogenous=True):
        """Get the current affine transformation matrix.

        Always get it in the format y = Ax + t

        Args:
            homogenous (bool, optional): Whether to return homogeneous coordinates. 
                Defaults to True.

        Returns:
            torch.Tensor: Affine transformation matrix.
                If homogenous=True: shape [N, D+1, D+1]
                If homogenous=False: shape [N, D, D+1]
        """
        affine = self.affine.clone()
        if self.around_center:  # we need to convert t' to t
            A = affine[:, :self.dims, :self.dims] + 0
            t = affine[:, :self.dims, -1] + 0
            t = t + self.center - (A @ self.center[..., None]).squeeze(-1)
            affine = torch.cat([A, t[..., None]], dim=-1)
        return torch.cat([affine, self.row], dim=1).contiguous() if homogenous else affine.contiguous()

    def get_warp_parameters(self, fixed_images: Union[BatchedImages, FakeBatchedImages], moving_images: Union[BatchedImages, FakeBatchedImages], shape=None):
        """Get transformed coordinates for warping the moving image.

        Computes the coordinate transformation from fixed to moving image space
        using the current affine parameters.

        Args:
            fixed_images (BatchedImages): Fixed reference images
            moving_images (BatchedImages): Moving images to be transformed
            shape (Optional[tuple]): Output shape for coordinate grid. 
                Defaults to fixed image shape.

        Returns:
            dict: Dictionary containing:
                - 'affine': Affine transformation matrix [N, D, D+1]
                - 'out_shape': Output shape for coordinate grid [N, 1, H, W, [D]]
        """
        # get coordinates of shape fixed image
        fixed_t2p = fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = moving_images.get_phy2torch().to(self.dtype)
        affinemat = self.get_affine_matrix()
        if shape is None:
            shape = fixed_images.shape
        affine = ((moving_p2t @ affinemat @ fixed_t2p)[:, :-1]).contiguous().to(self.dtype)
        return {
            'affine': affine,
            'out_shape': shape,
        }

    def optimize(self):
        """Optimize the affine registration parameters.

        Performs multi-resolution optimization of the affine transformation
        parameters using the configured similarity metric and optimizer.

        Args:
            None

        Returns:
            None
                transformed images at each scale. Otherwise returns None.
        """
        ''' Given fixed and moving images, optimize rigid registration '''
        verbose = self.progress_bar
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = self.moving_images.get_phy2torch().to(self.dtype)
        fixed_size = fixed_arrays.shape[2:]
        # save initial affine transform to initialize grid 

        for scale, iters in zip(self.scales, self.iterations):
            self.convergence_monitor.reset()
            prev_loss = np.inf
            # downsample fixed array and retrieve coords
            size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_size]
            mov_size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in moving_arrays.shape[2:]]
            # downsample
            if self.blur and scale > 1:
                sigmas = 0.5 * torch.tensor([sz/szdown for sz, szdown in zip(fixed_size, size_down)], device=fixed_arrays.device, dtype=moving_arrays.dtype)
                gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
                fixed_image_down = downsample(fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, gaussians=gaussians)
                moving_image_blur = downsample(moving_arrays, size=mov_size_down, mode=self.moving_images.interpolate_mode, gaussians=gaussians)
            else:
                if scale > 1:
                    fixed_image_down = F.interpolate(fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, align_corners=True)
                else:
                    fixed_image_down = fixed_arrays
                moving_image_blur = moving_arrays

            # this is in physical space
            pbar = tqdm(range(iters)) if verbose else range(iters)
            torch.cuda.empty_cache()
            for i in pbar:
                self.optimizer.zero_grad()
                affinemat = ((moving_p2t @ self.get_affine_matrix() @ fixed_t2p)[:, :-1]).contiguous().to(self.dtype)
                # sample from these coords
                moved_image = fireants_interpolator(moving_image_blur, affine=affinemat, 
                                out_shape=fixed_image_down.shape, mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
                # calculate loss function
                loss = self.loss_fn(moved_image, fixed_image_down)
                loss.backward()
                self.optimizer.step()
                # check for convergence
                cur_loss = loss.item()
                if self.convergence_monitor.converged(cur_loss):
                    break
                prev_loss = cur_loss
                if verbose:
                    pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, prev_loss))


if __name__ == '__main__':
    from fireants.io.image import Image, BatchedImages
    import torch
    import traceback
    import os
    torch.cuda.memory._record_memory_history()
    img_dtype = torch.bfloat16
    # path = os.environ['DATAPATH_R']
    # img1 = Image.load_file(f'{path}/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t2.nii.gz', dtype=img_dtype)
    # img2 = Image.load_file(f'{path}/BRATS2021/training/BraTS2021_00599/BraTS2021_00599_t2.nii.gz', dtype=img_dtype)

    ### works at native resolution with bf16 and PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512" and mse loss
    path = os.environ['DATA_PATH2']
    img1 = Image.load_file(f"{path}/fMOST/subject/15257_red_mm_IRA.nii.gz", dtype=img_dtype)
    img2 = Image.load_file(f"{path}/fMOST/subject/17109_red_mm_SLA.nii.gz", dtype=img_dtype)
    fixed = BatchedImages([img1, ])
    moving = BatchedImages([img2,])
    # iterations = [1000, 500, 250, 100]
    iterations = [100, 50, 20, 10]
    transform = AffineRegistration([8, 4, 2, 1], iterations, fixed, moving, loss_type='mse', optimizer='Adam', optimizer_lr=1e-3, tolerance=1e-3, dtype=torch.float32)
    try:
        transform.optimize()
    except torch.OutOfMemoryError as e:
        print(e)
        traceback.print_exc()
    print(np.around(transform.affine.data.cpu().numpy(), 4))
    torch.cuda.memory._dump_snapshot("affine_big.pkl")