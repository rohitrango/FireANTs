from fireants.registration.abstract import AbstractRegistration
from typing import List, Optional
import torch
from torch import nn
from fireants.io.image import BatchedImages
from torch.optim import SGD, Adam
from torch.nn import functional as F
from fireants.utils.globals import MIN_IMG_SIZE
from tqdm import tqdm
import numpy as np
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample

class AffineRegistration(AbstractRegistration):
    """Affine registration class for linear image alignment.

    This class implements affine registration between fixed and moving images using
    gradient descent optimization. Affine transformations are linear transformations that 
    include translation, rotation, scaling, and shearing operations.

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
        mi_kernel_type (str, optional): Kernel type for MI loss. Defaults to 'b-spline'.
        cc_kernel_type (str, optional): Kernel type for CC loss. Defaults to 'rectangular'.
        cc_kernel_size (int, optional): Kernel size for CC loss. Defaults to 3.
        tolerance (float, optional): Convergence tolerance. Defaults to 1e-6.
        max_tolerance_iters (int, optional): Max iterations for convergence. Defaults to 10.
        init_rigid (Optional[torch.Tensor], optional): Initial affine matrix. Defaults to None.
        custom_loss (nn.Module, optional): Custom loss module. Defaults to None.
        blur (bool, optional): Whether to blur images during downsampling. Defaults to True.
        moved_mask (bool, optional): Whether to mask moved image for loss. Defaults to False.
        loss_device (Optional[str], optional): Device to compute loss on. Defaults to None.

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
                optimizer_lr: float = 1e-3,
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                cc_kernel_size: int = 3,
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, 
                init_rigid: Optional[torch.Tensor] = None,
                custom_loss: nn.Module = None,
                blur: bool = True,
                moved_mask: bool = False,   # mask out moved image for loss
                loss_device: Optional[str] = None, **kwargs,
                ) -> None:

        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images, 
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, loss_params=loss_params,
                         cc_kernel_size=cc_kernel_size, tolerance=tolerance, max_tolerance_iters=max_tolerance_iters, **kwargs)
        device = self.device
        dims = self.dims
        self.blur = blur
        self.loss_device = loss_device
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
            affine = torch.eye(dims, dims+1).unsqueeze(0).repeat(self.opt_size, 1, 1)  # [N, D, D+1]

        # set parameters
        self.affine = nn.Parameter(affine.to(device).to(self.dtype))  # [N, D]
        self.row = torch.zeros((self.opt_size, 1, dims+1), device=device, dtype=self.dtype)   # keep this to append to affine matrix
        self.row[:, 0, -1] = 1.0
        # TODO: change this to move it 
        self.moved_mask = moved_mask
        # optimizer
        if optimizer == 'SGD':
            self.optimizer = SGD([self.affine], lr=optimizer_lr, **optimizer_params)
        elif optimizer == 'Adam':
            self.optimizer = Adam([self.affine], lr=optimizer_lr, **optimizer_params)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")
    
    def get_affine_matrix(self, homogenous=True):
        """Get the current affine transformation matrix.

        Args:
            homogenous (bool, optional): Whether to return homogeneous coordinates. 
                Defaults to True.

        Returns:
            torch.Tensor: Affine transformation matrix.
                If homogenous=True: shape [N, D+1, D+1]
                If homogenous=False: shape [N, D, D+1]
        """
        return torch.cat([self.affine, self.row], dim=1) if homogenous else self.affine
    
    def get_warped_coordinates(self, fixed_images: BatchedImages, moving_images: BatchedImages, shape=None):
        """Get transformed coordinates for warping the moving image.

        Computes the coordinate transformation from fixed to moving image space
        using the current affine parameters.

        Args:
            fixed_images (BatchedImages): Fixed reference images
            moving_images (BatchedImages): Moving images to be transformed
            shape (Optional[tuple]): Output shape for coordinate grid. 
                Defaults to fixed image shape.

        Returns:
            torch.Tensor: Transformed coordinates in normalized [-1,1] space
                Shape: [N, H, W, [D], dims]
        """
        # get coordinates of shape fixed image
        fixed_t2p = fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = moving_images.get_phy2torch().to(self.dtype)
        affinemat = self.get_affine_matrix()
        if shape is None:
            shape = fixed_images.shape
        # coords = F.affine_grid(init_grid, shape, align_corners=True)  # [N, H, W, [D], dims+1]
        # coords = torch.cat([coords, torch.ones(list(coords.shape[:-1]) + [1], device=coords.device, dtype=self.dtype)], dim=-1)
        # modify coords to be in physical space -> moving_p2t
        # coords = torch.einsum('ntd, n...d->n...t', fixed_t2p, coords)  # [N, H, W, [D], dims+1]  
        # coords = torch.einsum('ntd, n...d->n...t', affinemat, coords)  # [N, H, W, [D], dims+1]
        # coords = torch.einsum('ntd, n...d->n...t', moving_p2t, coords)  # [N, H, W, [D], dims+1]
        # modify coords to be in physical space -> moving_p2t
        # coords = torch.einsum('ntd, n...d->n...t', moving_p2t @ affinemat @ fixed_t2p, coords)  # [N, H, W, [D], dims+1]
        affine = (moving_p2t @ affinemat @ fixed_t2p)[:, :-1]
        coords = F.affine_grid(affine, shape, align_corners=True)  # [N, H, W, [D], dims+1]
        return coords

    def optimize(self, save_transformed=False):
        """Optimize the affine registration parameters.

        Performs multi-resolution optimization of the affine transformation
        parameters using the configured similarity metric and optimizer.

        Args:
            save_transformed (bool, optional): Whether to save transformed images
                at each scale. Defaults to False.

        Returns:
            Optional[List[torch.Tensor]]: If save_transformed=True, returns list of
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
        init_grid = torch.eye(self.dims, self.dims+1).to(self.fixed_images.device, self.dtype).unsqueeze(0).repeat(self.opt_size, 1, 1)  # [N, dims, dims+1]

        if save_transformed:
            transformed_images = []

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
                moving_image_blur = separable_filtering(moving_image_blur, gaussians)
            else:
                if scale > 1:
                    fixed_image_down = F.interpolate(fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, align_corners=True)
                else:
                    fixed_image_down = fixed_arrays
                moving_image_blur = moving_arrays


            # fixed_image_coords = F.affine_grid(init_grid, fixed_image_down.shape, align_corners=True)  # [N, H, W, [D], dims+1]
            # fixed_image_coords = torch.cat([fixed_image_coords, torch.ones(list(fixed_image_coords.shape[:-1]) + [1], device=fixed_image_coords.device, dtype=self.dtype)], dim=-1)
            # fixed_image_coords = torch.einsum('ntd, n...d->n...t', fixed_t2p, fixed_image_coords)  # [N, H, W, [D], dims+1]  
            # print(fixed_image_down.min(), fixed_image_down.max())
            # this is in physical space
            pbar = tqdm(range(iters)) if verbose else range(iters)
            torch.cuda.empty_cache()
            for i in pbar:
                self.optimizer.zero_grad()
                affinemat = (moving_p2t @ self.get_affine_matrix() @ fixed_t2p)[:, :-1]
                # coords = torch.einsum('ntd, n...d->n...t', affinemat, fixed_image_coords)  # [N, H, W, [D], dims+1]
                # coords = torch.einsum('ntd, n...d->n...t', moving_p2t, coords)  # [N, H, W, [D], dims+1]
                # coords = torch.einsum('ntd, n...d->n...t', moving_p2t @ affinemat, fixed_image_coords)  # [N, H, W, [D], dims+1]
                coords = F.affine_grid(affinemat, fixed_image_down.shape, align_corners=True)  # [N, H, W, [D], dims+1]
                # sample from these coords
                moved_image = F.grid_sample(moving_image_blur, coords.to(moving_image_blur.dtype), mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
                if self.moved_mask:
                    moved_mask = F.grid_sample(torch.ones_like(moving_image_blur), coords.to(moving_image_blur.dtype), mode='nearest', align_corners=True)
                else:
                    moved_mask = None

                if self.loss_device is not None:
                    moved_image = moved_image.to(self.loss_device)
                    fixed_image_down = fixed_image_down.to(self.loss_device)
                    if moved_mask is not None:
                        moved_mask = moved_mask.to(self.loss_device)
                # calculate loss function
                loss = self.loss_fn(moved_image, fixed_image_down, mask=moved_mask)
                loss.backward()
                # print(self.transl.grad, self.rotation.grad)
                self.optimizer.step()
                # check for convergence
                cur_loss = loss.item()
                if self.convergence_monitor.converged(cur_loss):
                    break
                prev_loss = cur_loss
                if verbose:
                    pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, prev_loss))
            if save_transformed:
                transformed_images.append(moved_image)

        if save_transformed:
            return transformed_images


if __name__ == '__main__':
    from fireants.io.image import Image, BatchedImages
    import torch
    import traceback
    torch.cuda.memory._record_memory_history()
    img_dtype = torch.bfloat16
    # img1 = Image.load_file('/data/rohitrango/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t2.nii.gz', dtype=img_dtype)
    # img2 = Image.load_file('/data/rohitrango/BRATS2021/training/BraTS2021_00599/BraTS2021_00599_t2.nii.gz', dtype=img_dtype)

    ### works at native resolution with bf16 and PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512" and mse loss
    img1 = Image.load_file("/mnt/rohit_data2/fMOST/subject/15257_red_mm_IRA.nii.gz", dtype=img_dtype)
    img2 = Image.load_file("/mnt/rohit_data2/fMOST/subject/17109_red_mm_SLA.nii.gz", dtype=img_dtype)
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