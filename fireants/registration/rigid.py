from fireants.registration.abstract import AbstractRegistration
from typing import List, Optional
import torch
from torch import nn
from fireants.io.image import BatchedImages
from torch.optim import SGD, Adam
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample
from fireants.utils.globals import MIN_IMG_SIZE

class RigidRegistration(AbstractRegistration):
    """Rigid registration class for 2D and 3D image registration.

    This class implements rigid registration (rotation and translation) with optional anisotropic scaling.
    The transformation is parameterized using:
        - Rotation: Uses Lie algebra so(n) for 2D/3D rotations
        - Translation: Direct parameterization in physical space
        - Scaling (optional): Log-scale parameters for each dimension

    Args:
        scales (List[float]): Downsampling factors for multi-resolution optimization
            Must be in descending order (e.g. [4,2,1]).
        iterations (List[int]): Number of iterations at each scale
            Must match length of scales.
        fixed_images (BatchedImages): Fixed/reference images
        moving_images (BatchedImages): Moving images to be registered
        loss_type (str, optional): Similarity metric ('cc', 'mi', 'mse', 'custom', 'noop'). Default: 'cc'
        optimizer (str, optional): Optimization algorithm ('Adam' or 'SGD'). Default: 'Adam'
        optimizer_params (dict, optional): Additional parameters for optimizer. Default: {}
        optimizer_lr (float, optional): Learning rate for optimizer. Default: 3e-3
        loss_params (dict, optional): Additional parameters for loss function. Default: {}
        mi_kernel_type (str, optional): Kernel type for MI loss. Default: 'b-spline'
        cc_kernel_type (str, optional): Kernel type for CC loss. Default: 'rectangular'
        tolerance (float, optional): Convergence tolerance. Default: 1e-6
        max_tolerance_iters (int, optional): Max iterations for convergence. Default: 10
        cc_kernel_size (int, optional): Kernel size for CC loss. Default: 3
        init_translation (Optional[torch.Tensor], optional): Initial translation. Default: None
        init_moment (Optional[torch.Tensor], optional): Initial rotation moment. Default: None
        scaling (bool, optional): Whether to optimize scaling parameters. Default: False
        custom_loss (nn.Module, optional): Custom loss module. Default: None
        blur (bool, optional): Whether to apply Gaussian blur during downsampling. Default: True

    Attributes:
        rotation (nn.Parameter): Rotation parameters in so(n)
        transl (nn.Parameter): Translation parameters
        logscale (nn.Parameter): Log-scale parameters (if scaling=True)
        moment (torch.Tensor): Current rotation moment matrix
        optimizer: Optimizer instance (Adam or SGD)
    """

    def __init__(self, scales: List[float], iterations: List[int], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                optimizer: str = 'Adam', optimizer_params: dict = {},
                optimizer_lr: float = 3e-3,
                loss_params: dict = {},
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, 
                cc_kernel_size: int = 3,
                init_translation: Optional[torch.Tensor] = None,
                init_moment: Optional[torch.Tensor] = None,
                scaling: bool = False,
                custom_loss: nn.Module = None, 
                blur: bool = True, 
                **kwargs
                ) -> None:
        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images, 
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, 
                         loss_params=loss_params,
                         cc_kernel_size=cc_kernel_size,
                         tolerance=tolerance, max_tolerance_iters=max_tolerance_iters, **kwargs)
        # initialize transform
        device = fixed_images.device
        self.dims = dims = self.moving_images.dims
        self.rotation_dims = rotation_dims = dims * (dims - 1) // 2
        self.rotation = nn.Parameter(torch.zeros((self.opt_size, rotation_dims), device=device, dtype=self.dtype))  # [N, Rd]
        # introduce some scaling parameter
        self.scaling = scaling
        if self.scaling:
            self.logscale = nn.Parameter(torch.zeros((self.opt_size, fixed_images.dims), device=device, dtype=self.dtype))
        else:
            self.logscale = torch.zeros((self.opt_size, fixed_images.dims), device=device, dtype=self.dtype)

        self.blur = blur
        # first three params are so(n) variables, last three are translation
        if init_translation is not None:
            transl = init_translation.to(device, dtype=self.dtype)
        else:
            transl = torch.zeros((self.opt_size, fixed_images.dims), dtype=self.dtype)  # [N, D]
        self.transl = nn.Parameter(transl.to(device, self.dtype))  # [N, D]
        # set init moment
        if init_moment is not None:
            self.moment = init_moment.to(device, self.dtype)
        else:
            self.moment = torch.eye(dims, device=device, dtype=self.dtype).unsqueeze(0).repeat(self.opt_size, 1, 1)
        # optimizer
        params = [self.rotation, self.transl]
        if scaling:
            params.append(self.logscale)
        
        if optimizer == 'SGD':
            self.optimizer = SGD(params, lr=optimizer_lr, **optimizer_params)
        elif optimizer == 'Adam':
            self.optimizer = Adam(params, lr=optimizer_lr, **optimizer_params)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")
    
    def get_rotation_matrix(self):
        """Compute the rotation matrix from so(n) parameters.

        For 2D: Uses direct angle parameterization
        For 3D: Uses Rodriguez formula to compute matrix exponential

        Returns:
            torch.Tensor: Batch of rotation matrices [N, dim+1, dim+1]
        """
        if self.dims == 2:
            rotmat = torch.zeros((self.opt_size, 3, 3), device=self.rotation.device, dtype=self.dtype)
            rotmat[:, 2, 2] = 1
            cos, sin = torch.cos(self.rotation[:, 0]), torch.sin(self.rotation[:, 0])
            rotmat[:, 0, 0] = cos
            rotmat[:, 0, 1] = -sin
            rotmat[:, 1, 0] = sin
            rotmat[:, 1, 1] = cos
        elif self.dims == 3:
            rotmat = torch.zeros((self.opt_size, 4, 4), device=self.rotation.device, dtype=self.dtype)
            skew = torch.zeros((self.opt_size, 3, 3), device=self.rotation.device, dtype=self.dtype)
            norm = torch.norm(self.rotation, dim=-1)+1e-8  # [N, 1]
            angle = norm[:, None, None]
            skew[:, 0, 1] = -self.rotation[:, 2]/norm
            skew[:, 0, 2] = self.rotation[:, 1]/norm
            skew[:, 1, 0] = self.rotation[:, 2]/norm
            skew[:, 1, 2] = -self.rotation[:, 0]/norm
            skew[:, 2, 0] = -self.rotation[:, 1]/norm
            skew[:, 2, 1] = self.rotation[:, 0]/norm
            rotmat[:, :3, :3] = torch.eye(3, device=self.rotation.device, dtype=self.dtype)[None] + torch.sin(angle) * skew + torch.matmul(skew, skew) * (1 - torch.cos(angle))
            rotmat[:, 3, 3] = 1
        else:
            raise ValueError(f"Dimensions {self.dims} not supported")
        
        # premulitply by moment
        rotmat[:, :self.dims, :self.dims] = rotmat[:, :self.dims, :self.dims] @ self.moment
        rotmat = rotmat.to(self.rotation.device, self.rotation.dtype)
        return rotmat 
    
    def get_rigid_matrix(self, homogenous=True):
        """Compute the complete rigid transformation matrix.

        Combines rotation, translation and optional scaling into a single matrix.

        Args:
            homogenous (bool, optional): Whether to return homogeneous matrix. Default: True

        Returns:
            torch.Tensor: If homogenous=True: [N, dim+1, dim+1] transformation matrices
                         If homogenous=False: [N, dim, dim+1] transformation matrices
        """
        rigidmat = self.get_rotation_matrix() # [N, dim+1, dim+1]
        scale = torch.exp(self.logscale)  # [N, D]
        scale = scale[..., None]
        # N, D = scale.shape
        # scalediag = torch.zeros((N, D, D), device=scale.device)
        # scalediag[:, np.arange(D), np.arange(D)] = scale
        matclone = rigidmat.clone()
        matclone[:, :-1, :-1] = scale * rigidmat[:, :-1, :-1]
        matclone[:, :-1, -1] = self.transl  # [N, dim+1, dim+1]
        return matclone if homogenous else matclone[:, :-1, :]  # [N, dim, dim+1]

    def get_warped_coordinates(self, fixed_images: BatchedImages, moving_images: BatchedImages, shape=None):
        """Compute transformed coordinates for the rigid registration.

        Applies the rigid transformation (rotation, translation, scaling) to map
        coordinates from fixed image space to moving image space.

        Args:
            fixed_images (BatchedImages): Fixed/reference images
            moving_images (BatchedImages): Moving images
            shape (Optional[tuple]): Output shape for coordinate grid

        Returns:
            torch.Tensor: Transformed coordinates in normalized [-1,1] space
                         Shape: [N, H, W, [D], dims]
        """
        fixed_t2p = fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = moving_images.get_phy2torch().to(self.dtype)
        rigid_matrix = self.get_rigid_matrix()
        if shape is None:
            shape = fixed_images.shape
        # init grid and apply rigid transformation
        # init_grid = torch.eye(self.dims, self.dims+1).to(self.fixed_images.device, self.dtype).unsqueeze(0).repeat(rigid_matrix.shape[0], 1, 1)  # [N, dims, dims+1]
        # coords = F.affine_grid(init_grid, shape, align_corners=True)  # [N, H, W, [D], dims+1]
        # coords = torch.cat([coords, torch.ones(list(coords.shape[:-1]) + [1], device=coords.device, dtype=self.dtype)], dim=-1)
        # coords = torch.einsum('ntd, n...d->n...t', rigid_matrix, coords)  # [N, H, W, [D], dims+1]
        rigidmat = (moving_p2t @ rigid_matrix @ fixed_t2p)[:, :-1]
        coords = F.affine_grid(rigidmat, shape, align_corners=True)  # [N, H, W, [D], dims+1]
        return coords.to(self.dtype)
    
    def optimize(self, save_transformed=False):
        """Optimize the rigid registration parameters.

        Performs multi-resolution optimization of the rigid transformation parameters
        using the configured similarity metric and optimizer.

        Args:
            save_transformed (bool, optional): Whether to save transformed images at each scale.
                                             Default: False

        Returns:
            Optional[List[torch.Tensor]]: If save_transformed=True, returns list of transformed
                                        images at each scale. Otherwise returns None.
        """
        ''' Given fixed and moving images, optimize rigid registration '''
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
            # reset
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

            # print(fixed_image_down.min(), fixed_image_down.max())
            # this is in physical space
            pbar = tqdm(range(iters)) if self.progress_bar else range(iters)
            for i in pbar:
                self.optimizer.zero_grad()
                rigid_matrix = self.get_rigid_matrix()
                # coords = torch.einsum('ntd, n...d->n...t', rigid_matrix, fixed_image_coords)  # [N, H, W, [D], dims+1]
                # coords = torch.einsum('ntd, n...d->n...t', moving_p2t, coords)  # [N, H, W, [D], dims+1]
                mat = (moving_p2t @ rigid_matrix @ fixed_t2p)[:, :-1]
                coords = F.affine_grid(mat, fixed_image_down.shape, align_corners=True)  # [N, H, W, [D], dims+1]
                # sample from these coords
                moved_image = F.grid_sample(moving_image_blur, coords.to(moving_image_blur.dtype), mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
                loss = self.loss_fn(moved_image, fixed_image_down) 
                loss.backward()
                self.optimizer.step()
                # check for convergence
                cur_loss = loss.item()
                if self.convergence_monitor.converged(cur_loss):
                    break
                prev_loss = cur_loss
                if self.progress_bar:
                    pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, prev_loss))
            # save transformed images
            if save_transformed:
                transformed_images.append(moved_image)
        if save_transformed:
            return transformed_images


if __name__ == '__main__':
    from fireants.io.image import Image, BatchedImages
    import torch
    import traceback
    torch.cuda.memory._record_memory_history()

    img1 = Image.load_file('/data/rohitrango/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t1.nii.gz')
    img2 = Image.load_file('/data/rohitrango/BRATS2021/training/BraTS2021_00599/BraTS2021_00599_t1.nii.gz')

    ## works at native resolution with bf16 and PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512" and mse loss
    # img1 = Image.load_file("/mnt/rohit_data2/fMOST/subject/15257_red_mm_IRA.nii.gz", dtype=torch.bfloat16)
    # img2 = Image.load_file("/mnt/rohit_data2/fMOST/subject/17109_red_mm_SLA.nii.gz", dtype=torch.bfloat16)
    print(img1.shape, img2.shape)
    fixed = BatchedImages([img1, ])
    moving = BatchedImages([img2,])
    # iterations = [1000, 500, 250, 100]
    iterations = [100, 50, 20, 10]
    transform = RigidRegistration([8, 4, 2, 1], iterations, fixed, moving, loss_type='mse', 
                                  scaling=False, 
                                  optimizer='Adam', optimizer_lr=5e-3, dtype=torch.float32)
    try:
        transform.optimize()
    except torch.OutOfMemoryError as e:
        print(e)
        traceback.print_exc()

    print(transform.rotation.data.float().cpu().numpy(), transform.transl.data.float().cpu().numpy(), torch.exp(transform.logscale.data.float()))
    torch.cuda.memory._dump_snapshot("rigid_big.pkl")