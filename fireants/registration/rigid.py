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

    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                optimizer: str = 'SGD', optimizer_params: dict = {},
                optimizer_lr: float = 0.1, 
                loss_params: dict = {},
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, 
                cc_kernel_size: int = 3,
                init_translation: Optional[torch.Tensor] = None,
                scaling: bool = False,
                custom_loss: nn.Module = None, 
                blur: bool = True,
                ) -> None:
        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images, 
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, 
                         loss_params=loss_params,
                         cc_kernel_size=cc_kernel_size,
                         tolerance=tolerance, max_tolerance_iters=max_tolerance_iters)
        # initialize transform
        device = fixed_images.device
        self.dims = dims = self.moving_images.dims
        self.rotation_dims = rotation_dims = dims * (dims - 1) // 2
        self.rotation = nn.Parameter(torch.zeros((fixed_images.size(), rotation_dims), device=device))  # [N, Rd]
        # introduce some scaling parameter
        self.scaling = scaling
        if self.scaling:
            self.logscale = nn.Parameter(torch.zeros((fixed_images.size(), fixed_images.dims), device=device))
        else:
            self.logscale = torch.zeros((fixed_images.size(), fixed_images.dims), device=device)

        self.blur = blur
        # first three params are so(n) variables, last three are translation
        if init_translation is not None:
            transl = init_translation
        else:
            transl = torch.zeros((fixed_images.size(), fixed_images.dims))  # [N, D]
        self.transl = nn.Parameter(transl.to(device))  # [N, D]
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
        if self.dims == 2:
            rotmat = torch.zeros((self.fixed_images.size(), 3, 3), device=self.rotation.device)
            rotmat[:, 2, 2] = 1
            cos, sin = torch.cos(self.rotation[:, 0]), torch.sin(self.rotation[:, 0])
            rotmat[:, 0, 0] = cos
            rotmat[:, 0, 1] = -sin
            rotmat[:, 1, 0] = sin
            rotmat[:, 1, 1] = cos
        elif self.dims == 3:
            rotmat = torch.zeros((self.fixed_images.size(), 4, 4), device=self.rotation.device)
            skew = torch.zeros((self.fixed_images.size(), 3, 3), device=self.rotation.device)
            norm = torch.norm(self.rotation, dim=-1)+1e-8  # [N, 1]
            angle = norm[:, None, None]
            skew[:, 0, 1] = -self.rotation[:, 2]/norm
            skew[:, 0, 2] = self.rotation[:, 1]/norm
            skew[:, 1, 0] = self.rotation[:, 2]/norm
            skew[:, 1, 2] = -self.rotation[:, 0]/norm
            skew[:, 2, 0] = -self.rotation[:, 1]/norm
            skew[:, 2, 1] = self.rotation[:, 0]/norm
            rotmat[:, :3, :3] = torch.eye(3, device=self.rotation.device)[None] + torch.sin(angle) * skew + torch.matmul(skew, skew) * (1 - torch.cos(angle))
            rotmat[:, 3, 3] = 1
        else:
            raise ValueError(f"Dimensions {self.dims} not supported")
        return rotmat
    
    def get_rigid_matrix(self):
        rigidmat = self.get_rotation_matrix() # [N, dim+1, dim+1]
        scale = torch.exp(self.logscale)  # [N, D]
        scale = scale[..., None]
        # N, D = scale.shape
        # scalediag = torch.zeros((N, D, D), device=scale.device)
        # scalediag[:, np.arange(D), np.arange(D)] = scale
        matclone = rigidmat.clone()
        matclone[:, :-1, :-1] = scale * rigidmat[:, :-1, :-1]
        matclone[:, :-1, -1] = self.transl  # [N, dim+1, dim+1]
        return matclone

    def optimize(self, save_transformed=False):
        ''' Given fixed and moving images, optimize rigid registration '''
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_torch2phy()
        moving_p2t = self.moving_images.get_phy2torch()
        fixed_size = fixed_arrays.shape[2:]
        # save initial affine transform to initialize grid 
        init_grid = torch.eye(self.dims, self.dims+1).to(self.fixed_images.device).unsqueeze(0).repeat(self.fixed_images.size(), 1, 1)  # [N, dims, dims+1]
        if save_transformed:
            transformed_images = []

        for scale, iters in zip(self.scales, self.iterations):
            # reset
            self.convergence_monitor.reset()
            prev_loss = np.inf
            # downsample fixed array and retrieve coords
            size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_size]
            # downsample
            if self.blur and scale > 1:
                sigmas = 0.5 * torch.tensor([sz/szdown for sz, szdown in zip(fixed_size, size_down)], device=fixed_arrays.device)
                gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
                fixed_image_down = downsample(fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, gaussians=gaussians)
                moving_image_blur = separable_filtering(moving_arrays, gaussians)
            else:
                fixed_image_down = F.interpolate(fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, align_corners=True)
                moving_image_blur = moving_arrays

            fixed_image_coords = F.affine_grid(init_grid, fixed_image_down.shape, align_corners=True)  # [N, H, W, [D], dims+1]
            fixed_image_coords_homo = torch.cat([fixed_image_coords, torch.ones(list(fixed_image_coords.shape[:-1]) + [1], device=fixed_image_coords.device)], dim=-1)
            fixed_image_coords_homo = torch.einsum('ntd, n...d->n...t', fixed_t2p, fixed_image_coords_homo)  # [N, H, W, [D], dims+1]  
            # print(fixed_image_down.min(), fixed_image_down.max())
            # this is in physical space
            pbar = tqdm(range(iters))
            for i in pbar:
                self.optimizer.zero_grad()
                rigid_matrix = self.get_rigid_matrix()
                coords = torch.einsum('ntd, n...d->n...t', rigid_matrix, fixed_image_coords_homo)  # [N, H, W, [D], dims+1]
                coords = torch.einsum('ntd, n...d->n...t', moving_p2t, coords)  # [N, H, W, [D], dims+1]
                # sample from these coords
                moved_image = F.grid_sample(moving_image_blur, coords[..., :-1], mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
                loss = self.loss_fn(moved_image, fixed_image_down) 
                loss.backward()
                self.optimizer.step()
                # check for convergence
                cur_loss = loss.item()
                if self.convergence_monitor.converged(cur_loss):
                    break
                prev_loss = cur_loss
                pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, prev_loss))
            # save transformed images
            if save_transformed:
                transformed_images.append(moved_image)
        if save_transformed:
            return transformed_images


if __name__ == '__main__':
    from fireants.io.image import Image, BatchedImages
    img1 = Image.load_file('/data/rohitrango/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t1.nii.gz')
    img2 = Image.load_file('/data/rohitrango/BRATS2021/training/BraTS2021_00599/BraTS2021_00599_t1.nii.gz')
    fixed = BatchedImages([img1, ])
    moving = BatchedImages([img2,])
    transform = RigidRegistration([8, 4, 2, 1], [1000, 500, 250, 100], fixed, moving, loss_type='cc', 
                                  scaling=False,
                                  optimizer='SGD', optimizer_lr=5e-3)
    transform.optimize()
    print(transform.rotation.data.cpu().numpy(), transform.transl.data.cpu().numpy(), torch.exp(transform.logscale.data))