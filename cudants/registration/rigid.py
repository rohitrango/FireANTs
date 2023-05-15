from cudants.registration.abstract import AbstractRegistration
from typing import List, Optional
import torch
from torch import nn
from cudants.io.image import BatchedImages
from torch.optim import SGD, Adam
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from cudants.losses.cc import gaussian_1d, separable_filtering
from cudants.utils.imageutils import downsample
from cudants.utils.globals import MIN_IMG_SIZE

class RigidRegistration(AbstractRegistration):

    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                optimizer: str = 'SGD', optimizer_params: dict = {},
                optimizer_lr: float = 0.1, optimizer_momentum: float = 0.0,
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, tolerance_mode: str = 'atol',
                init_translation: Optional[torch.Tensor] = None,
                custom_loss: nn.Module = None, 
                blur: bool = True,
                ) -> None:
        super().__init__(scales, iterations, fixed_images, moving_images, loss_type, mi_kernel_type, cc_kernel_type, custom_loss,
                         tolerance, max_tolerance_iters, tolerance_mode)
        # initialize transform
        device = fixed_images.device
        self.dims = dims = self.moving_images.dims
        self.rotation_dims = rotation_dims = dims * (dims - 1) // 2
        self.rotation = nn.Parameter(torch.zeros((fixed_images.size(), rotation_dims), device=device))  # [N, Rd]
        self.blur = blur
        # first three params are so(n) variables, last three are translation
        if init_translation is not None:
            transl = init_translation
        else:
            transl = torch.zeros((fixed_images.size(), fixed_images.dims))  # [N, D]
        self.transl = nn.Parameter(transl.to(device))  # [N, D]
        # optimizer
        if optimizer == 'SGD':
            self.optimizer = SGD([self.rotation, self.transl], lr=optimizer_lr, momentum=optimizer_momentum, **optimizer_params)
        elif optimizer == 'Adam':
            self.optimizer = Adam([self.rotation, self.transl], lr=optimizer_lr, **optimizer_params)
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
        rigidmat[:, :-1, -1] = self.transl  # [N, dim+1, dim+1]
        return rigidmat

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
            tol_ctr = 0
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
                if self.tolerance_mode == 'atol':
                    if (prev_loss - cur_loss) < self.tolerance:
                        tol_ctr+=1
                        if tol_ctr > self.max_tolerance_iters:
                            break
                    else:
                        tol_ctr=0
                elif self.tolerance_mode == 'rtol':
                    if (prev_loss - cur_loss)/(prev_loss) < self.tolerance:
                        tol_ctr+=1
                        if tol_ctr > self.max_tolerance_iters:
                            break
                    else:
                        tol_ctr=0
                prev_loss = cur_loss
                pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, prev_loss))
            # save transformed images
            if save_transformed:
                transformed_images.append(moved_image)
        if save_transformed:
            return transformed_images


if __name__ == '__main__':
    from cudants.io.image import Image, BatchedImages
    img1 = Image.load_file('/data/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t1.nii.gz')
    img2 = Image.load_file('/data/BRATS2021/training/BraTS2021_00599/BraTS2021_00599_t1.nii.gz')
    fixed = BatchedImages([img1, ])
    moving = BatchedImages([img2,])
    transform = RigidRegistration([8, 4, 2, 1], [1000, 500, 250, 100], fixed, moving, loss_type='cc', optimizer='SGD', optimizer_lr=5e-3)
    transform.optimize()
    print(transform.rotation.data.cpu().numpy(), transform.transl.data.cpu().numpy())