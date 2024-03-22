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
    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                optimizer: str = 'SGD', optimizer_params: dict = {},
                loss_params: dict = {},
                optimizer_lr: float = 0.1, 
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                cc_kernel_size: int = 3,
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, 
                init_rigid: Optional[torch.Tensor] = None,
                custom_loss: nn.Module = None,
                blur: bool = True,
                moved_mask: bool = False,   # mask out moved image for loss
                loss_device: Optional[str] = None,
                ) -> None:

        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images, 
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, loss_params=loss_params,
                         cc_kernel_size=cc_kernel_size, tolerance=tolerance, max_tolerance_iters=max_tolerance_iters)
        device = self.device
        dims = self.dims
        self.blur = blur
        self.loss_device = loss_device
        # first three params are so(n) variables, last three are translation
        if init_rigid is not None:
            affine = init_rigid
        else:
            affine = torch.eye(dims, dims+1).unsqueeze(0).repeat(fixed_images.size(), 1, 1)  # [N, D, D+1]
        self.affine = nn.Parameter(affine.to(device))  # [N, D]
        self.row = torch.zeros((fixed_images.size(), 1, dims+1), device=device)   # keep this to append to affine matrix
        self.row[:, 0, -1] = 1.0
        self.moved_mask = moved_mask
        # optimizer
        if optimizer == 'SGD':
            self.optimizer = SGD([self.affine], lr=optimizer_lr, **optimizer_params)
        elif optimizer == 'Adam':
            self.optimizer = Adam([self.affine], lr=optimizer_lr, **optimizer_params)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")
    
    def get_affine_matrix(self):
        return torch.cat([self.affine, self.row], dim=1)

    def optimize(self, save_transformed=False, verbose=True):
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
            self.convergence_monitor.reset()
            prev_loss = np.inf
            # downsample fixed array and retrieve coords
            size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_size]

            ## create fixed downsampled image, and blurred moving image
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
            if verbose:
                pbar = tqdm(range(iters))
            else:
                pbar = range(iters)
            torch.cuda.empty_cache()
            for i in pbar:
                self.optimizer.zero_grad()
                affinemat = self.get_affine_matrix()
                coords = torch.einsum('ntd, n...d->n...t', affinemat, fixed_image_coords_homo)  # [N, H, W, [D], dims+1]
                coords = torch.einsum('ntd, n...d->n...t', moving_p2t, coords)  # [N, H, W, [D], dims+1]
                # sample from these coords
                moved_image = F.grid_sample(moving_image_blur, coords[..., :-1], mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
                if self.moved_mask:
                    moved_mask = F.grid_sample(torch.ones_like(moving_image_blur), coords[..., :-1], mode='nearest', align_corners=True)
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
    img1 = Image.load_file('/data/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t2.nii.gz')
    img2 = Image.load_file('/data/BRATS2021/training/BraTS2021_00599/BraTS2021_00599_t2.nii.gz')
    fixed = BatchedImages([img1, ])
    moving = BatchedImages([img2,])
    transform = AffineRegistration([8, 4, 2, 1], [1000, 500, 250, 100], fixed, moving, loss_type='cc', optimizer='SGD', optimizer_lr=1e-3, tolerance=0)
    transform.optimize()
    print(np.around(transform.affine.data.cpu().numpy(), 4))
