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
from fireants.utils.opticalflow import OpticalFlow
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample
from fireants.utils.imageutils import scaling_and_squaring
from fireants.utils.imageutils import lie_bracket

class LogDemonsRegistration(AbstractRegistration):
    '''
    This class implements multi-scale log-demons registration

    Returns the warp field in pixel coordinates 
    '''    
    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, tolerance_mode: str = 'atol',
                init_affine: Optional[torch.Tensor] = None,
                cc_kernel_size: int = 3,
                custom_loss: nn.Module = None,
                ##### parameters for log-demons #####
                optical_flow_method: str = 'gauss-newton',
                optical_flow_sigma: float = 0.25,
                optical_flow_eps: float = 1e-3,   # epsilon for minimum gradient value in Thirions, etc.
                update_sigma: float = 0.25,
                eps_prime: float = 0.5,
                use_lie_bracket: bool = False,
                symmetric: bool = False,
                ) -> None:
        # initialize abstract registration
        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images, 
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, 
                         cc_kernel_size=cc_kernel_size, tolerance=tolerance, max_tolerance_iters=max_tolerance_iters, tolerance_mode=tolerance_mode)
        # log-demons requires that fixed and moving images have the same dimensions for computing optical flow
        assert fixed_images.shape == moving_images.shape, "Fixed and moving images must have the same dimensions"
        self.dims = fixed_images.dims
        # this will be a (D+1, D+1) matrix
        if init_affine is None:
            self.affine = torch.eye(self.dims + 1, device=self.device, dtype=torch.float32).repeat(fixed_images.size(), 1, 1)
        else:
            self.affine = init_affine.detach()
        self.symmetric = symmetric
        self._get_velocity_field_fn = self._get_forward_velocity if not symmetric else self._get_symmetric_velocity
        # check for lie bracket
        self.lie_bracket_fn = lie_bracket if use_lie_bracket else lambda *args: 0
        # no optimizer needed for this class
        self.optical_flow = OpticalFlow(optical_flow_method, sigma=optical_flow_sigma, no_grad=True, eps=optical_flow_eps, device=self.device)
        self.eps_prime = eps_prime
        self.update_gaussian = gaussian_1d(torch.tensor(update_sigma, device=self.device), truncated=2, approx='sampled')
        self.velocity_fields = []
        self.integrator_n = 10

    def _get_symmetric_velocity(self,
                              velocity_field, fixed_image_vgrid, fixed_image_affinecoords, 
                              fixed_image_down, moving_image_down, affine_map_inverse, affine_map_forward, permute_idx, scale_factor):
        '''
        Get forward and backward velocities between the fixed and moving images, and average them out
        '''
        # forward velocity
        displacement_field = scaling_and_squaring(velocity_field, fixed_image_vgrid, n=self.integrator_n)
        total_displacement_coords = fixed_image_affinecoords + displacement_field
        moved_image = F.grid_sample(moving_image_down, total_displacement_coords, mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
        v1 = self.optical_flow(fixed_image_down, moved_image)  # [N, dims, H, W, D]
        # v1 = torch.einsum('nij,nj...->ni...', affine_map_inverse, v1)  # [N, dims, H, W, D]
        norm = v1.abs().flatten(1).max(1).values.reshape(-1, 1, *([1]*self.dims)) + 1e-20
        v1 = v1 / norm * scale_factor
        v1 = v1.permute(*permute_idx)  

        # backward velocity (compute M(Ax + b), F(x - \phi(x)) and compute optical flow)
        moved_image_aff = F.grid_sample(moving_image_down, fixed_image_affinecoords, mode='bilinear', align_corners=True)
        # displacement_field_bwd = scaling_and_squaring(-torch.einsum('nij,n...j->n...i', affine_map_forward, velocity_field), fixed_image_vgrid, n=6)
        displacement_field_bwd = scaling_and_squaring(-velocity_field, fixed_image_vgrid, n=self.integrator_n)
        fixed_image_warp = F.grid_sample(fixed_image_down, fixed_image_vgrid + displacement_field_bwd, mode='bilinear', align_corners=True)
        v2 = self.optical_flow(moved_image_aff, fixed_image_warp)
        norm = v2.abs().flatten(1).max(1).values.reshape(-1, *([1]*(self.dims + 1))) + 1e-20
        v2 = v2 / norm * scale_factor
        ## move this to moving image coordinates
        v2 = F.grid_sample(v2, fixed_image_vgrid + displacement_field, mode='bilinear', align_corners=True)
        v2 = v2.permute(*permute_idx)
        # update velocity function
        velocity_field = velocity_field + 0.5*(v1 + 0.5*self.lie_bracket_fn(velocity_field, v1)) + 0.5*(-v2 + 0.5*self.lie_bracket_fn(velocity_field, v2))
        cur_loss = F.mse_loss(moved_image, fixed_image_down)
        return velocity_field, cur_loss, moved_image
        
    def _get_forward_velocity(self, 
                              velocity_field, fixed_image_vgrid, fixed_image_affinecoords, 
                              fixed_image_down, moving_image_down, affine_map_inverse, affine_map_forward, permute_idx, scale_factor):
        '''
        Get the forward velocity between the fixed and moving images by only displacing the moving image

        velocity_field is integrated to get the displacement field, which is added to the affine map
        velocity is computed 
        '''
        displacement_field = scaling_and_squaring(velocity_field, fixed_image_vgrid, n=self.integrator_n)
        total_displacement_coords = fixed_image_affinecoords + displacement_field
        moved_image = F.grid_sample(moving_image_down, total_displacement_coords, mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
        v = self.optical_flow(fixed_image_down, moved_image)  # [N, dims, H, W, D]
        norm = v.abs().flatten(1).max(1).values.reshape(-1, 1, *([1]*self.dims)) + 1e-20
        v = v / (norm) * scale_factor
        v = v.permute(*permute_idx)  # [N, H, W, D, dims]
        cur_loss = F.mse_loss(moved_image, fixed_image_down)
        # update velocity
        velocity_field = velocity_field + v + 0.5 * self.lie_bracket_fn(velocity_field, v)
        return velocity_field, cur_loss, moved_image

    @torch.no_grad()
    def optimize(self, save_transformed=False) -> None:
        '''
        Optimize the log-demons registration given the fixed and moving images
        '''
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_torch2phy()
        moving_p2t = self.moving_images.get_phy2torch()
        fixed_size = fixed_arrays.shape[2:]

        # this affine map is the composition of torch2phy and then the affine matrix
        # the grid is in physical space and begins from the affine map, subsequent deformations will be applied to this
        affine_map_fixed2moving = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]
        affine_map_forward = affine_map_fixed2moving[:, :, :-1]  # [N, dims, dims]
        affine_map_inverse = torch.inverse(affine_map_forward) # [N, dims, dims]
        init_map = torch.eye(self.dims, self.dims+1, device=self.device).unsqueeze(0).repeat(self.fixed_images.size(), 1, 1)

        # keep track of saved images if asked to
        if save_transformed:
            transformed_images = []
        
        velocity_field = None
        permute_idx = (0, 2, 3, 4, 1) if self.dims == 3 else (0, 2, 3, 1)
        # Run iterations
        for scale, iters in zip(self.scales, self.iterations):
            tol_ctr = 0
            prev_loss = np.inf
            # downsample fixed array and retrieve coords
            size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_size]
            # set scale factor
            scale_factor = 2.0/max(fixed_size) * self.eps_prime

            ## create smoothed and downsampled images
            sigmas = 0.5 * torch.tensor([sz/szdown for sz, szdown in zip(fixed_size, size_down)], device=fixed_arrays.device)
            gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
            fixed_image_down = downsample(fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, gaussians=gaussians)
            moving_image_down = downsample(moving_arrays, size=size_down, mode=self.moving_images.interpolate_mode, gaussians=gaussians)

            ### fixed_image_coords gives the coordinates to the affine map directly
            fixed_image_affinecoords = F.affine_grid(affine_map_fixed2moving, fixed_image_down.shape, align_corners=True)  # [N, H, W, [D], dims]
            fixed_image_vgrid  = F.affine_grid(init_map, fixed_image_down.shape, align_corners=True)                 # [N, H, W, [D], dims]

            # initialize velocity field 
            if velocity_field is None:
                velocity_field = torch.zeros_like(fixed_image_vgrid)  # [N, H, W, [D], dims]
            else:
                velocity_field = F.interpolate(velocity_field.permute(0, 4, 1, 2, 3), fixed_image_vgrid.shape[1:-1], mode=self.fixed_images.interpolate_mode, align_corners=True)
                velocity_field = velocity_field.permute(0, 2, 3, 4, 1)

            pbar = tqdm(range(iters))
            for i in pbar:
                # get the velocity field and loss
                velocity_field, cur_loss, moved_image = self._get_velocity_field_fn(velocity_field, fixed_image_vgrid, fixed_image_affinecoords, \
                                                        fixed_image_down, moving_image_down, affine_map_inverse, affine_map_forward, permute_idx, scale_factor)
                # compute update
                # velocity_field = velocity_field + v + 0.5 * self.lie_bracket_fn(velocity_field, v)
                velocity_field = separable_filtering(velocity_field, self.update_gaussian)
                pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, cur_loss))
            # save velocity field and optionally the transformed image
            self.velocity_fields.append(velocity_field+0)
            if save_transformed:
                transformed_images.append(moved_image)

        if save_transformed:
            return transformed_images

