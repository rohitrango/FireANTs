'''
author: rohitrango
'''
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union
from fireants.registration.deformation.abstract import AbstractDeformation
from fireants.io.image import Image, BatchedImages
from fireants.utils.imageutils import scaling_and_squaring, _find_integrator_n
from fireants.types import devicetype
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.util import grad_smoothing_hook
from fireants.utils.imageutils import compute_inverse_warp_exp
from fireants.utils.globals import MIN_IMG_SIZE
from copy import deepcopy

class GeodesicShooting(nn.Module, AbstractDeformation):
    '''
    Class for geodesic shooting, by optimizing a velocity field
    '''
    def __init__(self, fixed_images: BatchedImages, moving_images: BatchedImages,
                integrator_n: Union[str, int] = 6, 
                optimizer: str = 'Adam', optimizer_lr: float = 1e-2, optimizer_params: dict = {},
                smoothing_grad_sigma: float = 0.5,
                init_scale: int = 1,
                ) -> None:
        super().__init__()
        self.num_images = num_images = fixed_images.size()
        spatial_dims = fixed_images.shape[2:]  # [H, W, [D]]
        if init_scale > 1:
            spatial_dims = [max(int(s / init_scale), MIN_IMG_SIZE) for s in spatial_dims]
        self.n_dims = len(spatial_dims)  # number of spatial dimensions
        self.device = fixed_images.device
        # permute indices  (image to v and v to image)
        self.permute_imgtov = (0, *range(2, self.n_dims+2), 1)  # [N, HWD, dims] -> [N, HWD, dims] -> [N, dims, HWD]
        self.permute_vtoimg = (0, self.n_dims+1, *range(1, self.n_dims+1))  # [N, dims, HWD] -> [N, HWD, dims]
        # define velocity field
        velocity_field = torch.zeros([num_images, *spatial_dims, self.n_dims], dtype=torch.float32, device=fixed_images.device)  # [N, HWD, dims]
        # attach grad hook if smoothing is required
        self.smoothing_grad_sigma = smoothing_grad_sigma
        if smoothing_grad_sigma > 0:
            self.smoothing_grad_gaussians = [gaussian_1d(s, truncated=2) for s in (torch.zeros(self.n_dims, device=fixed_images.device) + smoothing_grad_sigma)]
        # init grid, velocity field and grad hook
        self.initialize_grid(spatial_dims)
        self.register_parameter('velocity_field', nn.Parameter(velocity_field))
        self.attach_grad_hook()
        # self.velocity_field = nn.Parameter(velocity_field)
        self.integrator_n = integrator_n
        # define optimizer
        self.optimizer = getattr(torch.optim, optimizer)([self.velocity_field], lr=optimizer_lr, **deepcopy(optimizer_params))
        self.optimizer_lr = optimizer_lr
        self.optimizer_name = optimizer
    
    def attach_grad_hook(self):
        ''' attack the grad hook to the velocity field if needed '''
        if self.smoothing_grad_sigma > 0:
            hook = partial(grad_smoothing_hook, gaussians=self.smoothing_grad_gaussians)
            self.velocity_field.register_hook(hook)
    
    def initialize_grid(self, size):
        ''' initialize grid to a size '''
        grid = F.affine_grid(torch.eye(self.n_dims, self.n_dims+1, device=self.device)[None].expand(self.num_images, -1, -1), \
                                  [self.num_images, self.n_dims, *size], align_corners=True)
        self.register_buffer('grid', grid)
        # self.grid = grid

    def set_zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()

    def get_warp(self):
        ''' integrate the velocity field to get the warp '''
        if self.integrator_n == 'auto':
            n = _find_integrator_n(self.velocity_field)
        else:
            n = self.integrator_n
        warp = scaling_and_squaring(self.velocity_field, self.grid, n=n)
        return warp
    
    def get_inverse_warp(self, *args, **kwargs):
        # if self.integrator_n == 'auto':
        #     n = _find_integrator_n(self.velocity_field)
        # else:
        #     n = self.integrator_n
        # invwarp = scaling_and_squaring(-self.velocity_field, self.grid, n=n)
        # return invwarp
        return compute_inverse_warp_exp(self.get_warp().detach(), self.grid)
    
    def set_size(self, size):
        ''' size: [H, W, D] or [H, W] '''
        mode = 'bilinear' if self.n_dims == 2 else 'trilinear'
        # keep old items for copying
        old_shape = self.velocity_field.shape
        old_optimizer_state = self.optimizer.state_dict()
        # get new velocity field
        velocity_field = F.interpolate(self.velocity_field.detach().permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True, 
                    ).permute(*self.permute_imgtov)
        velocity_field = nn.Parameter(velocity_field)
        self.register_parameter('velocity_field', velocity_field)
        self.attach_grad_hook()
        # self.velocity_field = velocity_field
        self.initialize_grid(size)
        self.optimizer = getattr(torch.optim, self.optimizer_name)([self.velocity_field], lr=self.optimizer_lr)
        # TODO: copy state variables from old optimizer
        state_dict = old_optimizer_state['state']
        old_optimizer_state['param_groups'] = self.optimizer.state_dict()['param_groups']
        for g in state_dict.keys():
            for k, v in state_dict[g].items():
                # this is probably a state of the tensor
                if isinstance(v, torch.Tensor) and v.shape == old_shape:
                    state_dict[g][k] = F.interpolate(v.permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True, 
                        ).permute(*self.permute_imgtov)
        #         if isinstance(v, torch.Tensor):
        #             print(k, v.shape)
        #         else:
        #             print(k, v)
        # input("Here.")
        self.optimizer.load_state_dict(old_optimizer_state)


if __name__ == '__main__':
    img1 = Image.load_file('/data/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t1.nii.gz')
    img2 = Image.load_file('/data/BRATS2021/training/BraTS2021_00597/BraTS2021_00597_t1.nii.gz')
    fixed = BatchedImages([img1, ])
    moving = BatchedImages([img2,])
    deformation = GeodesicShooting(fixed, moving )
    for i in range(100):
        deformation.set_zero_grad() 
        w = deformation.get_warp()
        loss = ((w-1/155)**2).mean()
        print(loss)
        loss.backward()
        deformation.step()
