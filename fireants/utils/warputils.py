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


import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from fireants.utils.util import catchtime
from typing import Optional, List

from scipy.ndimage import zoom

from fireants.io.image import BatchedImages, Image
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.types import ItemOrList

# from fireants.registration import GreedyRegistration
## using this to prevent circular import
import fireants.registration as regmodule

## Will contain standard warp processing functions
class InverseConsistencyOperator(nn.Module):
    '''
    Multi-scale inverse consistency operator (to compute inverse of warp)
    '''
    def __init__(self, ref_warp: torch.Tensor, image: BatchedImages):
        super().__init__()
        self.ref_warp = ref_warp.detach()
        self.image_size = image.shape[2:]
        self.dims = len(ref_warp.shape) - 2
        if self.dims == 3:  
            self.permute_vtoimg = (0, 4, 1, 2, 3)       # [b, h, w, d, c] -> [b, c, h, w, d]
            self.permute_imgtov = (0, 2, 3, 4, 1)
        elif self.dims == 2:
            self.permute_vtoimg = (0, 3, 1, 2)
            self.permute_imgtov = (0, 2, 3, 1)
        else:
            raise ValueError('Only 2D and 3D warps are supported')
    
    def forward(self, warp: torch.Tensor):
        ''' compute the inverse consistency loss '''
        shape = [1, 1] + list(warp.shape[1:-1])
        scale = [float(t)/float(s) for s, t in zip(shape[2:], self.image_size)]  # shape will only be downsampled versions of image_size
        scale = max(scale)    # some dimensions may be downsampled to 32 instead of H/s
        assert scale >= 1, f"Scale factor is {scale}, which is less than 1"
        # ref_shape = [1, 1] + list(self.ref_warp.shape[1:-1])
        # if ref_shape != shape:
        if scale > 1:
            linear = 'bilinear' if self.dims == 2 else 'trilinear'
            ref_warp_i = F.interpolate(self.ref_warp.permute(*self.permute_vtoimg), scale_factor=1.0/scale, mode=linear, align_corners=True).permute(*self.permute_imgtov)
        else:
            ref_warp_i = self.ref_warp
        # init grids
        grid = F.affine_grid(torch.eye(self.dims, self.dims+1, device=warp.device)[None], shape, align_corners=True)
        ref_grid = F.affine_grid(torch.eye(self.dims, self.dims+1, device=warp.device)[None], [1, 1] + list(ref_warp_i.shape[1:-1]), align_corners=True)
        # psi(phi(x))
        ## compute u * phi(x) instead of psi(phi(x)) because u = 0 outside of grid sample range, but not so much for psi
        loss1 = F.grid_sample((warp - grid).permute(*self.permute_vtoimg), ref_warp_i, mode='bilinear', align_corners=True).permute(*self.permute_imgtov) + ref_warp_i
        loss1 = F.mse_loss(loss1, ref_grid)
        # phi(psi(y))
        loss2 = F.grid_sample((ref_warp_i - ref_grid).permute(*self.permute_vtoimg), warp, mode='bilinear', align_corners=True).permute(*self.permute_imgtov) + warp
        loss2 = F.mse_loss(loss2, grid)
        return loss1 + loss2


class ShapeAveragingOperator(nn.Module):
    def __init__(self, ref_warp: torch.Tensor):
        super().__init__()
        self.ref_warp = ref_warp.detach()
        self.dims = len(ref_warp.shape) - 2
        if self.dims == 3:  
            self.permute_vtoimg = (0, 4, 1, 2, 3)       # [b, h, w, d, c] -> [b, c, h, w, d]
            self.permute_imgtov = (0, 2, 3, 4, 1)
            self.mode = 'trilinear'
        elif self.dims == 2:
            self.permute_vtoimg = (0, 3, 1, 2)
            self.permute_imgtov = (0, 2, 3, 1)
            self.mode = 'bilinear'
        else:
            raise ValueError('Only 2D and 3D warps are supported')
    
    def forward(self, warp: torch.Tensor):
        ''' compute the euclidean difference '''
        shape = warp.shape[1:-1]
        ref_shape = self.ref_warp.shape[1:-1]
        if shape != ref_shape:
            ref_warp_i = F.interpolate(self.ref_warp.permute(*self.permute_vtoimg), warp.shape[1:-1], mode=self.mode, align_corners=True)
            ref_warp_i = ref_warp_i.permute(*self.permute_imgtov)
        else:
            ref_warp_i = self.ref_warp
        loss = F.mse_loss(warp, ref_warp_i)
        return loss

def shape_averaging_invwarp( 
                  template_image: BatchedImages, ref_warp: torch.Tensor,
                  scales: List[float] = [4, 2, 1],
                  iterations: List[int] = [200, 100, 50],
    ):
    ''' 
    Optimize the warp using the template image and the reference warp
    '''
    reg = regmodule.GreedyRegistration( 
        scales=scales,
        iterations=iterations,
        fixed_images=template_image,
        moving_images=template_image,
        loss_type='noop',
        deformation_type='compositive',
        optimizer='adam',
        optimizer_lr=0.5,
        warp_reg=ShapeAveragingOperator(ref_warp),
        smooth_grad_sigma=1.0,
        smooth_warp_sigma=0.25,
    )
    reg.optimize(False)
    inverse_warp = compositive_warp_inverse(template_image, reg.warp.get_warp().detach(), displacement=False)
    return inverse_warp


def compositive_warp_inverse(image: BatchedImages, ref_warp: torch.Tensor,
                        scales: List[float] = [8, 4, 2, 1], 
                        iterations: List[int] = [200, 200, 100, 50],
                        smooth_grad_sigma: float = 0.0,
                        smooth_warp_sigma: float = 0.0,
                        displacement: bool = False,
                        ):
    '''
    Utility to compute the inverse of a compositive warp
    '''
    reg = regmodule.GreedyRegistration( 
        scales=scales,
        iterations=iterations,
        fixed_images=image,
        moving_images=image,
        loss_type='noop',
        deformation_type='compositive',
        optimizer='adam',
        optimizer_lr=0.5,
        warp_reg=InverseConsistencyOperator(ref_warp, image),
        smooth_grad_sigma=smooth_grad_sigma,
        smooth_warp_sigma=smooth_warp_sigma,
    )
    ## initialize
    with torch.no_grad():
        warp = reg.warp.warp.data
        B = warp.shape[0]
        shape = list(warp.shape[1:-1])
        linear = 'bilinear' if reg.dims == 2 else 'trilinear'
        grid = F.affine_grid(torch.eye(reg.dims, reg.dims+1, device=warp.device)[None], [1,1] + shape, align_corners=True)
        ref_warp_resized = F.interpolate(ref_warp.permute(*reg.warp.permute_vtoimg), shape, mode=linear, align_corners=True).permute(*reg.warp.permute_imgtov)
        warp.data.copy_(grid - ref_warp_resized)
        del grid, ref_warp_resized
    # reg.warp.
    reg.optimize(False)
    return reg.get_warped_coordinates(image, image, displacement=displacement)

### Utility to convert dense warp fields from pytorch format to scipy format
### Used to submit to learn2reg challenge 
def dense_warp_to_scipy_format(disp_field: torch.Tensor, zoom_factor: Optional[float] = None):
    ''' Convert a dense warp field to a scipy format '''
    dims = len(disp_field.shape) - 2
    if dims == 2:
        raise ValueError('Only 3D warps are supported (for now)')
    elif dims == 3:
        disps = []
        B, H, W, D, _ = disp_field.shape
        for b in range(B):
            disp = disp_field[b].detach().cpu().numpy()
            # [ZYX, [xyz]] -> [XYZ, [xyz]]
            disp = disp.transpose(2, 1, 0, 3)
            disp[..., 0] *= (D-1)/2
            disp[..., 1] *= (W-1)/2
            disp[..., 2] *= (H-1)/2
            if zoom_factor is not None:
                disp = [zoom(disp[..., i], zoom_factor, order=2) for i in range(3)]
                disp = np.stack(disp, axis=-1)
            disp = disp.transpose(3, 0, 1, 2)
            return disp
    else:
        raise ValueError('Only 2D and 3D warps are supported')
