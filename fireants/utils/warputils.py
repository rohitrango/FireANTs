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
from fireants.registration import GreedyRegistration

## Will contain standard warp processing functions

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
    reg = GreedyRegistration( 
        scales=scales,
        iterations=iterations,
        fixed_images=template_image,
        moving_images=template_image,
        loss_type='noop',
        deformation_type='compositive',
        optimizer='adam',
        optimizer_lr=0.5,
        optimizer_params={
            'optimize_inverse_warp': True
        },
        warp_reg=ShapeAveragingOperator(ref_warp),
        smooth_grad_sigma=1.0,
        smooth_warp_sigma=0.25,
    )
    reg.optimize(False)
    inverse_warp = reg.warp.get_inverse_warp(n_iters=20)  # [B, H, W, [D], dims]
    dims = len(inverse_warp.shape) - 2
    if dims == 2:
        B, H, W, _ = inverse_warp.shape
        grid = F.affine_grid(torch.eye(2, 3, device=inverse_warp.device)[None], (B, 1, H, W), align_corners=True).expand(B, -1, -1, -1)
    else:
        B, H, W, D, _ = inverse_warp.shape
        grid = F.affine_grid(torch.eye(3, 4, device=inverse_warp.device)[None], (B, 1, H, W, D), align_corners=True).expand(B, -1, -1, -1, -1)
    # add the grid to the inverse warp
    inverse_warp = inverse_warp + grid
    return inverse_warp

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
