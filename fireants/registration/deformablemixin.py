## Class to inherit common functions to Greedy and SyN

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
import numpy as np
from typing import List, Optional, Union, Callable
from tqdm import tqdm
import SimpleITK as sitk

from fireants.utils.globals import MIN_IMG_SIZE
from fireants.io.image import BatchedImages
from fireants.registration.abstract import AbstractRegistration
from fireants.registration.deformation.svf import StationaryVelocity
from fireants.registration.deformation.compositive import CompositiveWarp
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample


class DeformableMixin:

    def save_as_ants_transforms(reg, filenames: Union[str, List[str]]):
        ''' given a list of filenames, save the warp field as ants transforms '''
        if isinstance(filenames, str):
            filenames = [filenames]
        assert len(filenames) == reg.opt_size, "Number of filenames should match the number of warps"
        # get the warp field
        fixed_image: BatchedImages = reg.fixed_images
        moving_image: BatchedImages = reg.moving_images
        # get the moved coordinates
        moved_coords = reg.get_warped_coordinates(fixed_image, moving_image)   # [B, H, W, [D], dim]
        init_grid = F.affine_grid(torch.eye(reg.dims, reg.dims+1, device=moved_coords.device)[None], \
                                    fixed_image.shape, align_corners=True)
        # this is now moved displacements
        moved_coords = moved_coords - init_grid

        # convert this grid into moving coordinates 
        moving_t2p = moving_image.get_torch2phy()[:, :reg.dims, :reg.dims]
        moved_coords = torch.einsum('bij, b...j->b...i', moving_t2p, moved_coords)
        # save 
        for i in range(reg.opt_size):
            moved_disp = moved_coords[i].detach().cpu().numpy()  # [H, W, D, 3]
            savefile = filenames[i]
            # get itk image
            if len(fixed_image.images) < i:     # this image is probably broadcasted then
                itk_data = fixed_image.images[0].itk_image
            else:
                itk_data = fixed_image.images[i].itk_image
            # copy itk data
            warp = sitk.GetImageFromArray(moved_disp)
            warp.CopyInformation(itk_data)
            sitk.WriteImage(warp, savefile)
