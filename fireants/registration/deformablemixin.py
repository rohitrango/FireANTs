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
    """Mixin class providing common functionality for deformable registration classes.

    This mixin implements shared methods used by both GreedyRegistration and SyNRegistration
    classes, particularly for saving deformation fields in a format compatible with ANTs
    (Advanced Normalization Tools) and other widely used registration tools.

    The mixin assumes the parent class has:

    - opt_size: Number of registration pairs
    - dims: Number of spatial dimensions
    - fixed_images: BatchedImages object containing fixed images
    - moving_images: BatchedImages object containing moving images
    - get_warped_coordinates(): Method to get transformed coordinates
    """

    def save_as_ants_transforms(reg, filenames: Union[str, List[str]]):
        """Save deformation fields in ANTs-compatible format.

        Converts the learned deformation fields to displacement fields in physical space
        and saves them in a format that can be used by ANTs registration tools.
        The displacement fields are saved as multi-component images where each component
        represents the displacement along one spatial dimension.

        Args:
            filenames (Union[str, List[str]]): Path(s) where the transform(s) should be saved.
                If a single string is provided for multiple transforms, it will be treated
                as the first filename. For multiple transforms, provide a list of filenames
                matching the number of transforms.

        Raises:
            AssertionError: If number of filenames doesn't match number of transforms (opt_size)

        !!! caution "Physical Space Coordinates"
            The saved transforms are in physical space coordinates, not normalized [-1,1] space.
            The displacement fields are saved with the same orientation and spacing as the 
            fixed images.
        """
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
