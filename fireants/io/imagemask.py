# Copyright (c) 2026 Rohit Jena. All rights reserved.
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



'''
Utility to generate an image mask given an image
'''

import numpy as np
from fireants.io.image import Image
import SimpleITK as sitk

def generate_image_mask_allones(image: Image):
    '''
    given an image, generate a mask that has all ones

    this function allows application of masked loss functions if one image has a mask but the other one
    does not

    TODO: this function might be slow but optimize this later
    '''
    itk_img = image.itk_image
    mask_shape = tuple(list(image.shape)[2:])
    mask_img = np.ones(mask_shape)
    mask_itk = sitk.GetImageFromArray(mask_img)
    mask_itk.CopyInformation(itk_img)
    return Image(mask_itk, device=image.device, dtype=image.array.dtype)

def apply_mask_to_image(image: Image, mask: Image, optimize_memory: bool = True):
    '''
    wrapper function to apply mask - just appends it to the image
    '''
    return image.concatenate(mask, optimize_memory=optimize_memory)

def generate_and_apply_mask_allones(image: Image):
    '''
    returns a new image containing mask
    '''
    mask = generate_image_mask_allones(image)
    image = apply_mask_to_image(image, mask)
    return image

