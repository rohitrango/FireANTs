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


''' simple utilities for retrieving image transforms '''
import SimpleITK as sitk
import torch
from fireants.utils.globals import PERMITTED_ANTS_MAT_EXT, PERMITTED_ANTS_TXT_EXT
from fireants.utils.util import any_extension

def get_affine_transform_from_file(filename: str, dim: int):
    MAT_TXT_FILES = PERMITTED_ANTS_MAT_EXT + PERMITTED_ANTS_TXT_EXT
    if not any_extension(filename, MAT_TXT_FILES):
        raise ValueError(f"File {filename} is not an ANTS transform file")
    # read affine parameters
    transform = sitk.ReadTransform(filename)
    raise NotImplementedError
    return transform