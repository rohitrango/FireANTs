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