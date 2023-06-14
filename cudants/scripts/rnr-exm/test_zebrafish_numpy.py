'''
Run registration on mouse brain data
'''
from glob import glob
import time
import numpy as np
import torch
from torch import nn
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
from cudants.io.image import Image, BatchedImages
from cudants.registration.affine import AffineRegistration
from cudants.registration.rigid import RigidRegistration
from cudants.registration.greedy import GreedyRegistration
from cudants.registration.syn import SyNRegistration
import argparse
from tqdm import tqdm
import os
from os import path as osp
import h5py
from time import sleep
import shutil
from torch.nn import functional as F
import ray
from ray import tune, air
from tqdm import tqdm

import numpy as np
from scipy.ndimage import map_coordinates

# global parameters
ROOT_DIR = "/data/rohitrango/RnR-ExM/test"

if __name__ == '__main__':
    # arguments
    DEFORMATION_FILE = "rnr-exm/challenge_eval/submission/zebrafish/submission/zebrafish_test.h5"
    deformations = h5py.File(DEFORMATION_FILE, "r")
    for pair in [5, 6, 7]:
        df = deformations[f'pair{pair}'][:]
        print(df.shape, df.dtype)
        # test file
        image_file = h5py.File(osp.join(ROOT_DIR, f"zebrafish_pair{pair}.h5"), "r")
        moving = image_file['move'][:]
        image_file.close()
        #
        D, H, W, C = df.shape
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        warped_image = map_coordinates(moving, identity + df.transpose(3,0,1,2), order=1)
        # save to moved image
        moved_image_file = osp.join(ROOT_DIR, f"zebrafish_pair{pair}_warp.h5")
        image_file = h5py.File(moved_image_file, "w")
        image_file.create_dataset("warped", data=warped_image)
        image_file.close()
        print(f"saved to {moved_image_file}.")

