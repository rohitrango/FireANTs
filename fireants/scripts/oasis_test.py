from glob import glob
import time
import numpy as np
import torch
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
from fireants.io.image import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
import argparse
from tqdm import tqdm
from evaluate_metrics import compute_metrics
# import pickle
# import itertools
from torch.nn import functional as F

def dice_score(p, q):
    ''' computes the dice score between two tensors '''
    return 2.0 * (p * q).mean() / (p.mean() + q.mean())

def main():
    ''' test couple of images '''
    # fixed_id, moving_id = "OASIS_0011_0000", "OASIS_0301_0000"
    # fixed_id, moving_id = "OASIS_0002_0000", "OASIS_0339_0000"
    fixed_id, moving_id = "OASIS_0001_0000", "OASIS_0301_0000"

    fixed_image, moving_image = Image.load_file("/data/rohitrango/OASIS/imagesTr/{}.nii.gz".format(fixed_id)), \
        Image.load_file("/data/rohitrango/OASIS/imagesTr/{}.nii.gz".format(moving_id))
    fixed_image, moving_image = BatchedImages(fixed_image), BatchedImages(moving_image)
    # deformable = GreedyRegistration([4, 2, 1], [100, 100, 20],
    # deformable = SyNRegistration([4, 2, 1], [100, 100, 20],
    deformable = GreedyRegistration([4, 2, 1], [100, 100, 20],
                                fixed_image, moving_image, deformation_type='compositive',
                                optimizer='adam', optimizer_lr=0.5, cc_kernel_size=7,
                                smooth_grad_sigma=1,
                                smooth_warp_sigma=0.5)
    deformable.optimize(save_transformed=False)
    warp = deformable.get_warped_coordinates(fixed_image, moving_image)
    moved_image = deformable.evaluate(fixed_image, moving_image)
    ncc = -deformable.loss_fn(fixed_image(), moved_image)
    print(ncc.item()/fixed_image().numel())
    # plot dice score
    dice_scores = []
    fixed_seg, moving_seg = Image.load_file("/data/rohitrango/OASIS/labelsTr/{}.nii.gz".format(fixed_id)),\
          Image.load_file("/data/rohitrango/OASIS/labelsTr/{}.nii.gz".format(moving_id))
    fixed_data, moving_data = fixed_seg.array, moving_seg.array

    # compute metrics
    common_labels = set(torch.unique(fixed_data).tolist()).union(set(torch.unique(moving_data).tolist()))
    dice_img = []
    for lab in common_labels:
        if lab == 0:
            continue
        moving_seg_label = (moving_data == lab).float()
        fixed_seg_label = (fixed_data == lab).float()
        moved_seg_label = (F.grid_sample(moving_seg_label, warp, mode='bilinear', align_corners=True) >= 0.5).float()
        # dice
        dice = dice_score(moved_seg_label, fixed_seg_label).item()
        dice_img.append(dice)
    print(dice_img)
    print(np.mean(dice_img))

if __name__ == '__main__':
    main()