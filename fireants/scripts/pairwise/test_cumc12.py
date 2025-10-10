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


''' example script to test CUMC dataset '''
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
from fireants.scripts.evaluate_metrics import compute_metrics
import pickle

DATA_DIR = "/data/rohitrango/brain_data/CUMC12/Brains"
LABEL_DIR = "/data/rohitrango/brain_data/CUMC12/Atlases"

# first label is background 
labels_all = np.array([  0,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
        15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
        43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,
        83,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
        97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
       110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
       123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133])

def seg_preprocessor(segmentation: torch.Tensor):
    ''' custom preprocessor for IBSR dataset that maps only the common structures '''
    new_segmentation = torch.zeros_like(segmentation)
    for newidx, label in enumerate(labels_all):
        new_segmentation[segmentation == label] = newidx
    return new_segmentation

# parse
parser = argparse.ArgumentParser("Test IBSR dataset")
parser.add_argument('--algo', type=str, required=True, help='algorithm to use (greedy, syn)')
parser.add_argument('--device', type=str, required=True, help='device to use (cpu, cuda)')

if __name__ == '__main__':

    args = parser.parse_args()
    algo = args.algo
    device = args.device
    if device == 'cpu':
        torch.set_num_threads(32)
        max_samples = 10
    else:
        max_samples = np.inf
    

    loss_type = "fusedcc" if args.device == "cuda" else "cc"
    print("Loss type: {}, device: {}".format(loss_type, device))
    # Get images
    images = sorted(glob(DATA_DIR + "/*img"), key=lambda x: int(x.split("/")[-1].split(".")[0][1:]))
    labels = sorted(glob(LABEL_DIR + "/*img"), key=lambda x: int(x.split("/")[-1].split(".")[0][1:]))
    num_images = len(images)

    # record all times
    all_times = {}
    all_metrics = {}
    global_idx = -1

    # iterate through images
    pbar = tqdm(range(num_images))
    for i in pbar:
        fixed_image_path = images[i]
        fixed_seg_path = labels[i]
        # load batched images
        fixed_image = BatchedImages(Image.load_file(fixed_image_path, device=device))
        fixed_seg   = BatchedImages(Image.load_file(fixed_seg_path, is_segmentation=True, seg_preprocessor=seg_preprocessor, device=device))
        if global_idx >= max_samples:
            break

        for j in range(num_images):
            if j == i:
                continue
            global_idx += 1
            if global_idx >= max_samples:
                break
            # get moving image
            moving_image_path = images[j]
            moving_seg_path = labels[j]
            # load them
            moving_image = BatchedImages(Image.load_file(moving_image_path, device=device))
            moving_seg = BatchedImages(Image.load_file(moving_seg_path, is_segmentation=True, seg_preprocessor=seg_preprocessor, device=device))
            # affine pre-registration
            print("Registering {} to {}".format(fixed_image_path, moving_image_path))
            if device == 'cpu':
                affine_matrix = None
            else:
                affine = AffineRegistration([8, 4, 2, 1], [100, 50, 25, 20], fixed_image, moving_image, \
                    loss_type=loss_type, optimizer='Adam', optimizer_lr=3e-4, optimizer_params={}, cc_kernel_size=5)
                affine.optimize()
                affine_matrix = affine.get_affine_matrix().detach()
            # greedy registration
            if algo == 'greedy':
                deformable = GreedyRegistration(scales=[4, 2, 1], iterations=[200, 100, 25], fixed_images=fixed_image, moving_images=moving_image,
                loss_type=loss_type,
                                    cc_kernel_size=5, deformation_type='compositive', 
                                    smooth_grad_sigma=1,
                                    optimizer='adam', optimizer_lr=0.5, init_affine=affine_matrix)
            elif algo == 'syn':
                deformable = SyNRegistration(scales=[4, 2, 1], iterations=[100, 50, 25], fixed_images=fixed_image, moving_images=moving_image,
                loss_type=loss_type,
                                    cc_kernel_size=5, deformation_type='compositive', optimizer="adam", 
                                    optimizer_lr=0.5,
                                    smooth_grad_sigma=1, init_affine=affine_matrix)
            else:
                raise NotImplementedError
            a = time.time()
            deformable.optimize()
            b = time.time() - a
            print("\nTime taken: {:.2f} seconds\n".format(b))
            all_times[(i, j)] = b
            # evaluate
            if device == 'cpu':
                metrics = {}
            else:
                moved_seg_array = deformable.evaluate(fixed_seg, moving_seg)
                moved_seg_array = (moved_seg_array >= 0.5).float()
                # compute metrics
                metrics = compute_metrics(fixed_seg()[0].detach().cpu().numpy(), moved_seg_array[0].detach().cpu().numpy())
                str = ""
                for k, v in metrics.items():
                    str += f"{k}: {100*np.mean(v):.2f} "
                pbar.set_description(str)
            # append to dictionaries
            all_times[(i, j)] = b
            all_metrics[(i, j)] = metrics
    
    if device == 'cpu':
        with open('cumc12/all_times_{}_cpu.pkl'.format(args.algo), 'wb') as f:
            pickle.dump(all_times, f)
    else:
        with open('cumc12/all_times_{}.pkl'.format(args.algo), 'wb') as f:
            pickle.dump(all_times, f)
        with open('cumc12/all_metrics_{}.pkl'.format(args.algo), 'wb') as f:
            pickle.dump(all_metrics, f)
