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


''' example script to test IBSR dataset '''
from glob import glob
import time
import numpy as np
import torch
from fireants.io.image import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
import argparse
from tqdm import tqdm
from fireants.scripts.evaluate_metrics import compute_metrics
import pickle

DATA_DIR = "/data/rohitrango/brain_data/IBSR18/"
MAX_LABEL = 72
labels_all = np.array([ 0,  2,  3,  4,  5,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26,
        28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60])

def seg_preprocessor(segmentation: torch.Tensor):
    ''' custom preprocessor for IBSR dataset that maps only the common structures '''
    new_segmentation = torch.zeros_like(segmentation)
    for newidx, label in enumerate(labels_all):
        new_segmentation[segmentation == label] = newidx
    return new_segmentation

# parser
parser = argparse.ArgumentParser("Test IBSR dataset")
parser.add_argument('--algo', type=str, required=True, help='algorithm to use (greedy, syn)')
parser.add_argument('--device', type=str, required=True, help='device to use (cpu, cuda)')

if __name__ == '__main__':
    args = parser.parse_args()
    print("Using {} algorithm".format(args.algo))

    # compile all the paths
    dirs = sorted(glob(DATA_DIR + "IBSR_*"))
    images = [glob(x + "/*ana_strip.nii.gz") for x in dirs]
    labels = [glob(x + "/*seg_ana.nii.gz") for x in dirs]
    assert all([len(x) == 1 for x in images])
    assert all([len(x) == 1 for x in labels])
    images, labels = [x[0] for x in images], [x[0] for x in labels]
    num_images = len(images)

    # record all times
    all_times = {}
    all_metrics = {}

    device = args.device
    if device == 'cpu':
        torch.set_num_threads(32)
        max_samples = 10
    else:
        max_samples = np.inf
    loss_type = "fusedcc" if args.device == "cuda" else "cc"
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
                affine = AffineRegistration([8, 4, 2, 1], [200, 100, 50, 20], fixed_image, moving_image, \
                    loss_type=loss_type, optimizer='Adam', optimizer_lr=3e-4, optimizer_params={}, cc_kernel_size=5)
                affine.optimize()
                affine_matrix = affine.get_affine_matrix().detach()
            if args.algo == 'greedy':
            # greedy registration
                deformable = GreedyRegistration(scales=[4, 2, 1], iterations=[200, 100, 25], fixed_images=fixed_image, moving_images=moving_image,
                                    cc_kernel_size=5, deformation_type='compositive', 
                                    smooth_grad_sigma=1, 
                                    optimizer='adam', optimizer_lr=0.5, init_affine=affine_matrix)
            elif args.algo == 'syn':
                deformable = SyNRegistration(scales=[4, 2, 1], iterations=[100, 50, 50], fixed_images=fixed_image, moving_images=moving_image,
                                        cc_kernel_size=5, deformation_type='compositive', optimizer="adam", optimizer_lr=0.5,
                                        optimizer_params={
                                            'beta1': 0.5, 
                                            'beta2': 0.75,},
                                        smooth_grad_sigma=1, smooth_warp_sigma=0.5, init_affine=affine_matrix,
                                        )
            else:
                raise NotImplementedError
            # record time
            a = time.time()
            deformable.optimize()
            b = time.time() - a
            print("\nTime taken: {:.2f} seconds\n".format(b))
            # evaluate
            if device == 'cpu':
                metrics = {}
            else:
                moved_seg_array = (deformable.evaluate(fixed_seg, moving_seg) >= 0.5).float()
                fixed_seg_array = (fixed_seg() >= 0.5).float()

                # compute metrics
                metrics = compute_metrics(fixed_seg_array[0].detach().cpu().numpy(), moved_seg_array[0].detach().cpu().numpy())
                str = ""
                for k, v in metrics.items():
                    str += f"{k}: {100*np.mean(v):.2f} "
                pbar.set_description(str)
            # append to dictionaries
            all_times[(i, j)] = b
            all_metrics[(i, j)] = metrics
    
    # Save results
    if device == 'cpu':
        with open('ibsr/all_times_{}_cpu.pkl'.format(args.algo), 'wb') as f:
            pickle.dump(all_times, f)
    else:
        with open('ibsr/all_times_{}.pkl'.format(args.algo), 'wb') as f:
            pickle.dump(all_times, f)
        with open('ibsr/all_metrics_{}.pkl'.format(args.algo), 'wb') as f:
            pickle.dump(all_metrics, f)
