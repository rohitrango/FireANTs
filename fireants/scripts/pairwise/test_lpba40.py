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


''' example script to test LPBA dataset '''
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

DATA_DIR = "/data/rohitrango/brain_data/LPBA40/registered_label_pairs/"
LABEL_DIR = "/data/rohitrango/brain_data/LPBA40/registered_label_pairs/"

# first label is background 
labels_all = np.array([  0,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
        33,  34,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  61,
        62,  63,  64,  65,  66,  67,  68,  81,  82,  83,  84,  85,  86,
        87,  88,  89,  90,  91,  92, 101, 102, 121, 122, 161, 162, 163,
       164, 165, 166, 181, 182])

def seg_preprocessor(segmentation: torch.Tensor):
    ''' custom preprocessor for IBSR dataset that maps only the common structures '''
    new_segmentation = torch.zeros_like(segmentation)
    for newidx, label in enumerate(labels_all):
        new_segmentation[segmentation == label] = newidx
    return new_segmentation

parser = argparse.ArgumentParser("Test LPBA40 dataset")
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
    print("Using {} algorithm".format(args.algo))

    image_ids = range(1, 41)
    # record all times
    all_times = dict()
    all_metrics = dict()
    global_idx = -1

    # iterate through images
    for i in image_ids:
        fixed_image_path = DATA_DIR + "l{}_to_l{}.img".format(i, i)
        fixed_seg_path = LABEL_DIR + "l{}_to_l{}.img".format(i, i)
        # load batched images
        fixed_image = BatchedImages(Image.load_file(fixed_image_path, device=device))
        fixed_seg   = BatchedImages(Image.load_file(fixed_seg_path, is_segmentation=True, seg_preprocessor=seg_preprocessor, device=device))
        if global_idx >= max_samples:
            break

        for j in image_ids:
            if j == i:
                continue
            global_idx += 1
            if global_idx >= max_samples:
                break
            # get moving image
            moving_image_path = DATA_DIR + "l{}_to_l{}.img".format(j, i)
            moving_seg_path = LABEL_DIR + "l{}_to_l{}.img".format(j, i)
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
                deformable = GreedyRegistration(scales=[4, 2, 1], iterations=[200, 100, 50], fixed_images=fixed_image, moving_images=moving_image,
                                        loss_type=loss_type,
                                        cc_kernel_size=7, deformation_type='compositive', 
                                        smooth_grad_sigma=2,
                                        smooth_warp_sigma=0.5,
                                        optimizer='adam', optimizer_lr=0.5, init_affine=affine_matrix)
            elif algo == 'syn':
                deformable = SyNRegistration(scales=[4, 2, 1], iterations=[100, 75, 50], fixed_images=fixed_image, moving_images=moving_image,
                                        loss_type=loss_type,
                                        cc_kernel_size=5, deformation_type='compositive', optimizer="adam", optimizer_lr=0.5,
                                        smooth_grad_sigma=1, init_affine=affine_matrix)
            a = time.time()
            deformable.optimize()
            b = time.time() - a
            print("\nTime taken: {:.2f} seconds\n".format(b))
            # evaluate
            if device == 'cpu':
                metrics = {}
            else:
                moved_seg_array = deformable.evaluate(fixed_seg, moving_seg)
                moved_seg_array = (moved_seg_array >= 0.5).float()
                # print(fixed_seg()[0].shape, moved_seg_array[0].shape)
                metrics = compute_metrics(fixed_seg()[0].detach().cpu().numpy(), moved_seg_array[0].detach().cpu().numpy())
                str = ""
                for k, v in metrics.items():
                    str += f"{k}: {100*np.mean(v):.2f} "
                print(str)
            # add them to list
            all_times[(i, j)] = b
            all_metrics[(i, j)] = metrics

    for key in metrics.keys():
        met = [x[key] for x in all_metrics.values()]
        print(key, np.mean(met), np.std(met))

    if device == 'cpu':
        with open('lpba40/all_times_{}_cpu.pkl'.format(args.algo), 'wb') as f:
            pickle.dump(all_times, f)
    else:
        with open('lpba40/all_times_{}.pkl'.format(args.algo), 'wb') as f:
            pickle.dump(all_times, f)
        with open('lpba40/all_metrics_{}.pkl'.format(args.algo), 'wb') as f:
            pickle.dump(all_metrics, f)

    # # Save results
    # all_dice_scores = np.array(all_dice_scores)
    # print("Mean dice score (all): {}".format(all_dice_scores.mean()))
    # print("Std dice score (all): {}".format(all_dice_scores.std()))
    # print("Mean dice score: {}".format(all_dice_scores.mean(0)))
    # print("Std dice score: {}".format(all_dice_scores.std(0)))
    # print("Mean runtime: {}".format(np.array(all_times).mean()))
    # print("Std runtime: {}".format(np.array(all_times).std()))
    
    # # np.save('all_times.npy', np.array(all_times))
    # # np.save("all_dice_scores.npy", np.array(all_dice_scores))
    # # print("saved results.")
    # np.save('cumc12/all_times_syn.npy', np.array(all_times))
    # np.save("cumc12/all_dice_scores_syn.npy", np.array(all_dice_scores))
    # print("saved results.")

            
