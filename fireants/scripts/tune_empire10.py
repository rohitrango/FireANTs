''' Script to tune EMPIRE 10 dataset (contains only 30 pairs, so should be fast) '''
from glob import glob
import time
import numpy as np
import torch
import SimpleITK as sitk
from fireants.io.image import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
from fireants.utils.imageutils import jacobian
from evaluate_metrics import compute_metrics
from itertools import product
from functools import partial
import pandas as pd
import argparse
from os import path as osp
from torch import nn
import os
import torch.nn.functional as F
from contextlib import nullcontext

import ray
from ray import tune, air
from ray.air import session
from ray.tune.search import ConcurrencyLimiter, BasicVariantGenerator

SCANS_PATH = "/data/rohitrango/EMPIRE10/scans_v2/"
MASKS_PATH = "/data/rohitrango/EMPIRE10/lungMasks_v2/"
OUT_PATH = "/data/rohitrango/EMPIRE10/submission_v2/"
# SCANS_PATH = "/data/rohitrango/EMPIRE10/scans/"
# MASKS_PATH = "/data/rohitrango/EMPIRE10/lungMasks/"
# OUT_PATH = "/data/rohitrango/EMPIRE10/submission/"

class DiceLossModule(nn.Module):
    # Dice loss in the form of nn Module to pass into affine step 
    # assumes a single class only (for brevity)
    def __init__(self):
        super().__init__()
    
    def forward(self, moved_seg, fixed_seg, *args, **kwargs):
        # flatten the images into [B, N] and the compute dice score
        moved_flat = moved_seg.flatten(1)
        fixed_flat = fixed_seg.flatten(1)
        num = 2*(moved_flat*fixed_flat).sum(1)
        den = moved_flat.sum(1) + fixed_flat.sum(1)
        return 1 - (num/den).mean()

def get_image_paths(lung_id, masks=False):
    PATH = SCANS_PATH if not masks else MASKS_PATH
    fixed = osp.join(PATH, "{:02d}_Fixed.mhd".format(lung_id))
    moving = osp.join(PATH, "{:02d}_Moving.mhd".format(lung_id))
    return fixed, moving

def augment_image_with_mask(image, mask):
    ''' normalize the image from [0, 1] and then multiply with mask '''
    m, M = image.array.min(), image.array.max()
    image.array = (image.array - m)/(M - m)
    image.array *= mask.array
    return image

def register_dataset(config):
    ''' given a config, run registration on all pairs '''
    algo = config['algo']
    grad_sigma = config['grad_sigma']
    warp_sigma = config['warp_sigma']
    lr = config['lr']
    mode = config['mode']
    dice_loss_mod = DiceLossModule()
    cc_size = config['cc_size']

    # make the directory
    if mode == 'eval':
        out_root_dir = osp.join(OUT_PATH, algo + "_lungs")
        os.makedirs(out_root_dir, exist_ok=True)

    # tracker and count
    metric_tracker = {
        'dice': 0,
        'CC': 0,
        'badjacdet': 0,
    }
    count = 0

    for lung_id in range(1, 31):
        torch.cuda.empty_cache()
        time.sleep(2)
        fixed_img_path, moving_img_path = get_image_paths(lung_id)
        fixed_seg_path, moving_seg_path = get_image_paths(lung_id, masks=True)
        # load fixed images
        fixed_image, fixed_seg = Image.load_file(fixed_img_path), Image.load_file(fixed_seg_path, is_segmentation=True)
        fixed_image = augment_image_with_mask(fixed_image, fixed_seg)

        fixed_image, fixed_seg = BatchedImages(fixed_image), BatchedImages(fixed_seg)
        # same for moving images
        moving_image, moving_seg = Image.load_file(moving_img_path), Image.load_file(moving_seg_path, is_segmentation=True)
        moving_image = augment_image_with_mask(moving_image, moving_seg)
        moving_image, moving_seg = BatchedImages(moving_image), BatchedImages(moving_seg)
        # perform affine registration between masks, using dice score
        affine = AffineRegistration([6, 4, 2, 1], [200, 100, 50, 20], fixed_seg, moving_seg, \
                   loss_type='custom', custom_loss=dice_loss_mod, optimizer='Adam', optimizer_lr=3e-3, optimizer_params={}, cc_kernel_size=5)
        aff_start = time.time()
        affine.optimize(save_transformed=False)
        aff_end = time.time()
        affine_matrix = affine.get_affine_matrix().detach()
        del affine

        if algo == 'greedy':
            deformable = GreedyRegistration(scales=[6, 4, 2, 1], iterations=[200, 150, 75, 25], fixed_images=fixed_image, moving_images=moving_image,
                                    cc_kernel_size=cc_size, 
                                    # deformation_type='compositive', 
                                    deformation_type=config['deformation_type'],
                                    max_tolerance_iters=1000,
                                    smooth_grad_sigma=grad_sigma, smooth_warp_sigma=warp_sigma, 
                                    # optimizer_params={'beta1': config['beta1'], 'beta2': config['beta2']}, 
                                    optimizer='Adam',
                                    optimizer_lr=lr, init_affine=affine_matrix)
        elif algo == 'syn':
            deformable = SyNRegistration(scales=[6, 4, 2, 1], iterations=[200, 150, 75, 25], fixed_images=fixed_image, moving_images=moving_image,
                                    cc_kernel_size=cc_size, 
                                    # deformation_type='compositive', 
                                    deformation_type=config['deformation_type'],
                                    optimizer="Adam", optimizer_lr=lr,
                                    max_tolerance_iters=1000,
                                    optimizer_params={'beta1': config['beta1'], 'beta2': config['beta2']},
                                    smooth_grad_sigma=grad_sigma, smooth_warp_sigma=warp_sigma, init_affine=affine_matrix)
        def_start = time.time()
        deformable.optimize(save_transformed=False)
        def_end = time.time()

        print(f"For scan pair: {lung_id}")
        print(f"Affine time: {aff_end - aff_start}")
        print(f"Deformable time: {def_end - def_start}\n")
        print(f"Total time: {def_end + aff_end - def_start - aff_start}\n")

        # evaluate dice score, cross correlation, and rate of diffeomorphism
        grad_context = torch.no_grad
        with grad_context():
            # moved coordinates
            moved_coordinates = deformable.get_warped_coordinates(fixed_image, moving_image)  # [B, H, W, D, 3]
            loss_fn = deformable.loss_fn
            del deformable
            moved_seg_array = F.grid_sample(moving_seg(), moved_coordinates, mode='bilinear', align_corners=True)
            moved_seg_array = (moved_seg_array >= 0.5).float()
            dice_score = 1 - dice_loss_mod(moved_seg_array, fixed_seg())
            count += 1
            if mode == 'tune':
                # eval cross correlation
                moved_image_array = F.grid_sample(moving_image(), moved_coordinates, mode='bilinear', align_corners=True)
                # cc = -deformable.loss_fn(moved_image_array, fixed_image())
                cc = -loss_fn(moved_image_array, fixed_image())
                # jacobian determinant
                jac_v = jacobian(moved_coordinates, normalize=True).permute(0, 2, 3, 4, 1, 5)[:, 1:-1, 1:-1, 1:-1]
                badjacdet = (torch.linalg.det(jac_v) < 0).float().mean()
                # report metrics
                metric_tracker['dice'] += dice_score.item()
                metric_tracker['CC'] += cc.item() / np.prod(fixed_image().shape)
                metric_tracker['badjacdet'] += badjacdet.item()
                # average metric
                avg_metric = {k: v/count for k, v in metric_tracker.items()}
                # print(avg_metric)
                tune.report(**avg_metric)
            else:
                print(dice_score.item())
                # write to file
                #moved_lung = deformable.evaluate(fixed_image, moving_image)
                moved_lung = F.grid_sample(moving_image(), moved_coordinates, mode='bilinear', align_corners=True)
                moved_lung = moved_lung.detach().cpu().numpy().astype(np.float32)[0, 0]
                moved_lung = sitk.GetImageFromArray(moved_lung)
                moved_lung.CopyInformation(fixed_image.images[0].itk_image)
                sitk.WriteImage(moved_lung, osp.join(out_root_dir, "{:02d}.nii.gz".format(lung_id)))
                print("Written to {}".format(osp.join(out_root_dir, "{:02d}.nii.gz".format(lung_id))))

            ## we dont need the segmentations
            with grad_context():
                del fixed_seg, moving_seg
                warp = moved_coordinates[0]
                del fixed_image
                torch2phy = moving_image.get_torch2phy()  # [1, 4, 4]
                # warped coordinates in physical space
                warp = torch.einsum('zyxd,bd->zyxb', warp, torch2phy[0, :3, :3])  # [Z, Y, X, 3]
                warp = warp + torch2phy[0, :3, 3][None, None, None, :]  # [Z, Y, X, 3]
                # subtract grid
                px2phy = moving_image.images[0]._px2phy  # [4, 4]
                origin = (px2phy @ np.array([[50, 50, 50, 1]]).T)  # [3]   # this is the coordinate of the origin in physical coordinates
                print(origin)
                origin = origin[:3, 0]
                warp = warp - torch.tensor(origin, device=warp.device).float()[None, None, None, :]  # [Z, Y, X, 3]
                warp = warp[50:-50, 50:-50, 50:-50]  # [Z, Y, X, 3]
                # now subtract fixed coordinates
                del moving_image
                fixed_img_path2 = fixed_img_path.replace("_v2", "")
                fixed_image = Image.load_file(fixed_img_path2)
                torch2phy = fixed_image.torch2phy  # [1, 4, 4]
                init_grid = F.affine_grid(torch2phy[:, :3], fixed_image.array.shape, align_corners=True)[0]  # [Z, Y, X, 3]
                # del fixed_image
                print(init_grid.shape, warp.shape)
                warp_phy = (warp - init_grid).detach().cpu().numpy().astype(np.float32)
                # save deformation
                defX = sitk.GetImageFromArray(warp_phy[..., 0])
                defY = sitk.GetImageFromArray(warp_phy[..., 1])
                defZ = sitk.GetImageFromArray(warp_phy[..., 2])
                defX.CopyInformation(fixed_image.itk_image)
                defY.CopyInformation(fixed_image.itk_image)
                defZ.CopyInformation(fixed_image.itk_image)
                lung_dir = osp.join(out_root_dir, "{:02d}".format(lung_id))
                os.makedirs(lung_dir, exist_ok=True)
                sitk.WriteImage(defX, osp.join(lung_dir, "defX.mhd"))
                sitk.WriteImage(defY, osp.join(lung_dir, "defY.mhd"))
                sitk.WriteImage(defZ, osp.join(lung_dir, "defZ.mhd"))
                print("Written to {}".format(lung_dir))
                del fixed_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, help="greedy or syn", choices=['greedy', 'syn'])
    parser.add_argument("--mode", type=str, required=True, help="tune/eval", choices=['tune', 'eval'])
    parser.add_argument("--grad_sigma", type=float, required=False, help="grad sigma", default=6)
    parser.add_argument("--warp_sigma", type=float, required=False, help="warp sigma", default=0.4)
    parser.add_argument("--use_v1", action='store_true', help="use v1")
    parser.add_argument('--lr', type=float, default=0.25)
    parser.add_argument('--cc_size', type=int, default=5)
    parser.add_argument('--deformation_type', type=str, default='compositive')
    args = parser.parse_args()
    mode = args.mode
    config = {
        'algo': args.algo,
        'mode': mode,
        'beta1': 0.9,
        'beta2': 0.99,
        'cc_size': args.cc_size,
        'deformation_type': args.deformation_type,
        'lr': args.lr,
        'grad_sigma': tune.grid_search(np.arange(1, 10.5, 0.5)) if mode == 'tune' else args.grad_sigma,
        'warp_sigma': tune.grid_search(np.arange(0.2, 5, 0.2)) if mode == 'tune' else args.warp_sigma,
    }
    # register_dataset(config)
    if mode == 'tune':
        ray.init()
        tuner = tune.Tuner(
            tune.with_resources(register_dataset, resources={'cpu': 2, 'gpu': 1}),
            param_space=config,
            tune_config=tune.TuneConfig(num_samples=1),
            run_config=air.RunConfig(name="empire10_{}_{}".format(args.algo, args.deformation_type)),
        )
        print("Running...")
        results = tuner.fit()
        ray.shutdown()
    # create evaluation lung and deformation fields
    elif mode == 'eval':
        register_dataset(config)
    else:
        raise ValueError("Invalid mode")
