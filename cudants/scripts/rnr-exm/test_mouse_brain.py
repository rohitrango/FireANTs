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
from cudants.registration.greedy import GreedyRegistration
from cudants.registration.syn import SyNRegistration
import argparse
from tqdm import tqdm
import os
from os import path as osp
import h5py
from time import sleep
import shutil
import torch.nn.functional as F

import ray
from ray import tune, air

# global parameters
SPACING = [0.1625, 0.1625, 0.4]
ROOT_DIR = "/data/rohitrango/RnR-ExM"

def dice_score(moved_seg, fixed_seg):
    # get dice score b/w moved and fixed segmentations
    moved_flat = moved_seg.flatten(1)
    fixed_flat = fixed_seg.flatten(1)
    num = 2*(moved_flat*fixed_flat).sum(1)
    den = moved_flat.sum(1) + fixed_flat.sum(1)
    return (num/den).mean()

def memory_usage(tensor):
    # track memory usage
    dtype_size = tensor.element_size()
    num_elements = tensor.numel()
    overhead = max(tensor.storage().size() - num_elements * dtype_size, 0)
    total_memory_bytes = num_elements * dtype_size + overhead
    total_memory_mb = total_memory_bytes / 1024.0 / 1024.0
    return total_memory_mb

def normalize_intensity(image):
    m, M = image.min(), image.max()
    return (image - m)/(M - m) 

def load_mouse_brain(path, percentile=99.0, is_segmentation=False):
    ''' given path to h5py file, load the fixed and moving images '''
    ### To prevent huge load times, I've saved the images directly as nifti files
    image = Image.load_file(path, is_segmentation=is_segmentation)
    # process it normally like an image if not a segmentation
    if not is_segmentation:
        # if percentile < 100:
        #     per = np.percentile(image.array.cpu().numpy(), percentile)
        #     # per = torch.quantile(image.array.reshape(-1), percentile/100)   # input tensor too large
        #     image.array[image.array > per] = per
        minper, maxper = np.percentile(image.array.cpu().numpy(), [5, 95])
        image.array = torch.clamp(image.array, minper, maxper)
        image.array = normalize_intensity(image.array)
        image.array = image.array ** 2  # gamma correction
    # batchify it
    image = BatchedImages(image)
    return image

def run_registration_val(config):
    ''' given a config, run registation with given config  '''
    split = config['split']
    mode = config['mode']
    if mode == 'tune':
        torch.cuda.empty_cache()
        sleep(10)
        # tune.utils.wait_for_gpu(target_util=0.1)
    # load path dir
    path_dir = osp.join(ROOT_DIR, f"{split}")
    # only this one mouse brain to use
    # mouse_brains = sorted(glob(osp.join(path_dir, "mouse_pair4.h5")))
    fixed_brain, moving_brain = osp.join(path_dir, "mouse_pair4_fixed.nii.gz"), osp.join(path_dir, "mouse_pair4_moving.nii.gz")
    fixed_brain, moving_brain = load_mouse_brain(fixed_brain), load_mouse_brain(moving_brain)
    # load segmentations
    fixed_seg, moving_seg = osp.join(path_dir, "mouse_pair4_segmentation_fixed.nii.gz"), osp.join(path_dir, "mouse_pair4_segmentation_moving.nii.gz")
    fixed_seg, moving_seg = load_mouse_brain(fixed_seg, is_segmentation=True), load_mouse_brain(moving_seg, is_segmentation=True)

    # affine registration
    # affine = AffineRegistration([8, 4, 2], [0, 0, 0], fixed_brain, moving_brain, \
    # affine = AffineRegistration([8, 4, 2], [100, 50, 20], fixed_brain, moving_brain, \
    cc_size = 9
    affine = AffineRegistration([16, 8, 4], [1000, 500, 200], fixed_brain, moving_brain, \
            cc_kernel_size=cc_size,
            loss_type='mi', optimizer='Adam', optimizer_lr=5e-4, optimizer_params={}) #, cc_kernel_size=5)
    aff_start = time.time()
    affine.optimize(save_transformed=False)
    aff_end = time.time()
    affine_matrix = affine.get_affine_matrix().detach()
    print(affine_matrix)
    del affine
    torch.cuda.empty_cache()
    print("Clearing cache...")
    # deformable registration
    deformable = GreedyRegistration(scales=[4, 2], iterations=[150, 100], fixed_images=fixed_brain, moving_images=moving_brain,
                                    loss_type='cc', 
                                    cc_kernel_size=cc_size, deformation_type='compositive', 
                                    smooth_grad_sigma=config['grad_sigma'], smooth_warp_sigma=config['warp_sigma'],
                                    optimizer='adam', optimizer_lr=config['lr'], init_affine=affine_matrix)
    greedy_start = time.time()
    deformable.optimize(save_transformed=False)
    greedy_end = time.time()
    print("Affine time: ", aff_end - aff_start) 
    print("Greedy time: ", greedy_end - greedy_start)
    print("Total time: ", greedy_end - greedy_start + aff_end - aff_start)

    # now evaluate on the actual segmenation
    fixed_size = fixed_seg().shape[2:]
    with torch.no_grad():
        deformable.warp.set_size(fixed_size)
        moved_seg = deformable.evaluate(fixed_seg, moving_seg)
        # calculate
        dice = dice_score(fixed_seg(), moved_seg).item()
        del moved_seg
        moved_brain = deformable.evaluate(fixed_brain, moving_brain)
        # cc = -deformable.loss_fn(fixed_brain(), moved_brain).item() / np.prod(list(moved_brain.shape))
        cc = 0
        # save
        if mode == 'tune':
            # tune.report(dice=dice, cc=cc)
            tune.report(**{'dice': dice, 'cc': cc})
        else:
            # save this
            moved_brain = moved_brain.cpu().numpy()[0, 0]  # [H, W, D]
            moved_brain = (moved_brain * 5000).astype(np.uint16)
            moved_brain = sitk.GetImageFromArray(moved_brain)
            moved_brain.CopyInformation(fixed_brain.images[0].itk_image)
            sitk.WriteImage(moved_brain, osp.join(path_dir, "mouse_pair4_moved_brain.nii.gz"))
            print("Dice score: ", dice)
            print("cc score: ", cc)
        
    if mode == 'tune':
        pass
    else:
        with torch.no_grad():
            warp = deformable.get_warped_coordinates(fixed_brain, moving_brain).detach()  # [1, H, W, D, 3]
            warp = (warp - deformable.warp.grid)[0]  # [H, W, D, 3]
            Z, Y, X = warp.shape[:3]
            warp[..., 0] *= (X - 1)/2
            warp[..., 1] *= (Y - 1)/2
            warp[..., 2] *= (Z - 1)/2
            warp = warp.cpu().numpy()
            warp = warp[..., ::-1]
            print("Final warp shape: ", warp.shape)
        # convert these coordinates into the physical space?

    del deformable
    torch.cuda.empty_cache()
    print("Clearing cache...")
    # return depending on what is needed
    if mode == 'tune':
        tune.report(**{'dice': dice, 'cc': cc})
        return {'dice': dice, 'cc': cc}
    else:
        return warp


def run_registration_test(config):
    ''' given a config, run registation on the test images '''
    # load path dir
    path_dir = osp.join(ROOT_DIR, "test")
    # only this one mouse brain to use
    output_file = h5py.File("rnr-exm/challenge_eval/submission/mouse_test.h5", "w")
    cc_size = 15
    for brain_id in [5, 6, 7]:
        # get brains
        fixed_brain, moving_brain = osp.join(path_dir, f"mouse_pair{brain_id}_fixed.nii.gz"), osp.join(path_dir, f"mouse_pair{brain_id}_moving.nii.gz")
        fixed_brain, moving_brain = load_mouse_brain(fixed_brain), load_mouse_brain(moving_brain)
        # affine registration
        affine = AffineRegistration([16, 8, 4], [1000, 500, 200], fixed_brain, moving_brain, \
                cc_kernel_size=cc_size,
                loss_type='mi', optimizer='Adam', optimizer_lr=5e-4, optimizer_params={}) #, cc_kernel_size=5)
        aff_start = time.time()
        affine.optimize(save_transformed=False)
        aff_end = time.time()
        affine_matrix = affine.get_affine_matrix().detach()
        del affine
        torch.cuda.empty_cache()
        print("Clearing cache...")
        # deformable registration
        # deformable = GreedyRegistration(scales=[4, 2, 1.5], iterations=[100, 50, 50], fixed_images=fixed_brain, moving_images=moving_brain,
        # deformable = GreedyRegistration(scales=[8, 4, 2], iterations=[100, 50, 50], fixed_images=fixed_brain, moving_images=moving_brain,
        # deformable = GreedyRegistration(scales=[16, 8, 4], iterations=[300, 150, 100], fixed_images=fixed_brain, moving_images=moving_brain,
        deformable = GreedyRegistration(scales=[4, 2], iterations=[150, 100], fixed_images=fixed_brain, moving_images=moving_brain,
                                        cc_kernel_size=cc_size, deformation_type='compositive', 
                                        smooth_grad_sigma=config['grad_sigma'], smooth_warp_sigma=config['warp_sigma'],
                                        optimizer='adam', optimizer_lr=config['lr'], init_affine=affine_matrix)
        greedy_start = time.time()
        deformable.optimize(save_transformed=False)
        greedy_end = time.time()
        print("Affine time: ", aff_end - aff_start) 
        print("Greedy time: ", greedy_end - greedy_start)
        print("Total time: ", greedy_end - greedy_start + aff_end - aff_start)
        # now evaluate on the actual segmenation
        fixed_size = fixed_brain().shape[2:]
        with torch.no_grad():
            # deformable.warp.set_size(fixed_size)
            # save this
            moved_brain = deformable.evaluate(fixed_brain, moving_brain).cpu().numpy()[0, 0]  # [H, W, D]
            moved_brain = (moved_brain * 5000).astype(np.uint16)
            moved_brain = sitk.GetImageFromArray(moved_brain)
            moved_brain.CopyInformation(fixed_brain.images[0].itk_image)
            sitk.WriteImage(moved_brain, osp.join(path_dir, f"mouse_pair{brain_id}_moved_brain.nii.gz"))
            # save warp
            warp = deformable.get_warped_coordinates(fixed_brain, moving_brain).detach()  # [1, H, W, D, 3]
            grid = deformable.warp.grid
            del deformable
            grid = F.interpolate(grid.permute(0, 4, 1, 2, 3), size=fixed_size, mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)
            warp = (warp - grid)[0]  # [H, W, D, 3]
            Z, Y, X = warp.shape[:3]
            warp[..., 0] *= (X - 1)/2
            warp[..., 1] *= (Y - 1)/2
            warp[..., 2] *= (Z - 1)/2
            warp = warp.cpu().numpy()
            warp = warp[..., ::-1]
            warp = np.around(np.float32(warp), decimals=2)
            print("Final warp shape: ", warp.shape)
            # save this
        output_file.create_dataset(f"pair{brain_id}", data=warp, compression='gzip')
        # delete cache and prepare for next one
        # del deformable
        torch.cuda.empty_cache()
        print("Clearing cache...")
        sleep(10)
    # close file
    output_file.close()


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='mode to use (tune/val/test)')
    # parser.add_argument('--grad_sigma', type=float, default=6.8, help='sigma for gradient smoothing (used in tune mode)')
    # parser.add_argument('--warp_sigma', type=float, default=0.2, help='sigma for warping smoothing (used in tune mode)')
    parser.add_argument('--grad_sigma', type=float, default=6, help='sigma for gradient smoothing (used in tune mode)')
    parser.add_argument('--warp_sigma', type=float, default=0.25, help='sigma for warping smoothing (used in tune mode)')
    parser.add_argument('--exp_name', type=str, default='rnr_exm_mousebrain', help='experiment name')
    parser.add_argument('--lr', type=float, default=0.25, help='learning rate')

    # best values so far
    #  --grad_sigma 6 --warp_sigma 0.2 --lr 0.1 (cc_size 25)


    args = parser.parse_args()
    mode = args.mode
    exp_name = args.exp_name
    # mode is 
    config = {
        'split': 'val' if mode != 'test' else 'test',  # split to use
        'mode': mode,
        'lr': args.lr,
    }
    if mode == 'tune':
        # tune mode, run the registration with different parameters and save which one
        config['grad_sigma'] = tune.grid_search(np.arange(0, 8, 0.2)+0.2)
        config['warp_sigma'] = tune.grid_search(np.arange(0, 6, 0.2)+0.2)
        ray.init()
        if osp.exists(f'/home/rohitrango/ray_results/{exp_name}'):
            raise ValueError("Path already exists")
            # shutil.rmtree(f'/home/rohitrango/ray_results/{exp_name}')
            # print("Deleted existingray path")
        tuner = tune.Tuner(
            tune.with_resources(run_registration_val, resources={'cpu': 4, 'gpu': 1}),
            param_space=config,
            tune_config=tune.TuneConfig(num_samples=1),
            run_config=air.RunConfig(name=exp_name),
        )
        print("Running...")
        results = tuner.fit()
        ray.shutdown()
    elif mode == 'val':
        # val mode, run the registration with best set of parameters
        config['grad_sigma'] = args.grad_sigma
        config['warp_sigma'] = args.warp_sigma
        warp_displacement = run_registration_val(config)
        ## Save these displacement coordinates
        file = h5py.File("rnr-exm/challenge_eval/submission/mouse_val.h5", "w")
        file.create_dataset("pair4", data=warp_displacement)
        file.close()
    elif mode == 'test':
        # test mode, just save the outputs
        config['grad_sigma'] = args.grad_sigma
        config['warp_sigma'] = args.warp_sigma
        run_registration_test(config)  # this will automatically create the submission file
    else:
        raise ValueError("Invalid mode")
