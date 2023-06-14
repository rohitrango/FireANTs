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

# global parameters
ROOT_DIR = "/data/rohitrango/RnR-ExM"

def dice_score(moved_seg, fixed_seg):
    # get dice score b/w moved and fixed segmentations
    moved_flat = moved_seg.flatten(1)
    fixed_flat = fixed_seg.flatten(1)
    num = 2*(moved_flat*fixed_flat).sum(1)
    den = moved_flat.sum(1) + fixed_flat.sum(1)
    return (num/den).mean()

def normalize_intensity(image):
    m, M = image.min(), image.max()
    return (image - m)/(M - m) 

def load_zebrafish(path, mp=1, Mp=99):
    ''' given path to h5py file, load the fixed and moving images '''
    ### To prevent huge load times, I've saved the images directly as nifti files
    image = Image.load_file(path, is_segmentation=False)
    imgnumpy = image.array.cpu().numpy()
    m, M  = np.percentile(imgnumpy, [mp, Mp])
    image.array = torch.clamp(image.array, m, M)
    image.array = normalize_intensity(image.array)
    # image.array = image.array[..., ::2, ::2]
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
    else:
        print("Running val...")
    # load path dir
    path_dir = osp.join(ROOT_DIR, f"{split}")
    # only this one mouse brain to use
    fixed_image, moving_image = osp.join(path_dir, "zebrafish_pair4_fixed.nii.gz"), osp.join(path_dir, "zebrafish_pair4_moving.nii.gz")
    fixed_image, moving_image = load_zebrafish(fixed_image), load_zebrafish(moving_image)
    print(fixed_image().shape, moving_image().shape)
    cc_size = 7

    ####
    #### Step 1: Affine registration
    ####
    # # init center alignment
    rigid = torch.eye(3, 4).cuda()[None]  # [1, 3, 4]
    Zf, Yf, Xf = fixed_image().shape[2:]
    Zm, Ym, Xm = moving_image().shape[2:]
    spacing = fixed_image.images[0].itk_image.GetSpacing()  # spacing of x, y, z
    spacing2 = moving_image.images[0].itk_image.GetSpacing()  # spacing of x, y, z
    print(spacing, spacing2)
    print(Zf, Yf, Xf)
    print(Zm, Ym, Xm)
    # init rigid translation
    rigid[0, :, -1] = torch.tensor([Xf/2 - Xm/2, Yf/2 - Ym/2, Zf/2 - Zm/2]).cuda() * torch.tensor(spacing).cuda()
    transl = rigid[:, :, -1].clone()  # [1, 3]
    print(rigid)

    # # affine registration
    # # affine = AffineRegistration([4], [50], fixed_image, moving_image, \
    iters = [2500, 2000, 500, 250]
    loss_params = {
        'num_bins': 32,
    }
    affine = AffineRegistration([32, 16, 8, 4], iters, fixed_image, moving_image, \
            loss_params=loss_params,
            loss_type='mi', optimizer='Adam', optimizer_lr=5e-3, optimizer_params={}, cc_kernel_size=cc_size, init_rigid=rigid, mi_kernel_type='gaussian')
    affine.optimize(save_transformed=False)
    affine_matrix = affine.get_affine_matrix().detach()
    # print(affine_matrix)
    # affine = AffineRegistration([2], \
    #                             [150], fixed_image, moving_image, \
    #         loss_type='cc', optimizer='Adam', optimizer_lr=1e-3, moved_mask=True,
    #         optimizer_params={}, cc_kernel_size=11, init_rigid=affine_matrix[:, :3, :])
    # affine.optimize(save_transformed=False)
    # affine_matrix = affine.get_affine_matrix().detach()
    del affine
    print(affine_matrix)
    torch.cuda.empty_cache()
    print("Clearing cache...")

    # compute grid
    # affine_t2t = torch.matmul(affine_matrix, fixed_image.get_torch2phy())
    # affine_t2t = torch.matmul(moving_image.get_phy2torch(), affine_t2t)
    # affine_t2t = affine_t2t[:, :3, :]  # [N, 3, 4]

    # grid_affine = F.affine_grid(affine_t2t, fixed_image().shape, align_corners=True)  # [N, H, W, D, 3]
    
    ### Step 2: deformable registration
    deformable = GreedyRegistration(scales=[4, 2], iterations=[50, 50], fixed_images=fixed_image, moving_images=moving_image,
                                    cc_kernel_size=cc_size, deformation_type='compositive', 
                                    smooth_grad_sigma=config['grad_sigma'], smooth_warp_sigma=config['warp_sigma'],
                                    optimizer='adam', optimizer_lr=0.5, init_affine=affine_matrix)
    deformable.optimize(save_transformed=False)
    shape = fixed_image().shape[2:]
    grid_affine = deformable.get_warped_coordinates(fixed_image, moving_image, shape=shape).detach()
    del deformable

    # fixed_size = fixed_image().shape[2:]
    # del fixed_image, moving_image #, moved_brain
    # permute_vtoimg = (0, 4, 1, 2, 3)
    # permute_imgtov = (0, 2, 3, 4, 1)
    # now evaluate on the actual segmenation
    with torch.no_grad():
        # grid_affine = deformable.get_warped_coordinates(fixed_image, moving_image, shape=shape).detach()
        # del deformable
        # moved_brain = deformable.evaluate(fixed_image, moving_image, shape=shape).detach().cpu().numpy()[0, 0]  # [H, W, D]
        moved_brain = F.grid_sample(moving_image(), grid_affine, mode='bilinear', align_corners=True)[0, 0].detach().cpu().numpy()
        moved_brain = (moved_brain * 1000).astype(np.uint16)
        moved_brain = sitk.GetImageFromArray(moved_brain)
        moved_brain.CopyInformation(fixed_image.images[0].itk_image)
        sitk.WriteImage(moved_brain, "moved_zebrafish.nii.gz")

        ### TODO: Need to compute dice score
        fixed_seg, moving_seg = osp.join(path_dir, "zebrafish_pair4_segmentation_fixed.nii.gz"), osp.join(path_dir, "zebrafish_pair4_segmentation_moving.nii.gz")
        fixed_seg, moving_seg = Image.load_file(fixed_seg), Image.load_file(moving_seg)
        # get the arrays
        fixed_data, moving_data = fixed_seg.array, moving_seg.array
        # fixed_data, moving_data = fixed_data > 0, moving_data > 0

        labels = set(torch.unique(fixed_data).tolist()).intersection(set(torch.unique(moving_data).tolist()))
        print(fixed_data.shape, moving_data.shape, labels)
        del fixed_seg, moving_seg
        dices = []
        for lab in labels:
            if lab == 0:
                continue
            moving_seg_label = (moving_data == lab).float()
            moved_seg_label = F.grid_sample(moving_seg_label, grid_affine, mode='bilinear', align_corners=True)
            fixed_seg_label = (fixed_data == lab).float()
            moved_seg_label = (moved_seg_label >= 0.5)
            # compute
            dice = dice_score(fixed_seg_label, moved_seg_label).item()
            print("dice", dice)
            dices.append(dice)
            del moved_seg_label
        dice = np.mean(dices)

        print("Dice score: ", dice)
        input("enter...")
        # capture all the labels
        if mode == 'tune':
            tune.report(**{'dice': dice})

    # warp is saved at this point
    # return depending on what is needed
    if mode == 'tune':
        return {'dice': dice}
    else:
        return None

def run_registration_test(config):
    ''' given a config, run registation on the test images '''
    # load path dir
    path_dir = osp.join(ROOT_DIR, "test")
    # only this one mouse brain to use
    output_file = h5py.File("rnr-exm/challenge_eval/submission/zebrafish_test.h5", "a")
    cc_size = 15
    for brain_id in [5, 6, 7]:
    # for brain_id in [6]:
    # for brain_id in [5]:
        m, M = 1, 99
        m, M = 0, 100
        #### set parameters
        if brain_id == 5:
            # m, M = 5, 95
            scales = [32, 16, 8, 4]
            iters = [2500, 2000, 500, 250]
            # scales = [16, 8, 4]
            # iters = [2000, 500, 250]
            lr = 5e-3
            moved_mask = False
            num_bins = 48
        elif brain_id == 6:
            # scales = [16, 8, 4]
            # iters = [2000, 500, 250]
            scales = [32, 16, 8, 4]
            iters = [2500, 2000, 500, 250]
            lr = 5e-3
            num_bins = 24
            moved_mask = True
        elif brain_id == 7:
            scales = [32, 16, 8, 4]
            iters = [2500, 2000, 500, 250]
            # scales = [16, 8, 4]
            # iters = [2000, 500, 250]
            lr = 5e-3
            moved_mask = False
            num_bins = 48
        # get brains
        fixed_image, moving_image = osp.join(path_dir, f"zebrafish_pair{brain_id}_fixed.nii.gz"), osp.join(path_dir, f"zebrafish_pair{brain_id}_moving.nii.gz")
        fixed_image, moving_image = load_zebrafish(fixed_image, m, M), load_zebrafish(moving_image, m, M)
        # init center alignment
        rigid = torch.eye(3, 4).cuda()[None]  # [1, 3, 4]
        Zf, Yf, Xf = fixed_image().shape[2:]
        Zm, Ym, Xm = moving_image().shape[2:]
        spacing = fixed_image.images[0].itk_image.GetSpacing()  # spacing of x, y, z
        # init rigid translation
        rigid[0, :, -1] = torch.tensor([Xf/2 - Xm/2, Yf/2 - Ym/2, Zf/2 - Zm/2]).cuda() * torch.tensor(spacing).cuda()
        transl = rigid[:, :, -1].clone()  # [1, 3]
        print(rigid)

        ##############################################
        # Step 1 : affine registration
        ##############################################
        loss_params = {
            'num_bins': num_bins, 
        }
        if brain_id == 5 and False:
            # rigid = RigidRegistration(scales, iters, fixed_image, moving_image, \
            #         loss_params=loss_params, scaling=True,
            #         loss_type='mi', optimizer='Adam', optimizer_lr=lr, optimizer_params={}, cc_kernel_size=cc_size, init_translation=transl, mi_kernel_type='gaussian')
            # rigid.optimize(save_transformed=False)
            # affine_matrix = rigid.get_rigid_matrix().detach()
            # del rigid
            affine = AffineRegistration(scales, iters, fixed_image, moving_image, \
                    loss_params=loss_params,
                    loss_type='mi', optimizer='Adam', optimizer_lr=lr,
                    optimizer_params={}, cc_kernel_size=cc_size, init_rigid=rigid, mi_kernel_type='gaussian')
            affine.optimize(save_transformed=False)
            affine_matrix = affine.get_affine_matrix().detach()
            del affine
        else:
            affine = AffineRegistration(scales, iters, fixed_image, moving_image, \
                    loss_params=loss_params,
                    moved_mask=moved_mask,
                    loss_type='mi', optimizer='Adam', optimizer_lr=lr, 
                    optimizer_params={}, cc_kernel_size=cc_size, init_rigid=rigid, mi_kernel_type='gaussian')
            affine.optimize(save_transformed=False)
            affine_matrix = affine.get_affine_matrix().detach()
            del affine
        print(affine_matrix)
        torch.cuda.empty_cache()
        print("Clearing cache...")

        # compute grid
        affine_t2t = torch.matmul(affine_matrix, fixed_image.get_torch2phy())
        affine_t2t = torch.matmul(moving_image.get_phy2torch(), affine_t2t)
        affine_t2t = affine_t2t[:, :3, :]  # [N, 3, 4]

        grid_affine = F.affine_grid(affine_t2t, fixed_image().shape, align_corners=True)  # [N, H, W, D, 3]
        grid_init = F.affine_grid(torch.eye(3, 4, device=affine_t2t.device)[None], fixed_image().shape, align_corners=True)  # [N, H, W, D, 3]

        ##############################################
        # Step 2 : deformable registration
        ##############################################
        # deformable = GreedyRegistration(scales=[4], iterations=[0], fixed_images=fixed_image, moving_images=moving_image,
        #                             cc_kernel_size=cc_size, deformation_type='compositive', 
        #                             smooth_grad_sigma=config['grad_sigma'], smooth_warp_sigma=config['warp_sigma'],
        #                             optimizer='sgd', optimizer_lr=0.0, init_affine=affine_matrix)
        # deformable.optimize(save_transformed=False)
        # # now evaluate on the actual segmenation
        # fixed_size = fixed_image().shape[2:]
        # permute_imgtov = (0, 2, 3, 4, 1)
        # permute_vtoimg = (0, 4, 1, 2, 3)

        with torch.no_grad():
            moved_image = F.grid_sample(moving_image(), grid_affine, mode='bilinear', align_corners=True)[0, 0].detach().cpu().numpy()
            moved_image = (moved_image * 1000).astype(np.uint16)
            moved_image = sitk.GetImageFromArray(moved_image)
            moved_image.CopyInformation(fixed_image.images[0].itk_image)
            sitk.WriteImage(moved_image, f"moved_zebrafish_pair{brain_id}.nii.gz")
            print(f"Written to moved_zebrafish_pair{brain_id}.nii.gz")

            # save warp
            # Z, Y, X = Zm, Ym, Xm
            grid_affine[..., 0] = grid_affine[..., 0] * (Xm - 1)/2.0 + (Xm - 1)/2.0
            grid_affine[..., 1] = grid_affine[..., 1] * (Ym - 1)/2.0 + (Ym - 1)/2.0
            grid_affine[..., 2] = grid_affine[..., 2] * (Zm - 1)/2.0 + (Zm - 1)/2.0
            # same for grid init (identity)
            grid_init[..., 0] = grid_init[..., 0] * (Xf - 1)/2.0 + (Xf - 1)/2.0
            grid_init[..., 1] = grid_init[..., 1] * (Yf - 1)/2.0 + (Yf - 1)/2.0
            grid_init[..., 2] = grid_init[..., 2] * (Zf - 1)/2.0 + (Zf - 1)/2.0
            # difference
            warp = (grid_affine - grid_init)[0]
            warp = warp.cpu().numpy().astype(np.float32)
            print(warp.shape, warp.dtype)
            warp = warp[..., ::-1]
            warp = np.around(warp, decimals=2)
            print("Final warp shape: ", warp.shape)
            # save this

        torch.cuda.empty_cache()
        print("Clearing cache...")
        if f"pair{brain_id}" in output_file:
            del output_file[f"pair{brain_id}"]
        output_file.create_dataset(f"pair{brain_id}", data=warp, compression='gzip')

        # delete cache and prepare for next one
        # del deformable
        # sleep(5)
    # close file
    output_file.close()

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='mode to use (tune/val/test)')
    parser.add_argument('--grad_sigma', type=float, default=2, help='sigma for gradient smoothing (used in tune mode)')
    parser.add_argument('--warp_sigma', type=float, default=0.2, help='sigma for warping smoothing (used in tune mode)')
    parser.add_argument('--exp_name', type=str, default='rnr_exm_zebrafish', help='experiment name')

    args = parser.parse_args()
    mode = args.mode
    exp_name = args.exp_name
    # mode is 
    config = {
        'split': 'val' if mode != 'test' else 'test',  # split to use
        'mode': mode,
    }
    if mode == 'tune':
        # tune mode, run the registration with different parameters and save which one
        config['grad_sigma'] = tune.grid_search(np.arange(0, 16, 0.5)+0.5)
        config['warp_sigma'] = tune.grid_search(np.arange(0, 2, 0.2)+0.2)
        ray.init()
        if osp.exists(f'/home/rohitrango/ray_results/{exp_name}'):
            raise ValueError("Path already exists")
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
        # file = h5py.File("rnr-exm/challenge_eval/submission/zebrafish_val.h5", "w")
        # file.create_dataset("pair4", data=warp_displacement)
        # file.close()
    elif mode == 'test':
        # test mode, just save the outputs
        config['grad_sigma'] = args.grad_sigma
        config['warp_sigma'] = args.warp_sigma
        run_registration_test(config)  # this will automatically create the submission file
    else:
        raise ValueError("Invalid mode")
