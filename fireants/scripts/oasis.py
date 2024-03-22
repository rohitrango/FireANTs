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
import pickle
import itertools
from torch.nn import functional as F

from ray import tune, air
import os.path as osp
import ray

ROOT_DIR = "/data/rohitrango/OASIS/"

def dice_score(p, q):
    ''' computes the dice score between two tensors '''
    return 2.0 * (p * q).mean() / (p.mean() + q.mean())

def register_val_dataset(config, test=False):
    ''' given a configuration, register the images to each other determined by the random seed,
     and compute the overall dice score 
     '''
    rng = np.random.RandomState(config['seed'])
    images = sorted(glob(osp.join(ROOT_DIR, 'imagesTr', '*.nii.gz')))
    labels = sorted(glob(osp.join(ROOT_DIR, 'labelsTr', '*.nii.gz')))
    assert len(images) == len(labels)
    # tune
    if not test:
        pairs = itertools.product(range(len(images)), range(len(images)))
        pairs = list(filter(lambda x: x[0] != x[1], pairs))
        pairs = rng.permutation(pairs)[:config['num_val']]
    else:
        pairs = [(x, x+1) for x in range(len(images)-1)]
        pairs.append([pairs[-1][1], pairs[0][0]])
    print("Using #pairs: ", len(pairs))

    config['cc_size'] = int(np.around(config['cc_size']))
    # for each pair, register the images
    dices_all = []
    for fixed_id, moving_id in pairs:
        fixed_image, moving_image = Image.load_file(images[fixed_id]), Image.load_file(images[moving_id])
        fixed_image, moving_image = BatchedImages(fixed_image), BatchedImages(moving_image)
        # register
        if config['algo'] == 'greedy':
            deformable = GreedyRegistration([4, 2, 1], [200, 100, 50],
                                            fixed_image, moving_image, deformation_type='compositive',
                                            optimizer='adam', optimizer_lr=config['lr'], cc_kernel_size=1 + 2*config['cc_size'],   # 2k + 1
                                            smooth_grad_sigma=config['grad_sigma'],
                                            smooth_warp_sigma=config['warp_sigma'])
        elif config['algo'] == 'syn':
            deformable = SyNRegistration([4, 2, 1], [200, 100, 50],
                                            fixed_image, moving_image, deformation_type='compositive',
                                            optimizer='adam', optimizer_lr=config['lr'], cc_kernel_size=1 + 2*config['cc_size'],   # 2k + 1
                                            smooth_grad_sigma=config['grad_sigma'],
                                            smooth_warp_sigma=config['warp_sigma'])
        else:
            raise NotImplementedError
        # deformation
        deformable.optimize(save_transformed=False)
        warp = deformable.get_warped_coordinates(fixed_image, moving_image)
        del deformable

        # evaluate
        fixed_seg, moving_seg = Image.load_file(labels[fixed_id]), Image.load_file(labels[moving_id])
        fixed_data, moving_data = fixed_seg.array, moving_seg.array

        # compute metrics
        common_labels = set(torch.unique(fixed_data).tolist()).intersection(set(torch.unique(moving_data).tolist()))
        dice_img = []
        for lab in common_labels:
            if lab == 0:
                continue
            moving_seg_label = (moving_data == lab).float()
            fixed_seg_label = (fixed_data == lab).float()
            moved_seg_label = F.grid_sample(moving_seg_label, warp, mode='bilinear', align_corners=True)
            # dice
            dice = dice_score(moved_seg_label, fixed_seg_label).item()
            dice_img.append(dice)
        dice_img = np.mean(dice_img)
        dices_all.append(dice_img)
        print(f"Pair: {fixed_id}, {moving_id}, Dice: {dice_img}, Avg. dice: {np.mean(dices_all)}")
    # compute the overall dice score
    if not test:
        tune.report(dice=np.mean(dices_all))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="/data/rohitrango/OASIS/")
    parser.add_argument('--rng_seed', type=int, default=86781)  # random scribble
    parser.add_argument('--mode', type=str, required=True, choices=['tune', 'test'])
    parser.add_argument('--algo', type=str, required=True, choices=['greedy', 'syn'])
    parser.add_argument('--num_val', type=int, default=50)  # number of validation cases 
    parser.add_argument('--num_samples', type=int, default=1000)  # number of validation cases 
    args = parser.parse_args()

    # what configs to run
    if args.mode == 'tune':
        rem_configs = args.num_samples / 40
        s = int(np.sqrt(rem_configs))
        small_fac, large_fac = int(s/np.sqrt(2)), int(s * np.sqrt(2))

        config = {
            'seed': args.rng_seed,
            'num_val': args.num_val,
            'algo': args.algo,
            #'lr': tune.uniform(0, 1),
            #'grad_sigma': tune.uniform(0, 3),
            #'warp_sigma': tune.uniform(0, 2),
            #'cc_size': tune.uniform(1, 4),
            'lr': tune.grid_search(np.arange(0.1, 1.1, 10)),  # 10
            'grad_sigma': tune.grid_search(np.arange(0, 3, large_fac)),
            'warp_sigma': tune.grid_search(np.arange(0, 3, small_fac)),
            'cc_size': tune.grid_search([1, 2, 3, 4]),   # 4
        }
        # set algo
        from ray.tune.search.bayesopt import BayesOptSearch
        ray.init()
        scheduler = ray.tune.schedulers.FIFOScheduler()

        tuner = tune.Tuner(
            tune.with_resources(register_val_dataset, resources={'cpu': 0.5, 'gpu': 0.25}),
            tune_config=tune.TuneConfig(
                metric='dice',
                mode='max',
                num_samples=1,
                scheduler=scheduler,
            ),
            run_config=air.RunConfig(name="oasis_{}_{}samples".format(args.algo, args.num_samples)),
            param_space=config,
        )
        results = tuner.fit()
        ray.shutdown()
    elif args.mode == 'test':
        config = {
            'seed': args.rng_seed,
            'num_val': args.num_val,
            'algo': args.algo,
            'lr': 0.5,
            'grad_sigma': 1,
            'warp_sigma': 0.5,
            'cc_size': 3,
        }
        register_val_dataset(config, test=True)

    
