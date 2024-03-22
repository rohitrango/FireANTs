''' 
Script to combine hyperparameter tuning and evaluation on LPBA40 dataset
hyperparam selection is done using ray.tune
The hyperparameters are:
1. optimizer_lr
2. gradient smooth sigma
3. warp smooth sigma
4. beta1, beta2  (using adam only for now)
5. algorithm (greedy or syn)

code is borrowed from `test_lpba40.py`
'''
from glob import glob
import time
import numpy as np
import torch
import SimpleITK as sitk
from fireants.io.image import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
from evaluate_metrics import compute_metrics
from itertools import product
from functools import partial
import pandas as pd
import argparse

import ray
from ray import tune, air
from ray.air import session
from ray.tune.search import ConcurrencyLimiter, BasicVariantGenerator

def seg_preprocessor_full(segmentation: torch.Tensor, labels_all: np.ndarray):
    ''' custom preprocessor for IBSR dataset that maps only the common structures '''
    new_segmentation = torch.zeros_like(segmentation)
    for newidx, label in enumerate(labels_all):
        new_segmentation[segmentation == label] = newidx
    return new_segmentation

def registration_run(config):
    ''' This is the main function that uses the config from ray.tune and
    performs image registration across the dataset
    '''
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    DATA_DIR = "/data/rohitrango/brain_data/LPBA40/registered_pairs/"
    LABEL_DIR = "/data/rohitrango/brain_data/LPBA40/registered_label_pairs/"
    # first label is background 
    labels_all = np.array([  0,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
            33,  34,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  61,
            62,  63,  64,  65,  66,  67,  68,  81,  82,  83,  84,  85,  86,
            87,  88,  89,  90,  91,  92, 101, 102, 121, 122, 161, 162, 163,
        164, 165, 166, 181, 182])
    seg_preprocessor = partial(seg_preprocessor_full, labels_all=labels_all)
    image_ids = range(1, 41)
    print(config)

    # load all values from the config
    algo = config['algo']
    lr = config['lr']
    grad_sigma = config['grad_sigma']
    warp_sigma = config['warp_sigma']

    # get all pair ids and shuffle them (to create some independence of the samples)
    pair_ids = product(image_ids, image_ids)
    pair_ids = list(filter(lambda x: x[0] != x[1], pair_ids))
    rng = np.random.RandomState(654716)
    rng.shuffle(pair_ids)

    # iterate through pairs
    target_overlap = 0  # keep running sum
    count = 0

    for (i, j) in pair_ids:
        fixed_image_path = DATA_DIR + "l{}_to_l{}.img".format(i, i)
        fixed_seg_path = LABEL_DIR + "l{}_to_l{}.img".format(i, i)
        # load batched images
        fixed_image = BatchedImages(Image.load_file(fixed_image_path))
        fixed_seg   = BatchedImages(Image.load_file(fixed_seg_path, is_segmentation=True, seg_preprocessor=seg_preprocessor))
        # get moving image
        moving_image_path = DATA_DIR + "l{}_to_l{}.img".format(j, i)
        moving_seg_path = LABEL_DIR + "l{}_to_l{}.img".format(j, i)
        # load them
        moving_image = BatchedImages(Image.load_file(moving_image_path))
        moving_seg = BatchedImages(Image.load_file(moving_seg_path, is_segmentation=True, seg_preprocessor=seg_preprocessor))
        # affine pre-registration
        print("Registering {} to {}".format(fixed_image_path, moving_image_path))
        affine = AffineRegistration([8, 4, 2, 1], [100, 50, 25, 20], fixed_image, moving_image, \
            loss_type='cc', optimizer='Adam', optimizer_lr=3e-4, optimizer_params={}, cc_kernel_size=5)
        affine.optimize(save_transformed=False)
        # greedy registration
        comp = config['deformation_type'] == 'compositive'
        if algo == 'greedy':
            deformable = GreedyRegistration(scales=[4, 2, 1], iterations=[200, 100, 25], fixed_images=fixed_image, moving_images=moving_image,
                                    cc_kernel_size=5, 
                                    deformation_type=config['deformation_type'],
                                    smooth_grad_sigma=grad_sigma, smooth_warp_sigma=warp_sigma, 
                                    optimizer_params={'beta1': config['beta1'], 'beta2': config['beta2']} if comp else {},
                                    optimizer='Adam', optimizer_lr=lr, init_affine=affine.get_affine_matrix().detach())
        elif algo == 'syn':
            deformable = SyNRegistration(scales=[4, 2, 1], iterations=[100, 75, 50], fixed_images=fixed_image, moving_images=moving_image,
                                    cc_kernel_size=5, 
                                    optimizer="Adam", optimizer_lr=lr,
                                    deformation_type=config['deformation_type'],
                                    optimizer_params={'beta1': config['beta1'], 'beta2': config['beta2']} if comp else {},
                                    smooth_grad_sigma=grad_sigma, smooth_warp_sigma=warp_sigma, init_affine=affine.get_affine_matrix().detach())
        # a = time.time()
        deformable.optimize(save_transformed=False)
        # b = time.time() - a
        # evaluate
        moved_seg_array = deformable.evaluate(fixed_seg, moving_seg)
        moved_seg_array = (moved_seg_array >= 0.5).float()
        metrics = compute_metrics(fixed_seg()[0].detach().cpu().numpy(), moved_seg_array[0].detach().cpu().numpy())
        # add them to list
        target_overlap += metrics['target_overlap_klein']
        count += 1
        # tune.report(target_overlap=metrics['target_overlap_klein'])
        tune.report(target_overlap=target_overlap/count)
    # return final score
    return target_overlap / len(pair_ids)


if __name__ == '__main__':
    # define space of parameters to tune over
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, default='syn', help='syn or greedy')
    parser.add_argument('--deformation_type', type=str, required=False, default='compositive')
    args = parser.parse_args()
    algo = args.algo
    print("Using {} algorithm".format(algo))
    comp = args.deformation_type == 'compositive'

    space = {
        'algo': algo,
        'lr': tune.grid_search(np.linspace(0.2, 1, 5)) if comp else tune.grid_search(np.linspace(0.0002, 0.005, 5)),
        'deformation_type': args.deformation_type,
        # 'beta1': tune.grid_search(np.linspace(0.1, 1, 10)),
        # 'beta2': tune.grid_search(np.linspace(0.25, 1, 16)),
        'beta1': 0.9,
        'beta2': 0.999,
        'grad_sigma': tune.grid_search(np.linspace(0, 3, 16)),
        'warp_sigma': tune.grid_search(np.linspace(0, 1, 8)),
    }
    ray.init()
    tuner = tune.Tuner(
        tune.with_resources(registration_run, resources={'cpu': 4, 'gpu': 0.5}),
        param_space=space,
        run_config=air.RunConfig(name="lpba40_{}_{}".format(args.algo, args.deformation_type)),
        tune_config=tune.TuneConfig(
            num_samples=1,
            # search_alg=BasicVariantGenerator(max_concurrent=8),
            scheduler=tune.schedulers.ASHAScheduler(
                metric="target_overlap",
                mode="max",
                time_attr="training_iteration",
                max_t=40*39,
                grace_period=40,
                reduction_factor=3,
                brackets=1,
            ),
        ),
    )
    print("Running...")
    results = tuner.fit()
    print(results)
    # save results
    # results_df = results.results_df
    # pd.to_pickle(results_df, f'lpba40_tune_results_{algo}.pkl')
    ray.shutdown()
