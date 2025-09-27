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


''' author: rohitrango

This starter code is to build a template from a list of images and optional labelmaps
It should work for most applications, but is very easy to customize for your own needs

'''
import argparse
from typing import List
import os
import logging
logging.basicConfig()
import numpy as np

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F

from fireants.io.image import Image, BatchedImages
from fireants.registration.rigid import RigidRegistration
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
from fireants.registration.moments import MomentsRegistration
from fireants.utils.imageutils import LaplacianFilter
from fireants.utils.warputils import shape_averaging_invwarp
from fireants.interpolator import fireants_interpolator

from fireants.scripts.template.template_helpers import *
from fireants.scripts.template.datautils import get_image_dataloader
from fireants.scripts.template.registration_pipeline import register_batch
from fireants.registration.distributed import parallel_state

logger = logging.getLogger("build_template")
logger.setLevel(logging.INFO)

def setup_distributed(world_size):
    '''
    Setup distributed training
    '''
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    logger.info(f'Setting up distributed training with local rank {local_rank} and world size {world_size}.')
    parallel_state.initialize_parallel_state(data_parallel_size=world_size)

def dist_cleanup(world_size):
    ''' cleanup distributed training '''
    logger.info('Cleaning up distributed training')
    parallel_state.cleanup_parallel_state()

@hydra.main(version_base="1.2", config_path="./configs", config_name="oasis_deformable")
def main(args):
    '''
    Main function to build the template
    '''
    global logger
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    setup_distributed(world_size)
    assert parallel_state.is_initialized(), "Parallel state not initialized"
    assert parallel_state.get_grid_parallel_size() == 1, "Sharded template building is not supported yet"
    rank = parallel_state.get_data_parallel_rank()

    try_add_to_config(args, 'orientation', None)

    if rank == 0:
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    # check for bad argument config
    if not check_args(args, logger):
        dist_cleanup(world_size)
        return

    debug = args.debug
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode activated.")

    # set up parallel state (this will also set up current_device etc)
    device = parallel_state.get_device()
    logger_zero = logger if rank == 0 else None       # a reference to logger for stuff only where I want to print once

    # set up laplacian filter
    laplace = LaplacianFilter(dims=3, device=device, **dict(args.laplace_params))

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    # get image files (partitioned) for template creation
    image_dataloader = get_image_dataloader(args)
    image_dp_frac = len(image_dataloader.dataset) * 1.0 / image_dataloader.dataset.total_num_rows  # fraction of images in this dp rank
    total_dataset_size = image_dataloader.dataset.total_num_rows

    if args.init_template_path is not None:
        files = args.init_template_path.split(',')
        init_template = [Image.load_file(file, device=device, orientation=args.orientation) for file in files]
        init_template[0].concatenate(*init_template[1:], optimize_memory=True)
        init_template = init_template[0]
        if args.normalize_images:
            init_template.array = normalize(init_template.array.data)
        logger.info(f'Using provided multimodal template: {args.init_template_path}')
    else:
        # create template from average
        # Run timer
        with Timer('Averaging all images for initial template', logger_zero):
            init_template = get_average_template(image_dataloader, args)
            init_template.array = init_template.array * image_dp_frac
            # add template across all processes
            parallel_state.all_reduce_across_dp_ranks(init_template.array, op=torch.distributed.ReduceOp.SUM)

    logger.debug((init_template.shape, init_template.array.min(), init_template.array.max()))

    # Now we run the registration
    for epoch in range(args.template_iterations):
        logger.info("")
        logger.info(f"Starting epoch {epoch}")
        is_last_epoch = epoch == args.template_iterations - 1

        # init template batch
        # new template array and batchified into batch size of 1
        # it can be "broadcasted" to whatever size of moving images we have
        updated_template_arr = torch.zeros_like(init_template.array, device=device)
        init_template_batch = BatchedImages([init_template], optimize_memory=False)

        # if shape averaging is true, we need to keep track of warp statistics as well
        if args.shape_avg:
            B, C, H, W, D = init_template.array.shape
            avg_warp = torch.zeros((1, H, W, D, 3), device=device)
        else:
            avg_warp = None

        # load up batches
        for batch in image_dataloader:
            batch_data = batch['image']
            batch_size = batch_data.shape[0]
            init_template_batch.broadcast(batch_size)
            # register the batch
            moved_images, avg_warp = register_batch(
                init_template_batch,
                batch_data,
                args,
                logger,
                avg_warp,
                batch['identifier'],
                is_last_epoch,
            )
            # add it to the template
            updated_template_arr = updated_template_arr + moved_images.detach().sum(0, keepdim=True) * image_dp_frac
            del moved_images
        
        # update template
        parallel_state.all_reduce_across_dp_ranks(updated_template_arr, op=torch.distributed.ReduceOp.SUM)
        
        # perform shape averaging if specified
        if avg_warp is not None:
            avg_warp = avg_warp / total_dataset_size   # we computed sums, so we need to divide by total dataset size
            parallel_state.all_reduce_across_dp_ranks(avg_warp, op=torch.distributed.ReduceOp.SUM)
            # now we have added all the average grid coordinates, take inverse
            init_template_batch.broadcast(1)
            inverse_avg_warp = shape_averaging_invwarp(init_template_batch, avg_warp)
            updated_template_arr = laplace(updated_template_arr, itk_scale=True, learning_rate=1)
            updated_template_arr = fireants_interpolator(input=updated_template_arr, affine=None, grid=inverse_avg_warp, mode='bilinear', align_corners=True)

        # apply laplacian filter
        for _ in range(args.num_laplacian):
            updated_template_arr = laplace(updated_template_arr)
        
        if args.normalize_images:
            updated_template_arr = normalize(updated_template_arr)

        logger.debug("Template updated...")
        logger.debug(f"Rank {rank}, init template (min/mean/max): {init_template.array.min()}, {init_template.array.mean()}, {init_template.array.max()}")
        logger.debug(f"Rank {rank}, updated template (min/mean/max): {updated_template_arr.min()}, {updated_template_arr.mean()}, {updated_template_arr.max()}")

        # update the template array to new template
        init_template.array = updated_template_arr.detach()
        del updated_template_arr

        # save here
        if ((epoch % args.save_every == 0) or is_last_epoch) and rank == 0:
            img_array = init_template.array.cpu().numpy()[0, 0]
            if args.save_as_uint8:
                img_array = (normalize(img_array) * 255.0).astype(np.uint8)
            itk_img = sitk.GetImageFromArray(img_array)
            itk_img.CopyInformation(init_template.itk_image)
            sitk.WriteImage(itk_img, f"{args.save_dir}/template_{epoch}.nii.gz")
            logger.info(f"Saved template {epoch} to {args.save_dir}/template_{epoch}.nii.gz")

    # cleanup
    parallel_state.cleanup_parallel_state()
    

if __name__ == '__main__':
    main()
