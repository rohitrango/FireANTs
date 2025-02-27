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
from torch import distributed as dist

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F

from fireants.io.image import Image, BatchedImages
from fireants.registration import RigidRegistration, AffineRegistration, GreedyRegistration, SyNRegistration, MomentsRegistration
from fireants.scripts.template.template_helpers import *
from fireants.utils.imageutils import LaplacianFilter
from fireants.utils.warputils import shape_averaging_invwarp

logger = logging.getLogger("build_template")
logger.setLevel(logging.INFO)

def normalize(img):
    return (img - img.min()*1.0) / (img.max()*1.0 - img.min())

def setup_distributed(local_rank, world_size):
    '''
    Setup distributed training
    '''
    global logger
    logger.info(f'Setting up distributed training with local rank {local_rank} and world size {world_size}.')
    dist.init_process_group(backend='nccl')

def dist_cleanup(world_size):
    ''' cleanup distributed training '''
    global logger
    logger.info('Cleaning up distributed training')
    dist.destroy_process_group()

def add_shape(avg_warp: torch.Tensor, reg):
    if avg_warp is None:
        return None
    avg_warp = avg_warp + reg.get_warped_coordinates(reg.fixed_images, reg.moving_images).sum(0, keepdim=True)
    return avg_warp

def try_add_to_config(args, key, default):
    ''' 
    args: hydra args object 
    key: string key to check
    default: default value to add if key is not found
    '''
    try:
        _ = args[key]
    except:
        OmegaConf.update(args, key, default, force_add=True)
        

@hydra.main(version_base="1.2", config_path="./configs", config_name="oasis_deformable")
def main(args):
    '''
    Main function to build the template
    '''
    global logger
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    try_add_to_config(args, 'orientation', None)

    if local_rank == 0:
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

    # set up DDP
    setup_distributed(local_rank, world_size)
    logger_zero = logger if local_rank == 0 else None       # a reference to logger for stuff only where I want to print once
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    # set up laplacian filter
    laplace = LaplacianFilter(dims=3, device=device, **dict(args.laplace_params))

    if local_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    # get image files for template creation
    # DO NOT partition them yet, we will do that later depending on init template configuration
    image_files = get_image_files(args.image_file, args.image_prefix, args.image_suffix, args.num_subjects)
    image_identifiers = get_image_files(args.image_file, None, None, args.num_subjects)[local_rank::world_size]
    total_file_count = len(image_files)

    # additional file list (to save other stuff)
    additional_files = []
    for file, prefix, suffix, is_segm in zip(args.save_additional.image_file, args.save_additional.image_prefix, args.save_additional.image_suffix, args.save_additional.is_segmentation):
        # get file paths and identifiers
        add_files = get_image_files(file, prefix, suffix, args.num_subjects)
        add_identifiers = get_image_files(file, None, None, args.num_subjects)
        additional_files.append((add_files[local_rank::world_size], add_identifiers[local_rank::world_size], is_segm))          # partitioning additional files is fine
        if logger_zero is not None:
            logger_zero.info(f"Loaded files from {file}.")

    '''
    if initial template path is provided, we will use that as the initial template

    otherwise we will chunk the images, and create averages across all processes
    '''
    if args.init_template_path is not None:
        init_template = Image.load_file(args.init_template_path, device=device, orientation=args.orientation)
        if args.normalize_images:
            init_template.array = normalize(init_template.array.data)

        logger.info(f'Using provided template: {args.init_template_path}')
        # init image files
        image_files = image_files[local_rank::world_size]
        logger.info(f"Process {local_rank} has {len(image_files)}/{total_file_count} images.")
    else:
        logger.info('No initial template provided. Building template from average of images.')
        # create template from average
        # Run timer
        with Timer('Averaging all images for initial template', logger_zero):
            init_template = Image.load_file(image_files[0], device=device, orientation=args.orientation)
            if args.normalize_images:
                init_template.array = normalize(init_template.array.data)

            # chunk files
            image_files = image_files[local_rank::world_size]
            logger.info(f"Process {local_rank} has {len(image_files)}/{total_file_count} images.")
            # build template from average
            init_template_arr = None
            for imgfile in image_files:
                img = Image.load_file(imgfile, device=device, orientation=args.orientation)
                if args.normalize_images:
                    img.array = normalize(img.array.data)

                if init_template_arr is None:
                    init_template_arr = img.array
                else:
                    init_template_arr = init_template_arr + img.array
                del img
            init_template_arr = init_template_arr / total_file_count
            # add template across all processes
            dist.all_reduce(init_template_arr, op=dist.ReduceOp.SUM)
            init_template.delete_array()
            init_template.array = init_template_arr.detach()
            del init_template_arr

    # save initial template if specified
    if args.save_init_template and local_rank == 0:
        torch.save(init_template.array.cpu(), f"{args.save_dir}/init_template.pt")
        img_array = init_template.array.cpu().numpy()[0, 0]
        if args.save_as_uint8:
            img_array = (normalize(img_array) * 255.0).astype(np.uint8)
        itk_img = sitk.GetImageFromArray(img_array)
        itk_img.CopyInformation(init_template.itk_image)
        sitk.WriteImage(itk_img, f"{args.save_dir}/init_template.nii.gz")
        logger.info(f"Saved initial template to {args.save_dir}/init_template.nii.gz")

    logger.debug((init_template.shape, init_template.array.min(), init_template.array.max()))

    # Now we run the registration
    batch_size = args.batch_size
    for epoch in range(args.template_iterations):
        logger.info("")
        logger.info(f"Starting epoch {epoch}")
        is_last_epoch = epoch == args.template_iterations - 1
        # load up batches
        imgbatches = [image_files[i:i+batch_size] for i in range(0, len(image_files), batch_size)]
        imgidbatches = [image_identifiers[i:i+batch_size] for i in range(0, len(image_identifiers), batch_size)]

        # load up any additional files for last epoch
        additional_save_batches = []
        if is_last_epoch:
            for add_files, add_ids, is_segm in additional_files:
                add_filebatches = [add_files[i:i+batch_size] for i in range(0, len(add_files), batch_size)]
                add_idbatches = [add_ids[i:i+batch_size] for i in range(0, len(add_ids), batch_size)]
                additional_save_batches.append((add_filebatches, add_idbatches, is_segm))

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

        # run through batches
        for batchid, batch in enumerate(imgbatches):
            imgs = [Image.load_file(imgfile, device=device, orientation=args.orientation) for imgfile in batch]
            if args.normalize_images:
                for img in imgs:
                    img.array = normalize(img.array.data)

            moving_images_batch = BatchedImages(imgs)
            init_template_batch.broadcast(len(imgs))
            # variables to keep track of 
            moved_images = None
            # initialization of affine and deformable stages
            init_rigid = None
            init_affine = None
            # moments variables
            init_moment_rigid = None
            init_moment_transl = None

            if args.do_moments:
                logger.debug("Running moments registration")
                moments = MomentsRegistration(fixed_images=init_template_batch, \
                                              moving_images=moving_images_batch, \
                                                **dict(args.moments))
                moments.optimize(save_transformed=False)
                init_moment_rigid = moments.get_rigid_moment_init()
                init_moment_transl = moments.get_rigid_transl_init()
                init_rigid = moments.get_affine_init()      # for initializing affine if rigid is skipped

            if args.do_rigid:
                logger.debug("Running rigid registration")
                rigid = RigidRegistration(  fixed_images=init_template_batch, \
                                            moving_images=moving_images_batch, \
                                            init_translation=init_moment_transl, \
                                            init_moment=init_moment_rigid, \
                                          **dict(args.rigid))
                rigid.optimize(save_transformed=False)
                init_rigid = rigid.get_rigid_matrix()
                if args.last_reg == 'rigid':
                    moved_images = rigid.evaluate(init_template_batch, moving_images_batch)
                    # save the transformed images (todo: change this to some other save format)
                    if is_last_epoch:
                        if args.save_moved_images:
                            save_moved(moved_images, imgidbatches[batchid], args.save_dir, init_template)
                        # save additional files
                        save_additional(rigid, init_template_batch, additional_save_batches, batchid, args.save_dir)  # <-
                    # save shape
                    avg_warp = add_shape(avg_warp, rigid)
                del rigid
            
            if args.do_affine:
                logger.debug("Running affine registration")
                affine = AffineRegistration(fixed_images=init_template_batch, \
                                            moving_images=moving_images_batch, \
                                                init_rigid=init_rigid, \
                                                **dict(args.affine))
                affine.optimize(save_transformed=False)
                init_affine = affine.get_affine_matrix()
                if args.last_reg == 'affine':
                    moved_images = affine.evaluate(init_template_batch, moving_images_batch)
                    if is_last_epoch:
                        if args.save_moved_images:
                            save_moved(moved_images, imgidbatches[batchid], args.save_dir, init_template)
                        # save additional files
                        save_additional(affine, init_template_batch, additional_save_batches, batchid, args.save_dir)  # <-
                    # save shape
                    avg_warp = add_shape(avg_warp, affine)
                del affine
            
            if args.do_deform:
                logger.debug("Running deformable registration with {}".format(args.deform_algo))
                DeformableRegistration = GreedyRegistration if args.deform_algo == 'greedy' else SyNRegistration
                deform = DeformableRegistration(
                    fixed_images=init_template_batch, \
                    moving_images=moving_images_batch, \
                    init_affine=init_affine, \
                    **dict(args.deform)
                )
                # no need to check for last reg here, there is nothing beyond deformable
                moved_images = deform.optimize(save_transformed=True)[-1]   # this is relatively expensive step, get moved images here
                if is_last_epoch:
                    if args.save_moved_images:
                        save_moved(moved_images, imgidbatches[batchid], args.save_dir, init_template)
                    # save additional files
                    save_additional(deform, init_template_batch, additional_save_batches, batchid, args.save_dir)  # <-
                # save shape
                avg_warp = add_shape(avg_warp, deform)
                del deform
            
            # add it to the template
            updated_template_arr = updated_template_arr + moved_images.detach().sum(0, keepdim=True)/total_file_count
            del moved_images
        
        # update template
        dist.all_reduce(updated_template_arr, op=dist.ReduceOp.SUM)
        
        # perform shape averaging if specified
        if avg_warp is not None:
            avg_warp = avg_warp / total_file_count
            dist.all_reduce(avg_warp, op=dist.ReduceOp.SUM)
            # now we have added all the average grid coordinates, take inverse
            init_template_batch.broadcast(1)
            inverse_avg_warp = shape_averaging_invwarp(init_template_batch, avg_warp)
            updated_template_arr = laplace(updated_template_arr, itk_scale=True, learning_rate=1)
            updated_template_arr = F.grid_sample(updated_template_arr, inverse_avg_warp, align_corners=True)

        # apply laplacian filter
        for _ in range(args.num_laplacian):
            updated_template_arr = laplace(updated_template_arr)
        
        if args.normalize_images:
            updated_template_arr = normalize(updated_template_arr)

        logger.debug("Template updated...")
        logger.debug((local_rank, init_template.array.min(), init_template.array.mean(), init_template.array.max()))
        logger.debug((local_rank, updated_template_arr.min(), updated_template_arr.mean(), updated_template_arr.max()))

        # update the template array to new template
        init_template.array = updated_template_arr.detach()
        del updated_template_arr

        # save here
        if ((epoch % args.save_every == 0) or is_last_epoch) and local_rank == 0:
            torch.save(init_template.array.cpu(), f"{args.save_dir}/template_{epoch}.pt")
            img_array = init_template.array.cpu().numpy()[0, 0]
            if args.save_as_uint8:
                img_array = (normalize(img_array) * 255.0).astype(np.uint8)
            itk_img = sitk.GetImageFromArray(img_array)
            itk_img.CopyInformation(init_template.itk_image)
            sitk.WriteImage(itk_img, f"{args.save_dir}/template_{epoch}.nii.gz")
            logger.info(f"Saved template {epoch} to {args.save_dir}/template_{epoch}.nii.gz")

    # cleanup
    dist_cleanup(world_size)
    

if __name__ == '__main__':
    main()
