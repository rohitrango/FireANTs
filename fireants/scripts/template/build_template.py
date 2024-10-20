''' author: rohitrango

This starter code is to build a template from a list of images and optional labelmaps
It should work for most applications, but is very easy to customize for your own needs

'''
import argparse
from typing import List
import os
import logging
logging.basicConfig()

import torch
from torch import distributed as dist

import hydra
from omegaconf import DictConfig, OmegaConf

from fireants.io.image import Image, BatchedImages
from fireants.registration import RigidRegistration, AffineRegistration, GreedyRegistration, SyNRegistration
from fireants.scripts.template.template_helpers import *

logger = logging.getLogger("build_template")
logger.setLevel(logging.INFO)

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

@hydra.main(version_base="1.2", config_path="./configs", config_name="oasis_deformable")
def main(args):
    '''
    Main function to build the template
    '''
    global logger
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

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
        init_template = Image.load_file(args.init_template_path, device=device)
        logger.info(f'Using provided template: {args.init_template_path}')
        # init image files
        image_files = image_files[local_rank::world_size]
        logger.info(f"Process {local_rank} has {len(image_files)}/{total_file_count} images.")
    else:
        logger.info('No initial template provided. Building template from average of images.')
        # create template from average
        # Run timer
        with Timer('Averaging all images for initial template', logger_zero):
            init_template = Image.load_file(image_files[0], device=device)
            # chunk files
            image_files = image_files[local_rank::world_size]
            logger.info(f"Process {local_rank} has {len(image_files)}/{total_file_count} images.")
            # build template from average
            init_template_arr = None
            for imgfile in image_files:
                img = Image.load_file(imgfile, device=device)
                if init_template_arr is None:
                    init_template_arr = img.array
                else:
                    init_template_arr = init_template_arr + img.array
                del img
            init_template_arr = init_template_arr / total_file_count
            # add template across all processes
            dist.all_reduce(init_template_arr, op=dist.ReduceOp.SUM)
            init_template.delete_array()
            init_template.array = init_template_arr + 0

            # save initial template if specified
            if args.save_init_template and local_rank == 0:
                torch.save(init_template_arr.cpu(), f"{args.save_dir}/init_template.pt")

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

        # run through batches
        for batchid, batch in enumerate(imgbatches):
            imgs = [Image.load_file(imgfile, device=device) for imgfile in batch]
            moving_images_batch = BatchedImages(imgs)
            init_template_batch.broadcast(len(imgs))
            # variables to keep track of 
            moved_images = None
            # rigid registration
            init_rigid = None
            init_affine = None
            if args.do_rigid:
                logger.debug("Running rigid registration")
                rigid = RigidRegistration(fixed_images=init_template_batch, \
                                          moving_images=moving_images_batch, \
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
                moved_images = deform.optimize(save_transformed=True)[-1]   # this is relatively expensive step, get moved images here
                if is_last_epoch:
                    if args.save_moved_images:
                        save_moved(moved_images, imgidbatches[batchid], args.save_dir, init_template)
                    # save additional files
                    save_additional(deform, init_template_batch, additional_save_batches, batchid, args.save_dir)  # <-
                del deform
            
            # add it to the template
            updated_template_arr = updated_template_arr + moved_images.detach().sum(0, keepdim=True)/total_file_count
            del moved_images
        
        # update template
        dist.all_reduce(updated_template_arr, op=dist.ReduceOp.SUM)

        logger.debug("Template updated...")
        logger.debug((local_rank, init_template.array.min(), init_template.array.mean(), init_template.array.max()))
        logger.debug((local_rank, updated_template_arr.min(), updated_template_arr.mean(), updated_template_arr.max()))

        # update the template array to new template
        init_template.array = updated_template_arr + 0

        # save here
        if ((epoch % args.save_every == 0) or is_last_epoch) and local_rank == 0:
            torch.save(updated_template_arr, f"{args.save_dir}/template_{epoch}.pt")

    # cleanup
    dist_cleanup(world_size)
    

if __name__ == '__main__':
    main()