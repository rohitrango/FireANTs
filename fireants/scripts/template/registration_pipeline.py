from fireants.registration.rigid import RigidRegistration
from fireants.registration.affine import AffineRegistration
from fireants.registration.moments import MomentsRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
from fireants.io.image import BatchedImages

import os
import torch
from typing import Optional, List
from omegaconf import DictConfig
from logging import Logger

from fireants.scripts.template.template_helpers import add_shape
from fireants.io.image import FakeBatchedImages

def register_batch(
        init_template_batch: BatchedImages, 
        moving_images_batch: BatchedImages, 
        args: DictConfig, 
        logger: Logger,
        avg_warp: Optional[torch.Tensor],
        identifiers: List[str],
        is_last_epoch: bool = False,
    ):
    '''
    Register a batch of moving images to the template, and return the moved images.
    '''
    # variables to keep track of 
    moved_images = None
    # initialization of affine and deformable stages
    init_rigid = None
    init_affine = None
    # moments variables
    init_moment_rigid = None
    init_moment_transl = None

    # create file names for moved images
    moved_file_names = [os.path.join(args.save_dir, f"{identifier}.nii.gz") for identifier in identifiers]

    if args.do_moments:
        logger.debug("Running moments registration")
        moments = MomentsRegistration(fixed_images=init_template_batch, \
                                        moving_images=moving_images_batch, \
                                        **dict(args.moments))
        moments.optimize()
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
        rigid.optimize()
        init_rigid = rigid.get_rigid_matrix()
        if args.last_reg == 'rigid':
            moved_images = rigid.evaluate(init_template_batch, moving_images_batch)
            # save the transformed images (todo: change this to some other save format)
            if is_last_epoch and args.save_moved_images:
                FakeBatchedImages(moved_images, init_template_batch).write_image(moved_file_names)
            # save shape
            avg_warp = add_shape(avg_warp, rigid)
        del rigid
    
    if args.do_affine:
        logger.debug("Running affine registration")
        affine = AffineRegistration(fixed_images=init_template_batch, \
                                    moving_images=moving_images_batch, \
                                        init_rigid=init_rigid, \
                                        **dict(args.affine))
        affine.optimize()
        init_affine = affine.get_affine_matrix()
        if args.last_reg == 'affine':
            moved_images = affine.evaluate(init_template_batch, moving_images_batch)
            if is_last_epoch and args.save_moved_images:
                FakeBatchedImages(moved_images, init_template_batch).write_image(args.save_dir)
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
        deform.optimize()
        moved_images = deform.evaluate(init_template_batch, moving_images_batch)
        if is_last_epoch and args.save_moved_images:
            FakeBatchedImages(moved_images, init_template_batch).write_image(moved_file_names)
        # save shape
        avg_warp = add_shape(avg_warp, deform)
        del deform

    return moved_images, avg_warp