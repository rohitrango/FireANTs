'''
author: rohitrango

script to evaluate overlap metrics between label maps using template 

metrics:
- label overlap using the template as fixed image
- entropy of label map which are not background
'''

import numpy as np
import torch
import argparse
from fireants.io import Image, BatchedImages
from fireants.registration import GreedyRegistration

import logging
from logging import Logger
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    '''
    parse command line arguments
    '''
    parser = argparse.ArgumentParser(description='Evaluate overlap metrics between label maps using template')
    parser.add_argument('--template_path', type=str, help='template image')
    parser.add_argument('--images_list_file', type=str, help="file containing list of images to evaluate")
    return parser.parse_args()

@torch.no_grad()
def compute_dice(label1, label2):
    '''
    compute dice score between two label maps
    
    label1, label2: tensors of size [B, C, H, W, D]
    '''
    # get the intersection
    label1, label2 = label1.flatten(2), label2.flatten(2)
    intersection = (label1 * label2).mean(2)
    union = label1.mean(2) + label2.mean(2)
    dice = ((2 * intersection / union).mean(0)).detach().cpu().numpy()
    return dice


def compute_overlap_via_template(template, args):
    '''
    given a template image, and list of pairs, compute the overlap of dice score via these two images 
    '''
    # pandas script to read txt or csv file with , delimiter
    # assume header exists
    with open(args.images_list_file, 'r') as f:
        images_list = pd.read_csv(f, delimiter=',', header=0)
    
    # store results
    all_dice_scores = []
    
    # iterate over the images_list
    for index, row in images_list.iterrows():
        fix_img = row['fix_image']
        fix_lab = row['fix_label']
        mov_img = row['mov_image']
        mov_lab = row['mov_label']

        res = []
        ### run the registration for fixed and moving images
        ###
        for img, lab in zip([fix_img, mov_img], [fix_lab, mov_lab]):
            logger.warning(f"Processing image: {img} and label: {lab}")
            # load the images
            image = BatchedImages(Image.load_file(img))
            label = BatchedImages(Image.load_file(lab, is_segmentation=True))
            # compute the overlap
            reg = GreedyRegistration(scales=[6, 4, 2, 1], iterations=[200, 100, 50, 25], 
                                    fixed_images=template, moving_images=image, 
                                    deformation_type='compositive', reduction='sum',
                                    )
            reg.optimize()
            # get transformed label
            warpedlabel = reg.evaluate(template, label)
            res.append(warpedlabel)

        # compute the dice score
        dice_scores = compute_dice(res[0], res[1])
        logger.warning(f"Dice score: {dice_scores.mean()}")
        all_dice_scores.append(dice_scores)
    
    all_dice_scores = np.array(all_dice_scores)  # [N, C]
    
    logger.warning(f"Overall dice score: {all_dice_scores.mean()} +/- {all_dice_scores.std()}")
    logger.warning(f"Dice score by class: {all_dice_scores.mean(0)}")
    return all_dice_scores


def main():
    # main function
    args = parse_args()
    template = BatchedImages(Image.load_file(args.template_path), optimize_memory=True)
    logger.warning(f"Template shape: {template.images[0].array.shape}")
    ret_dice = compute_overlap_via_template(template, args)


if __name__ == '__main__':
    main()