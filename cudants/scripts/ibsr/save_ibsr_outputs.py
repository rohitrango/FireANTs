''' example script to test IBSR dataset '''
from glob import glob
import time
import numpy as np
import torch
from cudants.io.image import Image, BatchedImages
from cudants.registration.affine import AffineRegistration
from cudants.registration.greedy import GreedyRegistration
from cudants.registration.syn import SyNRegistration
import argparse
from tqdm import tqdm
import pickle
import SimpleITK as sitk

DATA_DIR = "/data/rohitrango/brain_data/IBSR18/"
MAX_LABEL = 72
labels_all = np.array([ 0,  2,  3,  4,  5,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26,
        28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60])

def seg_preprocessor(segmentation: torch.Tensor):
    ''' custom preprocessor for IBSR dataset that maps only the common structures '''
    new_segmentation = torch.zeros_like(segmentation)
    for newidx, label in enumerate(labels_all):
        new_segmentation[segmentation == label] = newidx
    return new_segmentation

# parser
parser = argparse.ArgumentParser("Test IBSR dataset")
parser.add_argument('--algo', type=str, required=True, help='algorithm to use (greedy, syn)')

if __name__ == '__main__':
    args = parser.parse_args()
    print("Using {} algorithm".format(args.algo))

    # compile all the paths
    dirs = sorted(glob(DATA_DIR + "IBSR_*"))
    images = [glob(x + "/*ana_strip.nii.gz") for x in dirs]
    labels = [glob(x + "/*seg_ana.nii.gz") for x in dirs]
    assert all([len(x) == 1 for x in images])
    assert all([len(x) == 1 for x in labels])
    images, labels = [x[0] for x in images], [x[0] for x in labels]
    num_images = len(images)

    # save images on this sublist
    sublist = [(0, 12), (9, 2), (9, 12), (2, 9), (12, 11), (9, 3), (5, 10), (0, 2), (9, 14)]

    for i, j in sublist:
    # iterate through images
        fixed_image_path = images[i]
        fixed_seg_path = labels[i]
        # load batched images
        fixed_image = BatchedImages(Image.load_file(fixed_image_path))
        # get moving image
        moving_image_path = images[j]
        moving_seg_path = labels[j]
        # load them
        moving_image = BatchedImages(Image.load_file(moving_image_path))
        # affine pre-registration
        print("Registering {} to {}".format(fixed_image_path, moving_image_path))
        affine = AffineRegistration([8, 4, 2, 1], [200, 100, 50, 20], fixed_image, moving_image, \
            loss_type='cc', optimizer='Adam', optimizer_lr=3e-4, optimizer_params={}, cc_kernel_size=5)
        affine.optimize(save_transformed=False)
        if args.algo == 'greedy':
        # greedy registration
            deformable = GreedyRegistration(scales=[4, 2, 1], iterations=[200, 100, 25], fixed_images=fixed_image, moving_images=moving_image,
                                cc_kernel_size=5, deformation_type='compositive', 
                                smooth_grad_sigma=1, max_tolerance_iters=100,
                                optimizer='adam', optimizer_lr=0.5, init_affine=affine.get_affine_matrix().detach())
        elif args.algo == 'syn':
            deformable = SyNRegistration(scales=[4, 2, 1], iterations=[100, 50, 50], fixed_images=fixed_image, moving_images=moving_image,
                                    cc_kernel_size=5, deformation_type='compositive', optimizer="adam", optimizer_lr=0.5,
                                    optimizer_params={
                                        'beta1': 0.5, 
                                        'beta2': 0.75,},
                                    max_tolerance_iters=100,
                                    smooth_grad_sigma=1, smooth_warp_sigma=0.5, init_affine=affine.get_affine_matrix().detach(),
                                    )
        else:
            raise NotImplementedError
        # record time
        deformable.optimize(save_transformed=False)
        # evaluate
        moved_array = deformable.evaluate(fixed_image, moving_image) 
        # append to dictionaries
        # save this image
        moved_array = moved_array[0, 0].detach().cpu().numpy()   # [H, W, C]
        # save this here
        newimg = sitk.GetImageFromArray(moved_array)
        newimg.CopyInformation(fixed_image.images[0].itk_image)
        sitk.WriteImage(newimg, "moved_{}_{}.nii.gz".format(i, j))
        print("Saved {} to {}".format(fixed_image_path, moving_image_path))
