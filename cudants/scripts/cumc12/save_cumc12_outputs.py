''' example script to test CUMC dataset '''
from glob import glob
import time
import numpy as np
import torch
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
from cudants.io.image import Image, BatchedImages
from cudants.registration.affine import AffineRegistration
from cudants.registration.greedy import GreedyRegistration
from cudants.registration.syn import SyNRegistration
import argparse
from tqdm import tqdm
import pickle

DATA_DIR = "/data/rohitrango/brain_data/CUMC12/Brains"
LABEL_DIR = "/data/rohitrango/brain_data/CUMC12/Atlases"

# first label is background 
labels_all = np.array([  0,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
        15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
        43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,
        83,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
        97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
       110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
       123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133])

def seg_preprocessor(segmentation: torch.Tensor):
    ''' custom preprocessor for IBSR dataset that maps only the common structures '''
    new_segmentation = torch.zeros_like(segmentation)
    for newidx, label in enumerate(labels_all):
        new_segmentation[segmentation == label] = newidx
    return new_segmentation

# parse
parser = argparse.ArgumentParser("Test IBSR dataset")
parser.add_argument('--algo', type=str, required=True, help='algorithm to use (greedy, syn)')

if __name__ == '__main__':

    args = parser.parse_args()
    algo = args.algo
    # Get images
    images = sorted(glob(DATA_DIR + "/*img"), key=lambda x: int(x.split("/")[-1].split(".")[0][1:]))
    labels = sorted(glob(LABEL_DIR + "/*img"), key=lambda x: int(x.split("/")[-1].split(".")[0][1:]))
    num_images = len(images)
    # iterate through images
    sublist = [(3 , 5), (2 , 12), (8 , 12), (12 , 2), (1 , 2), (12 , 4), (5 , 3), (1 , 3), (9 , 2), (10 , 4), (1 , 4)]
    
    for i, j in sublist:
        i,j = i-1, j-1
        fixed_image_path = images[i]
        fixed_seg_path = labels[i]
        # load batched images
        fixed_image = BatchedImages(Image.load_file(fixed_image_path))
        fixed_seg   = BatchedImages(Image.load_file(fixed_seg_path, is_segmentation=True, seg_preprocessor=seg_preprocessor))
        # get moving image
        moving_image_path = images[j]
        moving_seg_path = labels[j]
        # load them
        moving_image = BatchedImages(Image.load_file(moving_image_path))
        moving_seg = BatchedImages(Image.load_file(moving_seg_path, is_segmentation=True, seg_preprocessor=seg_preprocessor))
        # affine pre-registration
        print("Registering {} to {}".format(fixed_image_path, moving_image_path))
        affine = AffineRegistration([8, 4, 2, 1], [100, 50, 25, 20], fixed_image, moving_image, \
            loss_type='cc', optimizer='Adam', optimizer_lr=3e-4, optimizer_params={}, cc_kernel_size=5)
        affine.optimize(save_transformed=False)
        # greedy registration
        if algo == 'greedy':
            deformable = GreedyRegistration(scales=[4, 2, 1], iterations=[200, 100, 25], fixed_images=fixed_image, moving_images=moving_image,
                                cc_kernel_size=5, deformation_type='compositive', 
                                smooth_grad_sigma=1, max_tolerance_iters=100,
                                optimizer='adam', optimizer_lr=0.5, init_affine=affine.get_affine_matrix().detach())
        elif algo == 'syn':
            deformable = SyNRegistration(scales=[4, 2, 1], iterations=[100, 75, 50], fixed_images=fixed_image, moving_images=moving_image,
                                cc_kernel_size=5, deformation_type='compositive', optimizer="adam", optimizer_lr=0.5,
                                max_tolerance_iters=100,
                                smooth_grad_sigma=1, init_affine=affine.get_affine_matrix().detach())
        else:
            raise NotImplementedError
        deformable.optimize(save_transformed=False)
        # evaluate
        moved_array = deformable.evaluate(fixed_image, moving_image)
        moved_array = moved_array[0, 0].detach().cpu().numpy()   # [H, W, C]
        # save this here
        newimg = sitk.GetImageFromArray(moved_array)
        newimg.CopyInformation(fixed_image.images[0].itk_image)
        sitk.WriteImage(newimg, "moved_{}_{}.nii.gz".format(i, j))
        print("Saved {} to {}".format(fixed_image_path, moving_image_path))
