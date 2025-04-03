'''
Script to benchmark affine registration performance with different methods

Only works if fireants_fused_ops is installed.
'''
import torch
from fireants.io import Image, BatchedImages
from fireants.registration import AffineRegistration
import fireants_fused_ops as ffo
from fireants.interpolator import fireants_interpolator
from time import time
from tests.conftest import dice_loss
import itertools
import json

def test_affine_mse(fused_ops, fixed_image_path, moving_image_path, fixed_seg_path=None, moving_seg_path=None):
    '''
    fused_ops: bool, whether to use fused ops for grid_sample, etc.
    fixed_image_path: str, path to fixed image
    moving_image_path: str, path to moving image
    fixed_seg_path: str, path to fixed segmentation
    moving_seg_path: str, path to moving segmentation
    '''
    fireants_interpolator.use_ffo = fused_ops
    print(fireants_interpolator.use_ffo)

    # capture results here
    results = {}

    fixed_image = Image.load_file(fixed_image_path)
    moving_image = Image.load_file(moving_image_path)
    fixed_batch = BatchedImages([fixed_image, ])
    moving_batch = BatchedImages([moving_image, ])
    mem = torch.cuda.max_memory_allocated()
    # mem = torch.cuda.memory_stats()['active_bytes.all.peak']
    results['input_memory'] = mem / 1024**2

    # initialize registration
    start = time()
    reg = AffineRegistration([4, 2, 1], [250, 100, 50],
                             fixed_batch, moving_batch, 
                             optimizer_lr=1e-2, 
                             blur=False,
                             loss_type='mse', optimizer='Adam', max_tolerance_iters=1000)
    reg.optimize(False)
    torch.cuda.synchronize()
    end = time()
    mem = torch.cuda.max_memory_allocated() - mem
    # mem = torch.cuda.memory_stats()['active_bytes.all.peak'] - mem
    # store results
    results['time'] = end - start
    results['extra_memory'] = mem / 1024**2

    # load segs if provided
    if fixed_seg_path is not None:
        fixed_seg = Image.load_file(fixed_seg_path, is_segmentation=True)
        moving_seg = Image.load_file(moving_seg_path, is_segmentation=True)
        fixed_batch = BatchedImages([fixed_seg, ])
        moving_batch = BatchedImages([moving_seg, ])
        moved_seg = reg.evaluate(fixed_batch, moving_batch)
        dice_score = 1 - dice_loss(moved_seg, fixed_batch())
        results['dice_score'] = dice_score.item()
    
    return results


if __name__ == "__main__":
    fixed_image_path = "/mnt/rohit_data2/neurite-OASIS/OASIS_OAS1_0247_MR1/aligned_norm.nii.gz"
    # fixed_seg_path = "/mnt/rohit_data2/neurite-OASIS/OASIS_OAS1_0247_MR1/aligned_seg35.nii.gz"
    moving_image_path = "/mnt/rohit_data2/neurite-OASIS/OASIS_OAS1_0186_MR1/aligned_norm.nii.gz"
    # moving_seg_path = "/mnt/rohit_data2/neurite-OASIS/OASIS_OAS1_0186_MR1/aligned_seg35.nii.gz"
    fixed_seg_path = None
    moving_seg_path = None
    _ = test_affine_mse(True, fixed_image_path, moving_image_path, fixed_seg_path, moving_seg_path)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.memory._record_memory_history()
    for fused_ops in [False, True]:
        # reset stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        # print stats
        mem = torch.cuda.memory_allocated()
        mem2 = torch.cuda.memory_reserved()
        print(f"Memory allocated before function call: {mem / 1024**2} MB")
        print(f"Peak memory allocated before function call: {torch.cuda.max_memory_allocated() / 1024**2} MB")
        print(f"Memory reserved before function call: {mem2 / 1024**2} MB")
        print(f"Peak memory reserved before function call: {torch.cuda.max_memory_reserved() / 1024**2} MB")
        results = test_affine_mse(fused_ops, fixed_image_path, moving_image_path, fixed_seg_path, moving_seg_path)
        print(f"Results for fused_ops: {fused_ops}")
        print(json.dumps(results, indent=4))
        print("----------------------------------------------------------------")
    
    torch.cuda.memory._dump_snapshot("memory_history/benchmark_affine_mse.pkl")