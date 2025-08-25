'''
Script to benchmark greedy registration performance with different methods
Only works if fireants_fused_ops is installed.
'''
import os
import torch
from fireants.io import Image, BatchedImages
from fireants.registration.greedy import GreedyRegistration
import fireants_fused_ops as ffo
from fireants.interpolator import fireants_interpolator
from time import time
from tests.conftest import dice_loss
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np

def test_greedy_mi(fused_ops, fused_cc, fixed_image_path, moving_image_path, fixed_seg_path=None, moving_seg_path=None, iterations=[200, 100, 25], dtype=torch.float32):
    '''
    fused_ops: bool, whether to use fused ops for grid_sample, etc.
    fused_cc: bool, whether to use fused cc for cross-correlation
    fixed_image_path: str, path to fixed image
    moving_image_path: str, path to moving image
    fixed_seg_path: str, path to fixed segmentation
    moving_seg_path: str, path to moving segmentation
    '''
    fireants_interpolator.use_ffo = fused_ops

    # capture results here
    results = {}

    fixed_image = Image.load_file(fixed_image_path, dtype=dtype)
    moving_image = Image.load_file(moving_image_path, dtype=dtype)
    fixed_batch = BatchedImages([fixed_image, ])
    moving_batch = BatchedImages([moving_image, ])
    mem = torch.cuda.max_memory_allocated()
    results['input_memory'] = mem / 1024**2

    print(fixed_image.array.max())
    print(fixed_image.array.min())
    print(moving_image.array.max())
    print(moving_image.array.min())

    # initialize registration
    loss = 'fusedmi' if fused_cc else 'mi'
    # loss = 'mse'
    start = time()
    reg = GreedyRegistration([4, 2, 1], iterations,
                             fixed_batch, moving_batch, 
                             blur=True,
                             dtype=dtype if not fused_ops else torch.float32,
                             optimizer_params={'offload': False},
                             cc_kernel_size=7,
                             smooth_warp_sigma=0.25,
                             smooth_grad_sigma=0.5,
                             optimizer_lr=0.5,
                             mi_kernel_type='gaussian',
                             loss_type=loss, optimizer='Adam', max_tolerance_iters=1000)
    reg.optimize(False)
    torch.cuda.synchronize()
    end = time()
    mem = torch.cuda.max_memory_allocated() - mem
    # store results
    results['time'] = end - start
    results['extra_memory'] = mem / 1024**2

    # load segs if provided
    if fixed_seg_path is not None:
        fixed_seg = Image.load_file(fixed_seg_path, is_segmentation=True, dtype=dtype)
        moving_seg = Image.load_file(moving_seg_path, is_segmentation=True, dtype=dtype)
        fixed_batch = BatchedImages([fixed_seg, ])
        moving_batch = BatchedImages([moving_seg, ])
        moved_seg = reg.evaluate(fixed_batch, moving_batch)
        dice_score = 1 - dice_loss(moved_seg.to(float), fixed_batch().to(float))
        results['dice_score'] = dice_score.item()
    
    return results


if __name__ == "__main__":
    import os
    # variable to determine whether to run long optim
    # if this is false, we want to see memory usage and dont care about dice
    run_long_optim = os.getenv('RUN_LONG_OPTIM', 'false').lower() == 'true'
    dtype = torch.float32 if os.getenv("dtype", "fp32") == "fp32" else torch.bfloat16
    print(f"Using dtype {dtype}")
    if run_long_optim:
        iterations = [200, 100, 25]
    else:
        iterations = [10, 5, 2]

    path = os.environ['DATA_PATH2']
    fixed_image_path = f"{path}/neurite-OASIS/OASIS_OAS1_0247_MR1/aligned_norm.nii.gz"
    fixed_seg_path = f"{path}/neurite-OASIS/OASIS_OAS1_0247_MR1/aligned_seg35.nii.gz"
    moving_image_path = f"{path}/neurite-OASIS/OASIS_OAS1_0186_MR1/aligned_norm.nii.gz"
    moving_seg_path = f"{path}/neurite-OASIS/OASIS_OAS1_0186_MR1/aligned_seg35.nii.gz"
    if not run_long_optim:
        fixed_seg_path = None
        moving_seg_path = None
    # warmup
    # _ = test_greedy_mi(False, False, fixed_image_path, moving_image_path, fixed_seg_path, moving_seg_path, iterations=[2, 2, 2])
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    # torch.cuda.reset_peak_memory_stats()

    allresults = {}
    if not run_long_optim:
        torch.cuda.memory._record_memory_history()

    # run benchmarks
    for fused_cc, fused_ops in itertools.product([False, True], [False, True]):
    # for fused_cc, fused_ops in [[True, True]]:
        # reset stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        # print stats
        mem = torch.cuda.memory_allocated()
        print(f"Memory allocated before function call: {mem / 1024**2} MB")
        print(f"Peak memory allocated before function call: {torch.cuda.max_memory_allocated() / 1024**2} MB")
        results = test_greedy_mi(fused_ops, fused_cc, fixed_image_path, moving_image_path, fixed_seg_path, moving_seg_path, iterations=iterations, dtype=dtype)
        print(f"Results for fused_cc: {fused_cc}, fused_ops: {fused_ops}")
        print(json.dumps(results, indent=4))
        allresults[fused_cc, fused_ops] = results
        print("----------------------------------------------------------------")
    
    if not run_long_optim:
        torch.cuda.memory._dump_snapshot("memory_history/benchmark_greedy_mi.pkl")
