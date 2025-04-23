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
import matplotlib.pyplot as plt
import numpy as np
torch.backends.cudnn.benchmark = True

def test_affine_crosscorrelation(fused_ops, fused_cc, fixed_image_path, moving_image_path, fixed_seg_path=None, moving_seg_path=None,
                                 iterations=[250, 100, 50]):
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

    fixed_image = Image.load_file(fixed_image_path)
    moving_image = Image.load_file(moving_image_path)
    fixed_batch = BatchedImages([fixed_image, ])
    moving_batch = BatchedImages([moving_image, ])
    mem = torch.cuda.max_memory_allocated()
    results['input_memory'] = mem / 1024**2

    # initialize registration
    loss = 'fusedcc' if fused_cc else 'cc'
    start = time()
    reg = AffineRegistration([4, 2, 1], iterations,
                             fixed_batch, moving_batch, 
                             optimizer_lr=1e-2, 
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
        fixed_seg = Image.load_file(fixed_seg_path, is_segmentation=True)
        moving_seg = Image.load_file(moving_seg_path, is_segmentation=True)
        fixed_batch = BatchedImages([fixed_seg, ])
        moving_batch = BatchedImages([moving_seg, ])
        moved_seg = reg.evaluate(fixed_batch, moving_batch)
        dice_score = 1 - dice_loss(moved_seg, fixed_batch())
        results['dice_score'] = dice_score.item()
    
    return results


if __name__ == "__main__":
    import os
    fixed_image_path = f"{os.environ['DATA_PATH2']}/neurite-OASIS/OASIS_OAS1_0247_MR1/aligned_norm.nii.gz"
    # fixed_seg_path = f"{os.environ['DATA_PATH2']}/neurite-OASIS/OASIS_OAS1_0247_MR1/aligned_seg35.nii.gz"
    moving_image_path = f"{os.environ['DATA_PATH2']}/neurite-OASIS/OASIS_OAS1_0186_MR1/aligned_norm.nii.gz"
    # moving_seg_path = f"{os.environ['DATA_PATH2']}/neurite-OASIS/OASIS_OAS1_0186_MR1/aligned_seg35.nii.gz"
    fixed_seg_path = None
    moving_seg_path = None
    # warmup
    _ = test_affine_crosscorrelation(False, False, fixed_image_path, moving_image_path, fixed_seg_path, moving_seg_path, [5, 5, 5])
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    allresults = {}
    # torch.cuda.memory._record_memory_history()
    for fused_cc, fused_ops in itertools.product([False, True], [False, True]):
        # reset stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        # print stats
        mem = torch.cuda.memory_allocated()
        print(f"Memory allocated before function call: {mem / 1024**2} MB")
        print(f"Peak memory allocated before function call: {torch.cuda.max_memory_allocated() / 1024**2} MB")
        results = test_affine_crosscorrelation(fused_ops, fused_cc, fixed_image_path, moving_image_path, fixed_seg_path, moving_seg_path)
        print(f"Results for fused_cc: {fused_cc}, fused_ops: {fused_ops}")
        print(json.dumps(results, indent=4))
        allresults[fused_cc, fused_ops] = results
        print("----------------------------------------------------------------")
    
    # torch.cuda.memory._dump_snapshot("memory_history/benchmark_affine_cc.pkl")
    # Extract memory usage data
    keys = list(itertools.product([False, True], [False, True]))
    keyslegend = [f"fused_cc: {key[0]}\nfused_ops: {key[1]}" for key in keys]
    
    # Plot memory usage
    memory_data = [allresults[key]['extra_memory'] for key in keys]
    plt.figure(figsize=(10, 6))
    plt.bar(keyslegend, memory_data)
    plt.title('Memory Usage by Method')
    plt.ylabel('Memory Usage (MB)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("plots/benchmark_affine_cc.png")
    plt.close()

    # Plot time data
    time_data = [allresults[key]['time'] for key in keys]
    plt.figure(figsize=(10, 6))
    plt.bar(keyslegend, time_data)
    plt.title('Time Taken by Method')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("plots/benchmark_affine_cc_time.png")
    plt.close()