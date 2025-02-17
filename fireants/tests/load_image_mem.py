# check memory usage of the Image class
from fireants.io import Image, BatchedImages
import os
from glob import glob
import torch
import gc
from fireants.utils.util import get_gpu_memory

def profile_memory_usage(dirpath, dtype=torch.float32, num_images=16):
    """Profile memory usage with and without optimization"""
    results = {}
    
    # Clear GPU memory before each test
    for optimize in [True, False]:
        torch.cuda.empty_cache()
        gc.collect()
        
        # Record starting memory
        start_mem = get_gpu_memory()
        
        # Load images
        images = glob(os.path.join(dirpath, '**/aligned_norm.nii.gz'))[:num_images]
        images = [Image.load_file(image, dtype=dtype) for image in images]
        batch = BatchedImages(images, optimize_memory=optimize)
        
        # Record peak memory
        peak_mem = get_gpu_memory()
        
        # Calculate memory used
        mem_used = peak_mem - start_mem
        
        results[f'optimize_{optimize}'] = {
            'peak_memory_mb': peak_mem,
            'memory_used_mb': mem_used
        }
        
        # Clean up
        del batch
        del images
        torch.cuda.empty_cache()
        gc.collect()
    
    return results

if __name__ == '__main__':
    # Example usage
    dirpath = input("Enter the path to the OASIS directory: ") if os.environ.get('OASIS_DIRPATH') is None else os.environ.get('OASIS_DIRPATH')
    results = profile_memory_usage(dirpath, torch.float32, num_images=8)
    
    print("\nMemory Usage Results:")
    print("-" * 50)
    for condition, metrics in results.items():
        print(f"\n{condition}:")
        print(f"Peak Memory: {metrics['peak_memory_mb']:.2f} MB")
        print(f"Memory Used: {metrics['memory_used_mb']:.2f} MB")
    
    # run with bfloat16
    results = profile_memory_usage(dirpath, torch.bfloat16, num_images=8)
    print("\nMemory Usage Results with bfloat16:")
    print("-" * 50)
    for condition, metrics in results.items():
        print(f"\n{condition}:")
        print(f"Peak Memory: {metrics['peak_memory_mb']:.2f} MB")
        print(f"Memory Used: {metrics['memory_used_mb']:.2f} MB")
