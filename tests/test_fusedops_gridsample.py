import torch
import pytest
from time import time
from torch import nn
from torch.nn import functional as F
from fireants.interpolator.fused_grid_sample import fused_grid_sampler_3d
from fireants.interpolator.grid_sample import torch_grid_sampler_3d
# set seeds
seed = 1221
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)   

def test_fused_sampler_correctness():
    """Test correctness of fused grid sampler against baseline implementation"""
    # Test with different input sizes and shapes
    rng = torch.Generator()
    rng.manual_seed(seed)

    test_cases = [
        # (input_shape, output_shape)
        ((1, 1, 32, 32, 32), (16, 16, 16)),  # Downsampling
        ((1, 1, 16, 16, 16), (32, 32, 32)),  # Upsampling
        ((1, 1, 64, 64, 64), (32, 32, 32)),  # Downsampling
        ((1, 1, 32, 32, 32), (64, 64, 64)),  # Upsampling
    ]
    for input_shape, output_shape in test_cases:
        # Generate random input and affine matrix
        input = torch.randn(input_shape, generator=rng).cuda().abs() + 1e-3
        affine_3d = torch.linalg.matrix_exp(torch.randn(3, 3, generator=rng))
        affine_3d = torch.cat([affine_3d, 0.01 * torch.randn(3, 1, generator=rng)], dim=1)[None].cuda()
        
        # Test with different interpolation modes
        for mode in ["bilinear", "nearest"]:
            # Test with different padding modes
            for padding in ["zeros", "border"]:
                # Test with both align_corners settings
                for align_corners in [True, False]:
                    # Case 1: Affine-only transformation
                    output = fused_grid_sampler_3d(
                        input, affine=affine_3d, grid=None, mode=mode, padding_mode=padding, 
                        align_corners=align_corners, out_shape=output_shape, is_displacement=False
                    )
                    output_baseline = torch_grid_sampler_3d(
                        input, affine=affine_3d, out_shape=output_shape,
                        mode=mode, padding_mode=padding,
                        align_corners=align_corners
                    )
                    
                    # Check shapes match
                    assert output.shape == output_baseline.shape, f"Shape mismatch for affine-only case"
                    
                    # Check results match
                    rel_error = (torch.abs(output - output_baseline) / (1e-3 + torch.abs(output_baseline))).mean().item()
                    print(f"\nAffine-only case:")
                    print(f"Input shape: {input_shape}, Output shape: {output_shape}")
                    print(f"Mode: {mode}, Padding: {padding}, Align corners: {align_corners}")
                    print(f"Relative error: {rel_error}")
                    assert rel_error < 1e-4, f"Results don't match for {mode} {padding} {align_corners}"
                    
                    # Case 2: Full warp field
                    grid = torch.randn(1, *output_shape, 3, generator=rng).cuda() * 0.01 + F.affine_grid(torch.eye(3, 4)[None].cuda(), (1, 1, *output_shape), align_corners=align_corners)
                    output = fused_grid_sampler_3d(
                        input, affine=None, grid=grid, mode=mode, padding_mode=padding, 
                        align_corners=align_corners, out_shape=output_shape, is_displacement=False
                    )
                    output_baseline = torch_grid_sampler_3d(
                        input, affine=None, grid=grid, mode=mode, padding_mode=padding,
                        align_corners=align_corners, is_displacement=False
                    )
                    
                    # Check shapes match
                    assert output.shape == output_baseline.shape, f"Shape mismatch for full warp case"
                    
                    # Check results match
                    rel_error = (torch.abs(output - output_baseline) / (1e-5 + torch.abs(output_baseline))).mean().item()
                    print(f"\nFull warp case:")
                    print(f"Input shape: {input_shape}, Output shape: {output_shape}")
                    print(f"Mode: {mode}, Padding: {padding}, Align corners: {align_corners}")
                    print(f"Relative error: {rel_error}")
                    assert rel_error < 1e-4, f"Results don't match for full warp case"
                    
                    # Case 3: Displacement field
                    disp = torch.randn(1, *output_shape, 3, generator=rng).cuda() * 0.01
                    output = fused_grid_sampler_3d(
                        input, affine=affine_3d, grid=disp, mode=mode, padding_mode=padding, 
                        align_corners=align_corners, out_shape=output_shape, is_displacement=True
                    )
                    output_baseline = torch_grid_sampler_3d(
                        input, affine=affine_3d, grid=disp, mode=mode, padding_mode=padding,
                        align_corners=align_corners, is_displacement=True
                    )
                    
                    # Check shapes match
                    assert output.shape == output_baseline.shape, f"Shape mismatch for displacement case"
                    
                    # Check results match
                    rel_error = (torch.abs(output - output_baseline) / (1e-5 + torch.abs(output_baseline))).mean().item()
                    print(f"\nDisplacement case:")
                    print(f"Input shape: {input_shape}, Output shape: {output_shape}")
                    print(f"Mode: {mode}, Padding: {padding}, Align corners: {align_corners}")
                    print(f"Relative error: {rel_error}")
                    assert rel_error < 1e-4, f"Results don't match for displacement case"

def test_fused_sampler_performance():
    """Test performance of fused grid sampler against baseline implementation"""
    # Test with different sizes
    sizes = [
        # (input_shape, output_shape)
        ((1, 1, 64, 64, 64), (32, 32, 32)),
        ((1, 1, 128, 128, 128), (64, 64, 64)),
        ((1, 1, 256, 256, 256), (128, 128, 128)),
    ]
    rng = torch.Generator()
    rng.manual_seed(seed)

    for input_shape, output_shape in sizes:
        print(f"\nTesting with input shape {input_shape}, output shape {output_shape}")
        
        # Generate random input and affine matrix
        input = torch.randn(input_shape, generator=rng).cuda().abs() + 1e-3
        affine_3d = torch.linalg.matrix_exp(torch.randn(3, 3, generator=rng))
        affine_3d = torch.cat([affine_3d, torch.zeros(3, 1)], dim=1)[None].cuda()
        
        # Test all three cases
        cases = [
            ("Affine-only", affine_3d, None, False),
            ("Full warp", None, torch.randn(1, *output_shape, 3, generator=rng).cuda() * 0.01, False),
            ("Displacement", affine_3d, torch.randn(1, *output_shape, 3, generator=rng).cuda() * 0.01, True)
        ]
        
        for case_name, aff, grid, is_disp in cases:
            print(f"\nTesting {case_name} case:")
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Measure fused implementation
            start = time()
            output = fused_grid_sampler_3d(
                input, affine=aff, grid=grid, mode="bilinear", padding_mode="zeros", 
                align_corners=False, out_shape=output_shape, is_displacement=is_disp
            )
            torch.cuda.synchronize()
            fused_time = time() - start
            fused_memory = torch.cuda.max_memory_allocated() / 1024**2  
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Measure baseline implementation
            start = time()
            output_baseline = torch_grid_sampler_3d(
                input, affine=aff, grid=grid, mode="bilinear", padding_mode="zeros",
                align_corners=False, out_shape=output_shape, is_displacement=is_disp
            )
            torch.cuda.synchronize()
            baseline_time = time() - start
            baseline_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # Verify results match
            rel_error = (torch.abs(output - output_baseline) / (1e-5 + torch.abs(output_baseline))).mean().item()
            print(f"Relative error: {rel_error}")
            assert rel_error < 1e-4, f"Results don't match for {case_name} case"
            
            # Print performance metrics
            print(f"Time: {fused_time:.4f}s (fused) vs {baseline_time:.4f}s (baseline)")
            print(f"Memory: {fused_memory:.2f}MB (fused) vs {baseline_memory:.2f}MB (baseline)")
            print(f"Speedup: {baseline_time/fused_time:.2f}x")
            print(f"Memory reduction: {(baseline_memory - fused_memory)/baseline_memory*100:.2f}%")

if __name__ == '__main__':
    pytest.main([__file__])
