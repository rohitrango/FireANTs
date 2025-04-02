import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

def baseline_grid_sampler_3d(
    input: torch.Tensor,
    affine_3d: Optional[torch.Tensor] = None,
    grid: Optional[torch.Tensor] = None,
    interpolation_mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    out_shape: tuple = None,
    is_displacement: bool = False
) -> torch.Tensor:
    """
    Baseline implementation of 3D grid sampler that handles:
    1. Affine-only transformation (grid=None)
    2. Full warp field (grid provided, is_displacement=False)
    3. Displacement field (grid provided, is_displacement=True)
    
    Args:
        input: Input tensor of shape [B, C, Z, Y, X]
        affine_3d: Affine matrix of shape [B, 3, 4]
        grid: Optional grid tensor of shape [B, Z, Y, X, 3]
        interpolation_mode: Interpolation mode ("bilinear", "nearest", "bicubic")
        padding_mode: Padding mode ("zeros", "border", "reflection")
        align_corners: Whether to align corners
        out_shape: Output shape tuple (Z, Y, X)
        is_displacement: Whether grid is a displacement field
    
    Returns:
        Sampled tensor of shape [B, C, Z, Y, X]
    """
    B, C, Z, Y, X = input.shape
    
    # Case 1: Affine-only transformation
    if grid is None:
        if affine_3d is None:
            raise ValueError("Either grid or affine_3d must be provided")
        grid = F.affine_grid(affine_3d, (B, C, *out_shape), align_corners=align_corners)
        return F.grid_sample(input, grid, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)
    # Case 2: Full warp field
    if not is_displacement:
        return F.grid_sample(input, grid, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)
    # Case 3: Displacement field
    out_shape = grid.shape[1:-1]
    # Create identity grid if no affine provided
    if affine_3d is None:
        affine_3d = torch.eye(3, 4, device=input.device)[None].expand(B, -1, -1)
    # Create base grid
    base_grid = F.affine_grid(affine_3d, (B, C, *out_shape), align_corners=align_corners)
    # Add displacement
    final_grid = base_grid + grid
    return F.grid_sample(input, final_grid, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)

if __name__ == "__main__":
    # Test cases
    B, C, Z, Y, X = 1, 1, 32, 32, 32
    
    # Test 1: Affine-only
    input = torch.randn(B, C, Z, Y, X).cuda()
    affine = torch.eye(3, 4)[None].cuda()
    out = baseline_grid_sampler_3d(input, affine_3d=affine, out_shape=(16, 16, 16))
    print("Affine-only output shape:", out.shape)
    
    # Test 2: Full warp
    grid = torch.randn(B, 16, 16, 16, 3).cuda()
    out = baseline_grid_sampler_3d(input, grid=grid, is_displacement=False)
    print("Full warp output shape:", out.shape)
    
    # Test 3: Displacement
    disp = torch.randn(B, 16, 16, 16, 3).cuda() * 0.1
    out = baseline_grid_sampler_3d(input, grid=disp, is_displacement=True)
    print("Displacement output shape:", out.shape) 