# Copyright (c) 2026 Rohit Jena. All rights reserved.
#
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE

"""
Python frontend for fused generic-label grid sampler (2D and 3D).
Samples a label/one-hot map with affine and/or grid; returns argmax labels
and optional interpolated weights (for differentiable loss on weights).
"""
import logging
from typing import Optional, Tuple, Union

import torch

from fireants.interpolator.grid_sample import (
    get_min_coords2d,
    get_max_coords2d,
    get_min_coords3d,
    get_max_coords3d,
)

import fireants_fused_ops as ffo

logger = logging.getLogger(__name__)

GRID_SAMPLE_PADDING_MODES = {
    "zeros": 0,
    "border": 1,
    "reflection": 2,
}


class FusedGridSampler2dGenericLabel(torch.autograd.Function):
    """Autograd Function for 2D fused generic-label grid sampler."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        affine: Optional[torch.Tensor],
        grid: Optional[torch.Tensor],
        grid_affine: Optional[torch.Tensor],
        output_labels: Optional[torch.Tensor],
        output_weights: Optional[torch.Tensor],
        out_H: int,
        out_W: int,
        min_coords: Tuple[float, float],
        max_coords: Tuple[float, float],
        is_displacement: bool,
        padding_mode: int,
        align_corners: bool,
        return_weight: bool,
        background_label: Optional[float],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input: [B, C, H, W] one-hot or label weights
        affine: [B, 2, 3] or None
        grid: [B, H, W, 2] or None
        Returns (output_labels [B, 1, H, W], output_weights [B, 1, H, W] or None).
        """
        out_labels, out_weights = ffo.fused_grid_sampler_2d_generic_label_forward(
            input,
            affine,
            grid,
            grid_affine,
            output_labels,
            output_weights,
            out_H,
            out_W,
            min_coords[0],
            min_coords[1],
            max_coords[0],
            max_coords[1],
            is_displacement,
            padding_mode,
            align_corners,
            return_weight,
            background_label,
        )
        ctx.save_for_backward(input, affine, grid, grid_affine, out_labels)
        ctx.out_shape = (out_H, out_W)
        ctx.min_coords = min_coords
        ctx.max_coords = max_coords
        ctx.is_displacement = is_displacement
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        ctx.return_weight = return_weight
        return out_labels, out_weights

    @staticmethod
    def backward(ctx, grad_output_labels, grad_output_weights):
        input, affine, grid, grid_affine, output_labels = ctx.saved_tensors
        out_H, out_W = ctx.out_shape
        min_coords = ctx.min_coords
        max_coords = ctx.max_coords
        is_displacement = ctx.is_displacement
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        return_weight = ctx.return_weight

        # if we don't need to return weights, we can skip the backward pass
        if not return_weight:
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

        affine_requires_grad = affine is not None and affine.requires_grad
        grid_requires_grad = grid is not None and grid.requires_grad
        grad_affine = torch.zeros_like(affine) if (affine is not None and affine_requires_grad) else None
        grad_grid = torch.zeros_like(grid) if (grid is not None and grid_requires_grad) else None

        # Ensure C++ receives a contiguous tensor; use zeros when no upstream grad (so kernel still runs)
        assert grad_output_weights is not None
        grad_weight_to_pass = grad_output_weights.contiguous()

        ffo.fused_grid_sampler_2d_generic_label_backward(
            input,
            affine,
            grid,
            grid_affine,
            output_labels,
            grad_weight_to_pass,
            grad_affine,
            grad_grid,
            out_H,
            out_W,
            min_coords[0],
            min_coords[1],
            max_coords[0],
            max_coords[1],
            is_displacement,
            padding_mode,
            align_corners,
            return_weight,
        )
        return (
            None,  # input
            grad_affine,
            grad_grid,
            None,  # grid_affine
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FusedGridSampler3dGenericLabel(torch.autograd.Function):
    """Autograd Function for 3D fused generic-label grid sampler."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        affine: Optional[torch.Tensor],
        grid: Optional[torch.Tensor],
        grid_affine: Optional[torch.Tensor],
        output_labels: Optional[torch.Tensor],
        output_weights: Optional[torch.Tensor],
        out_D: int,
        out_H: int,
        out_W: int,
        min_coords: Tuple[float, float, float],
        max_coords: Tuple[float, float, float],
        is_displacement: bool,
        padding_mode: int,
        align_corners: bool,
        return_weight: bool,
        background_label: Optional[float],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input: [B, C, D, H, W]
        affine: [B, 3, 4] or None
        grid: [B, D, H, W, 3] or None
        Returns (output_labels [B, D, H, W], output_weights [B, C, D, H, W] or None).
        """
        out_labels, out_weights = ffo.fused_grid_sampler_3d_generic_label_forward(
            input,
            affine,
            grid,
            grid_affine,
            output_labels,
            output_weights,
            out_D,
            out_H,
            out_W,
            min_coords[0],
            min_coords[1],
            min_coords[2],
            max_coords[0],
            max_coords[1],
            max_coords[2],
            is_displacement,
            padding_mode,
            align_corners,
            return_weight,
            background_label,
        )
        ctx.save_for_backward(input, affine, grid, grid_affine, out_labels)
        ctx.out_shape = (out_D, out_H, out_W)
        ctx.min_coords = min_coords
        ctx.max_coords = max_coords
        ctx.is_displacement = is_displacement
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        ctx.return_weight = return_weight
        return out_labels, out_weights

    @staticmethod
    def backward(ctx, grad_output_labels, grad_output_weights):
        input, affine, grid, grid_affine, output_labels = ctx.saved_tensors
        out_D, out_H, out_W = ctx.out_shape
        min_coords = ctx.min_coords
        max_coords = ctx.max_coords
        is_displacement = ctx.is_displacement
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        return_weight = ctx.return_weight

        grad_affine = (
            torch.zeros_like(affine)
            if (affine is not None and affine.requires_grad)
            else None
        )
        grad_grid = (
            torch.zeros_like(grid)
            if (grid is not None and grid.requires_grad)
            else None
        )

        assert grad_output_weights is not None
        grad_weight_to_pass = grad_output_weights.contiguous()

        ffo.fused_grid_sampler_3d_generic_label_backward(
            input,
            affine,
            grid,
            grid_affine,
            output_labels,
            grad_weight_to_pass,
            grad_affine,
            grad_grid,
            out_D,
            out_H,
            out_W,
            min_coords[0],
            min_coords[1],
            min_coords[2],
            max_coords[0],
            max_coords[1],
            max_coords[2],
            is_displacement,
            padding_mode,
            align_corners,
            return_weight,
        )
        return (
            None,
            grad_affine,
            grad_grid,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_grid_sampler_2d_generic_label(
    input: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    grid: Optional[torch.Tensor] = None,
    grid_affine: Optional[torch.Tensor] = None,
    padding_mode: str = "zeros",
    align_corners: bool = True,
    min_coords: Optional[Tuple[float, float]] = None,
    max_coords: Optional[Tuple[float, float]] = None,
    out_shape: Optional[Tuple[int, int]] = None,
    is_displacement: bool = True,
    return_probs: bool = False,
    background_label: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    2D fused generic-label grid sampler.
    Samples a label/one-hot map [B, C, H, W] with optional affine and/or grid;
    returns integer labels (argmax over interpolated weights) and optionally
    the interpolated weight map for differentiable loss.

    Args:
        input: One-hot or label weights [B, C, H, W].
        affine: [B, 2, 3] or None.
        grid: [B, H, W, 2] or None.
        grid_affine: [B, 2, 2] or None (pre-grid affine for grid).
        padding_mode: "zeros", "border", or "reflection".
        align_corners: Whether to align corners.
        min_coords: (xmin, ymin) or None (computed from spatial size).
        max_coords: (xmax, ymax) or None.
        out_shape: (H, W) when grid is None; otherwise inferred from grid.
        is_displacement: Whether grid is a displacement field.
        return_probs: If True, return (labels, weights); else labels only.
        background_label: Optional background label value.

    Returns:
        out_labels [B, 1, H, W], or (out_labels, out_weights) when return_probs=True.
    """
    if grid is None:
        if out_shape is None:
            _, _, H, W = input.shape
            out_shape = (H, W)
            logger.warning("out_shape not provided, using input spatial shape")
        out_H, out_W = out_shape
    else:
        out_H, out_W = grid.shape[1:-1]

    if min_coords is None:
        min_coords = get_min_coords2d(out_H, out_W, align_corners)
    if max_coords is None:
        max_coords = get_max_coords2d(out_H, out_W, align_corners)

    assert input.is_contiguous(), "input must be contiguous"
    assert affine is None or affine.is_contiguous(), "affine must be contiguous"
    assert grid is None or grid.is_contiguous(), "grid must be contiguous"

    padding_mode_int = GRID_SAMPLE_PADDING_MODES.get(
        padding_mode, GRID_SAMPLE_PADDING_MODES["zeros"]
    )

    out_labels, out_weights = FusedGridSampler2dGenericLabel.apply(
        input,
        affine,
        grid,
        grid_affine,
        None,  # output_labels
        None,  # output_weights
        out_H,
        out_W,
        min_coords,
        max_coords,
        is_displacement,
        padding_mode_int,
        align_corners,
        return_probs,
        background_label,
    )
    if return_probs:
        return out_labels, out_weights
    return out_labels


def fused_grid_sampler_3d_generic_label(
    input: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    grid: Optional[torch.Tensor] = None,
    grid_affine: Optional[torch.Tensor] = None,
    padding_mode: str = "zeros",
    align_corners: bool = True,
    min_coords: Optional[Tuple[float, float, float]] = None,
    max_coords: Optional[Tuple[float, float, float]] = None,
    out_shape: Optional[Tuple[int, int, int]] = None,
    is_displacement: bool = True,
    return_probs: bool = False,
    background_label: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    3D fused generic-label grid sampler.
    Samples a label/one-hot map [B, C, D, H, W] with optional affine and/or grid;
    returns integer labels and optionally the interpolated weight map.

    Args:
        input: [B, C, D, H, W].
        affine: [B, 3, 4] or None.
        grid: [B, D, H, W, 3] or None.
        grid_affine: [B, 3, 3] or None.
        padding_mode: "zeros", "border", or "reflection".
        align_corners: Whether to align corners.
        min_coords: (xmin, ymin, zmin) or None.
        max_coords: (xmax, ymax, zmax) or None.
        out_shape: (D, H, W) when grid is None.
        is_displacement: Whether grid is displacement.
        return_probs: If True, return (labels, weights); else labels only.
        background_label: Optional background label.

    Returns:
        out_labels [B, D, H, W], or (out_labels, out_weights) when return_probs=True.
    """
    if grid is None:
        if out_shape is None:
            _, _, D, H, W = input.shape
            out_shape = (D, H, W)
            logger.warning("out_shape not provided, using input spatial shape")
        out_D, out_H, out_W = out_shape
    else:
        out_D, out_H, out_W = grid.shape[1:-1]

    if min_coords is None:
        min_coords = get_min_coords3d(out_D, out_H, out_W, align_corners)
    if max_coords is None:
        max_coords = get_max_coords3d(out_D, out_H, out_W, align_corners)

    assert input.is_contiguous(), "input must be contiguous"
    assert affine is None or affine.is_contiguous(), "affine must be contiguous"
    assert grid is None or grid.is_contiguous(), "grid must be contiguous"

    padding_mode_int = GRID_SAMPLE_PADDING_MODES.get(
        padding_mode, GRID_SAMPLE_PADDING_MODES["zeros"]
    )

    out_labels, out_weights = FusedGridSampler3dGenericLabel.apply(
        input,
        affine,
        grid,
        grid_affine,
        None,
        None,
        out_D,
        out_H,
        out_W,
        min_coords,
        max_coords,
        is_displacement,
        padding_mode_int,
        align_corners,
        return_probs,
        background_label,
    )
    if return_probs:
        return out_labels, out_weights
    return out_labels
