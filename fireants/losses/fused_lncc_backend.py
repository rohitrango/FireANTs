"""Optional LNCC backend backed by the standalone ``fused_lncc`` CUDA kernel.

    pip install fused_lncc --no-build-isolation   # needs PyTorch + a CUDA toolchain; see its README
    reg = GreedyRegistration(..., loss_type='fused_lncc')   # falls back to 'cc' if not installed
"""
import warnings
from typing import List, Optional, Union

import torch
from torch import nn

try:
    from fused_lncc import fused_lncc_loss
except ImportError as e:  # the loss_type dispatcher catches this and falls back to 'cc'
    raise ImportError("fused_lncc is not installed; `pip install fused_lncc`") from e

_SUPPORTED_KERNEL_SIZES = (3, 5, 7, 9)


class _ScaleGrad(torch.autograd.Function):
    """Identity forward; scales the gradient by a constant on the backward pass."""

    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.factor, None


class FusedLNCCLoss(nn.Module):
    """Rectangular LNCC via the fused_lncc kernel, matching ``FusedLocalNormalizedCrossCorrelationLoss``.

    Faster and lighter than the cuDNN path, but narrower: 3D, rectangular window, odd kernel in
    {3, 5, 7, 9}, mean reduction, gradient to ``pred`` only, single GPU (no grid-parallel sharding).
    Other configurations
    (gaussian, masking, sum/none reduction, symmetric/SyN gradients) raise; use ``loss_type='fusedcc'``.

    The gradient is exact (cosine 1.0 vs MONAI), unlike fusedcc's default ``use_ants_gradient=True``
    ANTs approximation. The denominator regularizer differs from fusedcc only in near-constant
    windows, which shifts the loss value there but not the gradient or the registration result.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: Union[int, List[int]] = 3,
        reduction: str = "mean",
        smooth_dr: float = 1e-5,
        kernel_type: str = "rectangular",
        masked: bool = False,
        use_ants_gradient: Optional[bool] = None,
        **kwargs,  # absorb fusedcc args this kernel does not use (smooth_nr, use_separable, ...)
    ) -> None:
        super().__init__()
        if use_ants_gradient:
            warnings.warn("fused_lncc always uses the exact gradient; use_ants_gradient=True is ignored.")
        if spatial_dims != 3:
            raise NotImplementedError(f"fused_lncc supports 3D only (got {spatial_dims}); use loss_type='fusedcc'")
        if kernel_type != "rectangular":
            raise NotImplementedError(f"fused_lncc supports rectangular only (got '{kernel_type}'); use loss_type='fusedcc'")
        if reduction != "mean":
            raise NotImplementedError(f"fused_lncc supports reduction='mean' only (got '{reduction}'); use loss_type='fusedcc'")
        if masked:
            raise NotImplementedError("fused_lncc does not support masked mode; use loss_type='fusedcc'")

        # kernel_size may be a single int or one per multi-resolution scale.
        self.kernel_size_list = list(kernel_size) if isinstance(kernel_size, (list, tuple)) else None
        for k in self.kernel_size_list or [kernel_size]:
            if k not in _SUPPORTED_KERNEL_SIZES:
                raise ValueError(f"fused_lncc supports kernel_size in {_SUPPORTED_KERNEL_SIZES}, got {k}")
        self.kernel_size = self.kernel_size_list[0] if self.kernel_size_list else kernel_size
        self.smooth_dr = float(smooth_dr)

    # multi-scale hooks called by the registration framework
    def set_scales(self, scales) -> None:
        self.scales = scales
        if self.kernel_size_list:
            assert len(self.kernel_size_list) == len(scales), "kernel_size list must match the number of scales"

    def set_iterations(self, iterations) -> None:
        self.iterations = iterations

    def set_current_scale_and_iterations(self, scale, iters) -> None:
        if self.kernel_size_list:
            self.kernel_size = self.kernel_size_list[self.scales.index(scale)]

    def get_image_padding(self) -> int:
        return (self.kernel_size - 1) // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
        # The kernel differentiates w.r.t. `pred` only; route the grad-requiring image there.
        if pred.requires_grad and target.requires_grad:
            raise NotImplementedError("fused_lncc has a pred-only gradient; symmetric/SyN is unsupported, use loss_type='fusedcc'")
        if target.requires_grad and not pred.requires_grad:
            pred, target = target, pred
        # Kernel returns 1 - mean(ncc); -1.0 matches fusedcc's -mean(ncc) sign. fusedcc's gradient is
        # numel/kernel_vol larger than the true mean gradient, so rescale to match it (same lr converges alike).
        loss = fused_lncc_loss(pred, target, self.kernel_size, self.smooth_dr)
        return _ScaleGrad.apply(loss, pred.numel() / self.kernel_size ** 3) - 1.0
