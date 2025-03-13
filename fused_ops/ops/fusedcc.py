'''
Fused Cross correlation
'''
from time import time, sleep
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
from fireants.types import ItemOrList
import fireants_fused_ops as ffo
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss

reduction_table = {
    'none': ffo.Reduction.NONE,
    'sum': ffo.Reduction.SUM,
    'mean': ffo.Reduction.MEAN
}

class FusedNCC3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_img, target_img, kernel_size, nr, dr, reduction):
        reduction = reduction_table[reduction.lower()]
        B, C, H, W, D = input_img.shape
        interm = torch.zeros(B, 5 * C, H, W, D, device=input_img.device)
        interm[:, :C, :, :, :].data.copy_(input_img)
        interm[:, C:2*C, :, :, :].data.copy_(target_img)
        interm[:, 2*C:3*C, :, :, :].data.copy_(input_img).mul_(input_img)
        interm[:, 3*C:4*C, :, :, :].data.copy_(target_img).mul_(target_img)
        interm[:, 4*C:, :, :, :].data.copy_(input_img).mul_(target_img)  # [B, 5C, H, W, D]
        print(interm.device)
        # compute kernel 
        kernel_vol = kernel_size ** 3
        avg_filt = torch.ones(5*C, 1, kernel_size, kernel_size, kernel_size, device=input_img.device) / kernel_vol
        padding = (kernel_size - 1) // 2
        interm = F.conv3d(interm, avg_filt, padding=padding, stride=1, groups=interm.shape[1])
        out = ffo.cc3d_fwd_interm_v1(interm, int(kernel_vol), reduction, nr, dr)
        ctx.save_for_backward(interm, input_img, target_img, kernel_size, nr, dr, reduction)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        interm, input_img, target_img, kernel_size, nr, dr, reduction = ctx.saved_tensors
        # B, C, H, W, D = input_img.shape
        # grad_input_img = torch.zeros(B, C, H, W, D, device=input_img.device)
        # grad_target_img = torch.zeros(B, C, H, W, D, device=input_img.device)
        # ffo.cc3d_bwd_interm_v1(interm, input_img, target_img, grad_output, grad_input_img, grad_target_img, kernel_size, nr, dr, reduction)
        return None, None, None, None, None, None, None

class FusedLocalNormalizedCrossCorrelationLoss(nn.Module):
    """
    Local squared zero-normalized cross-correlation.
    The loss is based on a moving kernel/window over the y_true/y_pred,
    within the window the square of zncc is calculated.
    The kernel can be a rectangular / triangular / gaussian window.
    The final loss is the averaged loss over all windows.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: int = 3,
        reduction: str = "mean",
        smooth_nr: float = 0,
        smooth_dr: float = 1e-5,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel spatial size, must be odd.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
        """
        super().__init__()
        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")
        self.reduction = reduction

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        # _kernel = look_up_option(kernel_type, kernel_dict)
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if pred.ndim - 2 != self.ndim:
            raise ValueError(f"expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}")
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")
        return FusedNCC3d.apply(pred, target, self.kernel_size, self.smooth_nr, self.smooth_dr, self.reduction)

if __name__ == '__main__':
    for i in range(4, 10):
        N = 2 ** i
        img1 = torch.rand(1, 1, N, N, N).cuda()
        img2 = torch.rand(1, 1, N, N, N).cuda()
        
        # Calculate input tensor memory
        input_memory = (img1.element_size() * img1.nelement() + 
                       img2.element_size() * img2.nelement()) / 1024**2  # MB
        
        loss = FusedLocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction='mean').cuda()
        loss_baseline = LocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction='mean').cuda()
        total = 0
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        
        start = time()
        out = loss(img1, img2)
        torch.cuda.synchronize()
        out_time = time() - start
        fused_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory  # Subtract input memory

        # Reset memory stats again
        torch.cuda.reset_peak_memory_stats()
        
        start = time()
        out_baseline = loss_baseline(img1, img2)
        torch.cuda.synchronize()
        out_baseline_time = time() - start
        baseline_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory  # Subtract input memory

        out1 = -out.item()
        out2 = out_baseline.item()
        rel_err = abs(out1 - out2) / abs(out2)
        print(f"N: {N}, rel_err: {rel_err}")
        print(f"out_time: {out_time:.4f}, out_baseline_time: {out_baseline_time:.4f}, speed up: {out_baseline_time / out_time:.2f}x")
        print(f"Input memory: {input_memory:.2f}MB")
        print(f"Fused memory (excluding input): {fused_memory:.2f}MB, Baseline memory (excluding input): {baseline_memory:.2f}MB")
        print(f"Memory reduction: {baseline_memory/fused_memory:.4f}x")
        print()

