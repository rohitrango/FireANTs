'''
Fused Cross correlation
'''
from time import time, sleep
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F
import sys
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
from fireants.types import ItemOrList
import fireants_fused_ops as ffo
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
from fireants.tests.cc_mem_test import fast_lncc

reduction_table = {
    'none': ffo.Reduction.NONE,
    'sum': ffo.Reduction.SUM,
    'mean': ffo.Reduction.MEAN
}

class FusedNCC3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_img, target_img, kernel_size, nr, dr, reduction, use_ants_gradient):
        reduction = reduction_table[reduction.lower()]
        B, C, H, W, D = input_img.shape
        interm = torch.zeros(B, 5 * C, H, W, D, device=input_img.device, dtype=input_img.dtype)
        # interm[:, :C, :, :, :] = input_img
        # interm[:, C:2*C, :, :, :] = target_img
        # interm[:, 2*C:3*C, :, :, :] = input_img * input_img
        # interm[:, 3*C:4*C, :, :, :] = target_img * target_img
        # interm[:, 4*C:, :, :, :] = input_img * target_img  # [B, 5C, H, W, D]
        ffo.create_intermediates(input_img, target_img, interm)
        # torch.cat uses more memory
        # compute kernel 
        kernel_vol = kernel_size ** 3
        avg_filt = torch.ones(5*C, 1, kernel_size, kernel_size, kernel_size, device=input_img.device, dtype=input_img.dtype) / kernel_vol
        padding = (kernel_size - 1) // 2
        interm = F.conv3d(interm, avg_filt, padding=padding, stride=1, groups=interm.shape[1])
        out = ffo.cc3d_fwd_interm_v1(interm, int(kernel_vol), reduction, nr, dr)
        ctx.save_for_backward(interm, input_img, target_img, out, avg_filt * kernel_vol)
        ctx.kernel_size = kernel_size
        ctx.nr = nr
        ctx.dr = dr 
        ctx.reduction = reduction
        ctx.use_ants_gradient = use_ants_gradient
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved tensors
        interm, input_img, target_img, out, avg_filt = ctx.saved_tensors
        kernel_size, nr, dr, reduction, use_ants_gradient = ctx.kernel_size, ctx.nr, ctx.dr, ctx.reduction, ctx.use_ants_gradient
        B, C, H, W, D = input_img.shape
        # initialize gradients
        grad_input_img = None
        grad_target_img = None
        # if backward is called, first tensor will always require grad
        # second may or may not require grad
        if input_img.requires_grad:
            grad_input_img = torch.zeros(B, C, H, W, D, device=input_img.device, dtype=input_img.dtype)
        if target_img.requires_grad:
            grad_target_img = torch.zeros(B, C, H, W, D, device=input_img.device, dtype=input_img.dtype)
        # compute gradients
        # compute correct gradients (maybe slightly slower)
        ffo.cc3d_bwd_modify_interm_v1(interm, input_img, target_img, grad_output, grad_input_img, grad_target_img, kernel_size, nr, dr, reduction)
        # convolve with average filter depending on whether grad_target_img is None
        # if using ants_gradient, the convolution is skipped (ignore interactions from other neighboring pixels)
        if not use_ants_gradient:
            padding = (kernel_size - 1) // 2
            if grad_target_img is not None:
                interm = F.conv3d(interm, avg_filt, padding=padding, stride=1, groups=interm.shape[1])
            else:
                avg_filt = avg_filt[:3*C]
                interm[:, :3*C, :, :, :] = F.conv3d(interm[:, :3*C, :, :, :], avg_filt, padding=padding, stride=1, groups=3*C)
        # solve for grad_input_img and grad_target_img
        ffo.cc3d_bwd_compute_grads(interm, input_img, target_img, grad_input_img, grad_target_img)

        # skip reduction for now
        # if grad_input_img is not None and reduction == ffo.Reduction.MEAN:
        #     grad_input_img = grad_input_img / (H * W * D)
        # if grad_target_img is not None and reduction == ffo.Reduction.MEAN:
        #     grad_target_img = grad_target_img / (H * W * D)

        return grad_input_img, grad_target_img, None, None, None, None, None


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
        use_ants_gradient: bool = False,
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
        self.use_ants_gradient = use_ants_gradient

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
        # check if both pred and target require grad
        pgrad, tgrad = pred.requires_grad, target.requires_grad
        # if both or neither require grad, dont shuffle the order 
        if (pgrad and tgrad) or (not pgrad and not tgrad):
            return FusedNCC3d.apply(pred, target, self.kernel_size, self.smooth_nr, self.smooth_dr, self.reduction, self.use_ants_gradient)
        else:
            # if only pred requires grad, swap pred and target
            if pgrad:
                return FusedNCC3d.apply(pred, target, self.kernel_size, self.smooth_nr, self.smooth_dr, self.reduction, self.use_ants_gradient)
            else:
                return FusedNCC3d.apply(target, pred, self.kernel_size, self.smooth_nr, self.smooth_dr, self.reduction, self.use_ants_gradient)


def test_fused_cc_fwd_and_mem():
    for i in range(6, 10):
    # torch.cuda.memory._record_memory_history()
    # for i in range(6, 7):
        N = 2 ** i
        img1 = torch.rand(1, 1, N, N, N).cuda()
        img2 = torch.rand(1, 1, N, N, N).cuda().requires_grad_(True)
        
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

        # baseline 2
        torch.cuda.reset_peak_memory_stats()
        start = time()
        out_baseline2 = fast_lncc(img1, img2, 3)
        torch.cuda.synchronize()
        # out_baseline2.backward()
        # torch.cuda.synchronize()
        out_baseline2_time = time() - start
        baseline2_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory  # Subtract input memory

        out1 = -out.item()
        out2 = out_baseline.item()
        out3 = out_baseline2.item()
        rel_err = abs(out1 - out2) / abs(out2)
        rel_err2 = abs(out1 - out3) / abs(out3)
        print(f"N: {N}, rel_err: {rel_err}, rel_err2: {rel_err2}")
        print(f"out: {out1}, out_baseline: {out2}, out_baseline2: {out3}")
        print(f"out_time: {out_time:.4f}, out_baseline_time: {out_baseline_time:.4f}, speed up: {out_baseline_time / out_time:.2f}x")
        print(f"out_time2: {out_baseline2_time:.4f}, out_baseline_time2: {out_baseline2_time:.4f}, speed up: {out_baseline2_time / out_time:.2f}x")
        print(f"Input memory: {input_memory:.2f}MB")
        print(f"Fused memory (excluding input): {fused_memory:.2f}MB, Baseline memory (excluding input): {baseline_memory:.2f}MB, Baseline2 memory (excluding input): {baseline2_memory:.2f}MB")
        print(f"Memory reduction: {(baseline_memory - fused_memory)/baseline_memory:.4f}%, {(baseline2_memory - fused_memory)/baseline2_memory:.4f}%")
        print()

def test_fused_cc_bwd_and_mem():
    ''' check backward memory usage '''
    for i in range(6, 10):
        N = 2 ** i
        eps = 1e-1
        img1 = (torch.rand(1, 1, N, N, N) + eps).cuda().requires_grad_(True)
        img2 = (torch.rand(1, 1, N, N, N) + eps).cuda().requires_grad_(True) 
        
        # Calculate input tensor memory
        input_memory = (img1.element_size() * img1.nelement() + 
                       img2.element_size() * img2.nelement()) / 1024**2  # MB
        
        loss = FusedLocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction='mean', use_ants_gradient=False).cuda()
        loss_baseline = LocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction='mean').cuda()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        
        t0 = time()
        out = -loss(img1, img2)
        torch.cuda.synchronize()
        t1 = time()
        out.backward()
        torch.cuda.synchronize()
        t2 = time()
        fwd_time_ours = t1 - t0
        bwd_time_ours = t2 - t1
        grad_ours = img2.grad / N**3
        fused_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory  # Subtract input memory

        img2.grad = None
        # Reset memory stats again
        torch.cuda.reset_peak_memory_stats()
        
        t0 = time()
        out_baseline = loss_baseline(img1, img2)
        torch.cuda.synchronize()
        t1 = time()
        out_baseline.backward()
        torch.cuda.synchronize()
        t2 = time()
        fwd_time_baseline = t1 - t0
        bwd_time_baseline = t2 - t1
        grad_baseline = img2.grad
        baseline_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory  # Subtract input memory

        # baseline 2
        img2.grad = None
        torch.cuda.reset_peak_memory_stats()
        t0 = time()
        out_baseline2 = fast_lncc(img1, img2, 3)
        torch.cuda.synchronize()
        t1 = time()
        out_baseline2.backward()
        torch.cuda.synchronize()
        t2 = time()
        fwd_time_baseline2 = t1 - t0
        bwd_time_baseline2 = t2 - t1
        grad_baseline2 = img2.grad
        baseline2_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory  # Subtract input memory

        out = out.item()
        out_baseline = out_baseline.item()
        out_baseline2 = out_baseline2.item()
        rel_err = abs(out - out_baseline) / abs(out_baseline)
        rel_err2 = abs(out - out_baseline2) / abs(out_baseline2)
        rel_grad_err = ((grad_ours - grad_baseline).abs() / (1e-7 + grad_baseline.abs())).mean()
        rel_grad_err2 = ((grad_ours - grad_baseline2).abs() / (1e-7 + grad_baseline2.abs())).mean()
        rel_grad_baselines = ((grad_baseline - grad_baseline2).abs() / (1e-7 + grad_baseline2.abs())).mean()

        print(f"N: {N}, rel_err: {rel_err}, rel_err2: {rel_err2}")
        print(f"rel_grad_err: {rel_grad_err}, rel_grad_err2: {rel_grad_err2}, rel_grad_baselines: {rel_grad_baselines}")
        print()
        print(f"out: {out}, out_baseline: {out_baseline}, out_baseline2: {out_baseline2}")

        # compare fwd_time of ours and baselines, and speedup
        print(f"fwd_time_ours: {fwd_time_ours:.4f}, fwd_time_baseline: {fwd_time_baseline:.4f}, speedup: {fwd_time_baseline / fwd_time_ours:.2f}x")
        print(f"fwd_time_ours: {fwd_time_ours:.4f}, fwd_time_baseline2: {fwd_time_baseline2:.4f}, speedup: {fwd_time_baseline2 / fwd_time_ours:.2f}x")
        print()

        # same for backward time
        print(f"bwd_time_ours: {bwd_time_ours:.4f}, bwd_time_baseline: {bwd_time_baseline:.4f}, speedup: {bwd_time_baseline / bwd_time_ours:.2f}x")
        print(f"bwd_time_ours: {bwd_time_ours:.4f}, bwd_time_baseline2: {bwd_time_baseline2:.4f}, speedup: {bwd_time_baseline2 / bwd_time_ours:.2f}x")
        print()

        print(f"Input memory: {input_memory:.2f}MB")
        print(f"Fused memory (excluding input): {fused_memory:.2f}MB, Baseline memory (excluding input): {baseline_memory:.2f}MB, Baseline2 memory (excluding input): {baseline2_memory:.2f}MB")
        print(f"Memory reduction: {100*(baseline_memory - fused_memory)/baseline_memory:.4f}%, {100*(baseline2_memory - fused_memory)/baseline2_memory:.4f}%")
        print("--------------------------------")
    # torch.cuda.memory._dump_snapshot("fused_cc_bwd_mem.pkl")

if __name__ == '__main__':
    # check backward
    # N = 32
    # img1 = torch.randn(1, 1, N, N, N).cuda()
    # img2 = torch.randn(1, 1, N, N, N).cuda().requires_grad_(True)
    # # tensor = img2 
    # # print(tensor.shape)
    # # print(tensor.is_contiguous())
    # # print(tensor.storage().data_ptr())  # Check memory address
    # # print(tensor.is_contiguous(memory_format=torch.contiguous_format))
    # # print("--------------------------------")

    # loss = FusedLocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction='mean').cuda()
    # out = loss(img1, img2)
    # print(out)
    # out.backward()
    # print(img2.grad)
    # sys.exit()
    # test_fused_cc_fwd_and_mem()
    test_fused_cc_bwd_and_mem()
