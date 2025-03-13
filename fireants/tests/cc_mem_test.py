from time import perf_counter as time, sleep
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
from fireants.types import ItemOrList
from fireants.utils.imageutils import separable_filtering

import math
import numpy as np

def fast_lncc(Ii:torch.Tensor, Ji:torch.Tensor, win:Optional[int]=3):
    ndims = len(list(Ii.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    # get dimension of volume
    # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(Ii.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims if win is None else win
    if isinstance(win, int):
        win = [win] * ndims

    # compute filters
    sum_filt = torch.ones([5, 1, *win], device=Ii.device, dtype=Ii.dtype)
    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)
    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)
    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji
    
    all_five = torch.cat((Ii, Ji, I2, J2, IJ),dim=1)
    all_five_conv = conv_fn(all_five, sum_filt, stride=stride, padding=padding, groups=5)
    I_sum, J_sum, I2_sum, J2_sum, IJ_sum = torch.split(all_five_conv, 1, dim=1)
    win_size = np.prod(win)

    cross = IJ_sum - J_sum/win_size*I_sum
    I_var = I2_sum - I_sum/win_size*I_sum
    J_var = J2_sum - J_sum/win_size*J_sum

    cc = cross * cross / (I_var * J_var + 1e-5)
    return -torch.mean(cc)


@torch.compile
def cc_checkpoint_compile(target, pred, kernel, kernel_vol):
    '''
    This function is used to compute the intermediate results of the loss.
    '''
    t2, p2, tp = target * target, pred * pred, target * pred
    kernel, kernel_vol = kernel.to(pred), kernel_vol.to(pred)
    # kernel_nd = self.kernel_nd.to(pred)
    kernels = [kernel] * 3
    kernels_t = kernels_p = kernels
    kernel_vol_t = kernel_vol_p = kernel_vol
    # compute intermediates
    def avg_filter(target, kernels_t, kernel_vol_t):
        t_sum = separable_filtering(target, kernels=kernels_t)
        t_avg = t_sum / kernel_vol_t
        return t_sum, t_avg
    
    t_sum, t_avg = avg_filter(target, kernels_t, kernel_vol_t)
    p_sum, p_avg = avg_filter(pred, kernels_p, kernel_vol_p)

    def sum_filter(target, kernels_t):
        sumfilt = separable_filtering(target, kernels=kernels_t)
        return sumfilt
    
    t2_sum = sum_filter(t2, kernels_t)
    p2_sum = sum_filter(p2, kernels_p)
    tp_sum = sum_filter(tp, kernels_t)

    def cross_filter(tp_sum, p_avg, t_sum):
        return (tp_sum.to(pred) - p_avg * t_sum.to(pred))  # on pred device
    
    cross = cross_filter(tp_sum, p_avg, t_sum)
    smooth_dr = 1e-5
    
    def var_filter(t2sum, tavg, tsum):
        return torch.max(
            t2sum - tavg * tsum, torch.as_tensor(smooth_dr, dtype=t2sum.dtype, device=t2sum.device)
        ).to(pred)
    
    t_var = var_filter(t2_sum, t_avg, t_sum)
    p_var = var_filter(p2_sum, p_avg, p_sum)

    smooth_nr = 0
    def ncc_filter(cross, t_var, p_var):
        ncc: torch.Tensor = (cross * cross + smooth_nr) / ((t_var * p_var) + smooth_dr)
        return ncc

    ncc = ncc_filter(cross, t_var, p_var)
    return ncc

if __name__ == "__main__":
    k = 5

    torch.cuda.memory._record_memory_history()
    size = (2, 1, 256, 256, 256)
    img = torch.randn(*size).cuda()
    img2 = torch.randn(*size).cuda().requires_grad_(True)
    print("Without checkpointing")
    loss_fn = LocalNormalizedCrossCorrelationLoss(reduction="mean", kernel_size=k, checkpointing=False)
    start = time()
    loss = loss_fn(img, img2)
    loss.backward()
    torch.cuda.synchronize()
    end = time()
    print(f"time: {end - start}")
    print("loss.item()", loss.item())
    del loss, loss_fn, img, img2
    torch.cuda.empty_cache()
    sleep(1)

    img = torch.randn(*size).cuda()
    img2 = torch.randn(*size).cuda().requires_grad_(True)
    print("\nWith checkpointing")
    loss_fn = LocalNormalizedCrossCorrelationLoss(reduction="mean", kernel_size=k, checkpointing=True)
    start = time()
    loss = loss_fn(img, img2)
    loss.backward()
    torch.cuda.synchronize()
    end = time()
    print(f"time: {end - start}")
    print("loss.item()", loss.item())
    kernel = loss_fn.kernel
    kernel_vol = loss_fn.kernel_vol
    del loss, loss_fn, img, img2
    torch.cuda.empty_cache()
    sleep(1)

    img = torch.randn(*size).cuda()
    img2 = torch.randn(*size).cuda().requires_grad_(True)
    print("\nWith torch compile")
    # warmup
    loss = cc_checkpoint_compile(img, img2, kernel, kernel_vol).mean()
    start = time()
    loss = cc_checkpoint_compile(img, img2, kernel, kernel_vol).mean()
    loss.backward()
    torch.cuda.synchronize()
    end = time()
    print(f"time: {end - start}")
    print("loss.item()", loss.item())
    # third one is a torch compile version
    del loss, img, img2
    torch.cuda.empty_cache()
    sleep(1)

    print("\nWith fast lncc")
    img = torch.randn(*size).cuda()
    img2 = torch.randn(*size).cuda().requires_grad_(True)
    start = time()
    loss = fast_lncc(img, img2, win=k)
    loss.backward()
    torch.cuda.synchronize()
    end = time()
    print(f"time: {end - start}")
    print("loss.item()", loss.item())
    del loss, img, img2
    torch.cuda.empty_cache()
    sleep(1)

    fast_lncc_compile = torch.compile(fast_lncc)
    print("\nWith torch compile + fast lncc")
    img = torch.randn(*size).cuda()
    img2 = torch.randn(*size).cuda().requires_grad_(True)
    loss = fast_lncc_compile(img, img2, win=3) # warmup
    start = time()
    loss = fast_lncc_compile(img, img2, win=k)
    loss.backward()
    torch.cuda.synchronize()
    end = time()
    print(f"time: {end - start}")
    print("loss.item()", loss.item())

    sleep(1)
    torch.cuda.memory._dump_snapshot("cc_mem_test.pkl")
