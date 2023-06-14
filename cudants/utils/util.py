from typing import List
from time import perf_counter
from contextlib import contextmanager
import torch

from torch.nn import functional as F
from cudants.losses.cc import gaussian_1d, separable_filtering

class catchtime:
    ''' class to naively profile pieces of code '''
    def __init__(self, str=None) -> None:
        self.str = str

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'{self.str}: Time: {self.time:.3f} seconds'
        print(self.readout)


def _assert_check_scales_decreasing(scales: List[int]):
    ''' Check if the list of scales is in decreasing order '''
    for i in range(len(scales)-1):
        if scales[i] <= scales[i+1]:
            raise ValueError("Scales must be in decreasing order")


def grad_smoothing_hook(grad: torch.Tensor, gaussians: List[torch.Tensor]):
    ''' this backward hook will smooth out the gradient using the gaussians 
    has to be called with a partial function
    '''
    # grad is of shape [B, H, W, D, dims]
    if len(grad.shape) == 5:
        permute_vtoimg = (0, 4, 1, 2, 3)
        permute_imgtov = (0, 2, 3, 4, 1)
    elif len(grad.shape) == 4:
        permute_vtoimg = (0, 3, 1, 2)
        permute_imgtov = (0, 2, 3, 1)
    return separable_filtering(grad.permute(*permute_vtoimg), gaussians).permute(*permute_imgtov)


def compose_warp(warp1: torch.Tensor, warp2: torch.Tensor, grid: torch.Tensor):
    '''
    warp1 and warp2 are displacement maps u(x) and v(x) of size [N, H, W, D, dims]

    phi1(x) = x + u(x)
    phi2(x) = x + v(x)
    (phi1 \circ phi2 )(x) = phi1(x + v(x)) = x + v(x) + u(x + v(x))
    '''
    if len(warp1.shape) == 5:
        permute_vtoimg = (0, 4, 1, 2, 3)
        permute_imgtov = (0, 2, 3, 4, 1)
    elif len(warp1.shape) == 4:
        permute_vtoimg = (0, 3, 1, 2)
        permute_imgtov = (0, 2, 3, 1)
    # compute u(x + v(x)) 
    warp12 = F.grid_sample(warp1.permute(*permute_vtoimg), grid + warp2, mode='bilinear', align_corners=True).permute(*permute_imgtov)
    return warp2 + warp12
