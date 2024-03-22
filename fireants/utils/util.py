from typing import List
from time import perf_counter
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from fireants.losses.cc import gaussian_1d, separable_filtering
from collections import deque
import numpy as np

class ConvergenceMonitor:
    def __init__(self, N, slope):
        """
        Initialize the ConvergenceMonitor class.
        Args:
        - N: number of values to keep track of.
        """
        self.N = N
        self.losses = deque(maxlen=N)
        self.slope = slope

    def update(self, loss):
        """Append a new loss value to the monitor."""
        self.losses.append(loss)

    def _compute_slope(self):
        """Compute the slope of the best-fit line using simple linear regression."""
        if len(self.losses) < 2:
            # Can't compute a slope with less than 2 points
            return 0

        x = np.arange(len(self.losses))
        y = np.array(self.losses)

        # Compute the slope (m) of the best-fit line y = mx + c
        # m = (NΣxy - ΣxΣy) / (NΣx^2 - (Σx)^2)
        xy_sum = np.dot(x, y)
        x_sum = x.sum()
        y_sum = y.sum()
        x_squared_sum = (x**2).sum()
        N = len(self.losses)

        numerator = N * xy_sum - x_sum * y_sum
        denominator = N * x_squared_sum - x_sum**2
        if denominator == 0:
            return 0

        slope = numerator / denominator
        return slope

    def converged(self, loss=None):
        """Check if the loss has increased (i.e., slope > threshold).
        optionally, update the monitor with a new loss value.
        """
        if loss is not None:
            self.update(loss)
        if len(self.losses) < self.N:
            return False
        else:
            slope = self._compute_slope()
            return slope > self.slope
    
    def reset(self):
        self.losses.clear()


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

def collate_fireants_fn(batch):
    '''
    collate batch of arbitrary lists/tuples/dicts with collating the Images into BatchedImages object 
    '''
    raise NotImplementedError