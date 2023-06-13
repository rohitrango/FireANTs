from typing import List
from time import perf_counter
from contextlib import contextmanager
import torch

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
    return separable_filtering(grad, gaussians)