from abc import ABC, abstractmethod
from typing import List
import torch
from torch import nn
from cudants.utils.util import _assert_check_scales_decreasing
from cudants.losses import GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss
from torch.optim import SGD, Adam
from cudants.io.image import BatchedImages

def dummy_loss(*args):
    return 0

class AbstractRegistration(ABC):

    def __init__(self,
                scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                custom_loss: nn.Module = None,
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, tolerance_mode: str = 'atol'
                ) -> None:
        '''
        Initialize abstract registration class

        '''
        self.scales = scales
        _assert_check_scales_decreasing(self.scales)
        self.iterations = iterations
        assert len(self.iterations) == len(self.scales), "Number of iterations must match number of scales"
        # check for fixed and moving image sizes
        self.fixed_images = fixed_images
        self.moving_images = moving_images
        assert (self.fixed_images.size() == self.moving_images.size()), "Number of fixed and moving images must match"
        
        self.tolerance = tolerance
        self.max_tolerance_iters = max_tolerance_iters
        self.tolerance_mode = tolerance_mode

        # initialize losses
        if loss_type == 'mi':
            self.loss_fn = GlobalMutualInformationLoss(kernel_type=mi_kernel_type)
        elif loss_type == 'cc':
            self.loss_fn = LocalNormalizedCrossCorrelationLoss(kernel_type=cc_kernel_type)
        elif loss_type == 'custom':
            self.loss_fn = custom_loss
        else:
            raise ValueError(f"Loss type {loss_type} not supported")

    @abstractmethod
    def optimize(self):
        pass
