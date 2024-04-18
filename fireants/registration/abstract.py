from abc import ABC, abstractmethod
from typing import List
import torch
from torch import nn
from fireants.utils.util import _assert_check_scales_decreasing
from fireants.losses import GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss
from torch.optim import SGD, Adam
from fireants.io.image import BatchedImages
from typing import Optional
from fireants.utils.util import ConvergenceMonitor
from torch.nn import functional as F

def dummy_loss(*args):
    return 0

class AbstractRegistration(ABC):

    def __init__(self,
                scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                custom_loss: nn.Module = None,
                loss_params: dict = {},
                cc_kernel_size: int = 3, 
                reduction: str = 'mean',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, 
                ) -> None:
        '''
        Initialize abstract registration class
        '''
        super().__init__()
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
        self.convergence_monitor = ConvergenceMonitor(self.max_tolerance_iters, self.tolerance)

        self.device = fixed_images.device
        self.dims = self.fixed_images.dims

        # initialize losses
        if loss_type == 'mi':
            self.loss_fn = GlobalMutualInformationLoss(kernel_type=mi_kernel_type, reduction=reduction, **loss_params)
        elif loss_type == 'cc':
            self.loss_fn = LocalNormalizedCrossCorrelationLoss(kernel_type=cc_kernel_type, spatial_dims=self.dims, 
                                                               kernel_size=cc_kernel_size, reduction=reduction, **loss_params)
        elif loss_type == 'custom':
            self.loss_fn = custom_loss
        elif loss_type == 'mse':
            self.loss_fn = partial(F.mse_loss, reduction=reduction)
        else:
            raise ValueError(f"Loss type {loss_type} not supported")

    @abstractmethod
    def optimize(self):
        pass
