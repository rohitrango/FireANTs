from cudants.registration.abstract import AbstractRegistration
from typing import List, Optional
import torch
from torch import nn
from cudants.io.image import BatchedImages
from torch.optim import SGD, Adam
from torch.nn import functional as F
from cudants.utils.globals import MIN_IMG_SIZE
from tqdm import tqdm
import numpy as np
from cudants.utils.opticalflow import OpticalFlow

class LogDemonsRegistration(AbstractRegistration):
    '''
    This class implements multi-scale log-demons registration
    '''    
    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, tolerance_mode: str = 'atol',
                init_affine: Optional[torch.Tensor] = None,
                custom_loss: nn.Module = None,
                ##### parameters for log-demons #####
                sigmas: List[float] = [1.0],
                ) -> None:
        # initialize abstract registration
        super().__init__(scales, iterations, fixed_images, moving_images, loss_type, mi_kernel_type, cc_kernel_type, custom_loss,
                         tolerance, max_tolerance_iters, tolerance_mode)
        self.dims = fixed_images.dims
        # this will be a (D+1, D+1) matrix
        if init_affine is None:
            self.affine = torch.eye(self.dims + 1, device=self.device, dtype=torch.float32).repeat(fixed_images.batch_size, 1, 1)
        else:
            self.affine = init_affine
        # no optimizer needed for this class
        self.optical_flow = OpticalFlow(optical_flow_method, sigma=optical_flow_sigma, no_grad=True, eps=eps, device=self.device)
        
        
        
