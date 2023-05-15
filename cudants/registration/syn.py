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

class SynRegistration(AbstractRegistration):
    '''
    This class implements Symmetric normalization tools
    '''    
    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                optimizer: str = 'SGD', optimizer_params: dict = {},
                optimizer_lr: float = 0.1, optimizer_momentum: float = 0.0,
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, tolerance_mode: str = 'atol',
                init_affine: Optional[torch.Tensor] = None,
                custom_loss: nn.Module = None) -> None:
        # initialize abstract registration
        super().__init__(scales, iterations, fixed_images, moving_images, loss_type, mi_kernel_type, cc_kernel_type, custom_loss,
                         tolerance, max_tolerance_iters, tolerance_mode)
        self.dims = fixed_images.dims
        
