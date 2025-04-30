from abc import ABC, abstractmethod
from typing import List
import torch
from torch import nn
from fireants.utils.util import _assert_check_scales_decreasing
from fireants.losses import GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss, NoOp, MeanSquaredError
from torch.optim import SGD, Adam
from fireants.io.image import BatchedImages, FakeBatchedImages
from typing import Optional, Union
from fireants.utils.util import ConvergenceMonitor
from torch.nn import functional as F
from functools import partial
from fireants.utils.imageutils import is_torch_float_type
from fireants.interpolator import fireants_interpolator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def dummy_loss(*args):
    return 0

class AbstractRegistration(ABC):
    """Base class for all registration algorithms in FireANTs.

    This abstract class provides the core functionality and interface for image registration,
    handling all the common functionality for all linear and non-linear registration algorithms.
    It handles features like
        - multi-resolution optimization, 
        - arbitrary similarity metrics, and 
        - convergence monitoring.

    Args:
        scales (List[int]): Downsampling factors for multi-resolution registration.
            Must be in descending order (e.g. [4,2,1]).
        iterations (List[float]): Number of iterations to perform at each scale.
            Must match length of scales.
        fixed_images (BatchedImages): Fixed/reference images to register to.
        moving_images (BatchedImages): Moving images to be registered.
        loss_type (str, optional): Similarity metric to use. A set of predefined loss functions are provided:
            - 'mi': Mutual information
                Global mutual information between the fixed image and the moved image is measured using Parzen windowing using different kernel types.
                Mutual information is implemented using the `GlobalMutualInformationLoss` class.
            - 'cc': Cross correlation (default)
                Local normalized cross correlation between the fixed image and the moved image is measured using a kernel of size `cc_kernel_size`. This is a de-facto standard similarity metric for both linear and non-linear image registration. It also consumes less very little memory than mutual information.  
                Cross correlation is implemented using the `LocalNormalizedCrossCorrelationLoss` class.
            - 'mse': Mean squared error
                Mean squared error is implemented using the `MeanSquaredError` class.
                This is the fastest similarity metric, but it is not robust to outliers, multi-modal images, or even intensity inhomogeneities within the same modality. Good for testing registration pipelines.
            - 'custom': Custom loss function (see `custom_loss` for more details)
            - 'noop': No similarity metric  (this is useful for testing regularizations)
        mi_kernel_type (str, optional): Kernel type for mutual information loss. Default: 'b-spline'
        cc_kernel_type (str, optional): Kernel type for cross correlation loss. Default: 'rectangular'
        custom_loss (nn.Module, optional): Custom loss module if loss_type='custom'.
            See [Custom Loss Functions](../advanced/customloss.md) for more details on how to implement custom loss functions.
        loss_params (dict, optional): Additional parameters for loss function. See the documentation of the loss function for more details on what parameters are available. This implementation abstracts away the loss function implementation details from the registration pipeline.
        cc_kernel_size (int, optional): Kernel size for cross correlation loss. Default: 3
        reduction (str, optional): Loss reduction method. Default: 'mean'
        tolerance (float, optional): Convergence tolerance. Default: 1e-6
        max_tolerance_iters (int, optional): Max iterations for convergence check. Default: 10
        progress_bar (bool, optional): Whether to show progress bar. Default: True

    Methods:
        optimize(): Abstract method to perform registration optimization
        get_warped_coordinates(): Abstract method to get transformed coordinates
        evaluate(): Apply learned transformation to new images

    Note:
        The number of scales and iterations must match, and scales must be in descending order.
        The fixed and moving images must be broadcastable in batch dimension.
    """

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
                progress_bar: bool = True,
                dtype: torch.dtype = torch.float32,
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
        # assert (self.fixed_images.size() == self.moving_images.size()), "Number of fixed and moving images must match"

        # check if sizes are broadcastable
        fsize, msize = self.fixed_images.size(), self.moving_images.size()
        assert (fsize == msize) or (fsize == 1) or (msize == 1), "Number of fixed and moving images must match or broadcastable"
        self.opt_size = max(fsize, msize)
        
        self.tolerance = tolerance
        self.max_tolerance_iters = max_tolerance_iters
        self.convergence_monitor = ConvergenceMonitor(self.max_tolerance_iters, self.tolerance)

        self.device = fixed_images.device
        self.dtype = dtype
        if not is_torch_float_type(self.dtype):
            raise ValueError(f"non-float dtype {self.dtype} is not supported for registration")

        self.dims = self.fixed_images.dims
        self.progress_bar = progress_bar        # variable to show or hide progress bar
        # initialize losses
        if loss_type == 'mi':
            self.loss_fn = GlobalMutualInformationLoss(kernel_type=mi_kernel_type, reduction=reduction, **loss_params)
        elif loss_type == 'cc':
            self.loss_fn = LocalNormalizedCrossCorrelationLoss(kernel_type=cc_kernel_type, spatial_dims=self.dims, 
                                                               kernel_size=cc_kernel_size, reduction=reduction, **loss_params)
        elif loss_type == 'fusedcc':
            from fireants.losses.fusedcc import FusedLocalNormalizedCrossCorrelationLoss
            self.loss_fn = FusedLocalNormalizedCrossCorrelationLoss(spatial_dims=self.dims, 
                                                    kernel_size=cc_kernel_size, reduction=reduction, **loss_params)
        elif loss_type == 'custom':
            self.loss_fn = custom_loss
        elif loss_type == 'noop':
            self.loss_fn = NoOp()
        elif loss_type == 'mse':
            # self.loss_fn = partial(F.mse_loss, reduction=reduction)
            self.loss_fn = MeanSquaredError(reduction=reduction)
        else:
            raise ValueError(f"Loss type {loss_type} not supported")
        
        # see if loss can store the iterations
        if hasattr(self.loss_fn, 'set_iterations'):
            logger.info("Setting iterations for loss function")
            self.loss_fn.set_iterations(self.iterations)
        if hasattr(self.loss_fn, 'set_scales'):
            logger.info("Setting scales for loss function")
            self.loss_fn.set_scales(self.scales)

        self.print_init_msg()

    def print_init_msg(self):
        logger.info(f"Registration of type {self.__class__.__name__} initialized with dtype {self.dtype}")

    @abstractmethod
    def optimize(self):
        ''' 
        Abstract method to perform registration optimization
        '''
        pass

    @abstractmethod
    def get_warp_parameters(self, fixed_images: Union[BatchedImages, FakeBatchedImages], moving_images: Union[BatchedImages, FakeBatchedImages], shape=None):
        ''' Get dictionary of parameters to pass into fireants_interpolator '''
        raise NotImplementedError("This method must be implemented by the registration class")
    
    @abstractmethod
    def get_inverse_warp_parameters(self, fixed_images: Union[BatchedImages, FakeBatchedImages], moving_images: Union[BatchedImages, FakeBatchedImages], shape=None):
        ''' Get dictionary of parameters to pass into fireants_interpolator '''
        raise NotImplementedError("This method must be implemented by the registration class")

    def get_warped_coordinates(self, fixed_images: Union[BatchedImages, FakeBatchedImages], moving_images: Union[BatchedImages, FakeBatchedImages], shape=None):
        '''Get the transformed coordinates for warping the moving image.

        This abstract method must be implemented by all registration classes to define how
        coordinates are transformed from the fixed image space to the moving image space.
        The transformed coordinates are used with grid_sample to warp the moving image.

        Args:
            fixed_images (BatchedImages): Fixed/reference images that define the target space.
                The output coordinates will be in this image's coordinate system.
            moving_images (BatchedImages): Moving images to be transformed.
                Used to access the original image space parameters.
            shape (Optional[Tuple[int, ...]]): Optional output shape for the coordinate grid.
                If None, uses the shape of the fixed image.

        Returns:
            torch.Tensor: Transformed coordinates in normalized [-1, 1] space.
                Has shape [N, H, W, [D], dims] where:
                    N = batch size
                    H, W, [D] = spatial dimensions (D only for 3D)
                    dims = number of spatial dimensions (2 or 3)
                These coordinates can be used directly with `torch.nn.functional.grid_sample()`

        Note:
            - Coordinates are returned in the normalized [-1, 1] coordinate system required by grid_sample
            - The transformation maps from fixed image space to moving image space (backward transform)
            - Physical space transformations are handled internally using the image metadata
        '''
        params = self.get_warp_parameters(fixed_images, moving_images, shape)
        if 'affine' in params and 'grid' not in params:
            return F.affine_grid(params['affine'], params['out_shape'], align_corners=True)
        elif 'affine' in params and 'grid' in params:
            # this is just a warp field
            affine = params['affine']
            grid = params['grid']
            grid = fireants_interpolator.affine_warp(affine, grid, align_corners=True)
            return grid
        else:
            raise ValueError(f"Invalid warp parameters with keys {params.keys()}")

    def get_inverse_warped_coordinates(self, fixed_images: Union[BatchedImages, FakeBatchedImages], moving_images: Union[BatchedImages, FakeBatchedImages], shape=None):
        ''' Get inverse warped coordinates for the moving image.

        This method is useful to analyse the effect of how the moving coordinates (fixed images) are transformed
        '''
        params = self.get_inverse_warp_parameters(fixed_images, moving_images, shape)
        if 'affine' in params and 'grid' not in params:
            return F.affine_grid(params['affine'], params['out_shape'], align_corners=True)
        elif 'affine' in params and 'grid' in params:
            # this is just a warp field
            affine = params['affine']
            grid = params['grid']
            shape = [affine.shape[0], 1] + list(grid.shape[1:-1])
            grid = fireants_interpolator.affine_warp(affine, grid, align_corners=True)
            return grid
        else:
            raise ValueError(f"Invalid warp parameters with keys {params.keys()}")

    def save_moved_images(self, moved_images: Union[BatchedImages, FakeBatchedImages, torch.Tensor], filenames: Union[str, List[str]], moving_to_fixed: bool = True):
        '''
        Save the moved images to disk.

        Args:
            moved_images (Union[BatchedImages, FakeBatchedImages, torch.Tensor]): The moved images to save.
            filenames (Union[str, List[str]]): The filenames to save the moved images to.
            moving_to_fixed (bool, optional): If True, the moving images are saved to the fixed image space. Defaults to True.
                if False, we are dealing with an image that is moved from fixed space to moving space            
        '''
        if isinstance(moved_images, BatchedImages):
            moved_images_save = FakeBatchedImages(moved_images(), moved_images)   # roundabout way to call the fakebatchedimages
        elif isinstance(moved_images, torch.Tensor):
            moved_images_save = FakeBatchedImages(moved_images, self.fixed_images if moving_to_fixed else self.moving_images)
        else:
            # if it is already a fakebatchedimages, we can just use it
            moved_images_save = moved_images
        moved_images_save.write_image(filenames)


    def evaluate_inverse(self, fixed_images: Union[BatchedImages, torch.Tensor], moving_images: Union[BatchedImages, torch.Tensor], shape=None, **kwargs):
        ''' Apply the inverse of the learned transformation to new images.

        This method is useful to analyse the effect of how the moving coordinates (fixed images) are transformed
        '''
        if isinstance(fixed_images, torch.Tensor):
            fixed_images = FakeBatchedImages(fixed_images, self.fixed_images)
        if isinstance(moving_images, torch.Tensor):
            moving_images = FakeBatchedImages(moving_images, self.moving_images)

        fixed_arrays = moving_images()
        fixed_moved_coords = self.get_inverse_warp_parameters(fixed_images, moving_images, shape=shape, **kwargs)
        fixed_moved_image = fireants_interpolator(fixed_arrays, **fixed_moved_coords, mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
        return fixed_moved_image


    def evaluate(self, fixed_images: Union[BatchedImages, torch.Tensor], moving_images: Union[BatchedImages, torch.Tensor], shape=None):
        '''Apply the learned transformation to new images.

        This method applies the registration transformation learned during optimization
        to a new set of images. It can be used to:
            - Validate registration performance on test images
            - Apply learned transformations to new data
            - Transform auxiliary data (e.g. segmentation masks) using learned parameters
        
        All registration classes will implement their own `get_warped_coordinates` method, 
        which is used to apply the learned transformation to new images.

        Args:
            fixed_images (BatchedImages): Fixed/reference images that define the target space
            moving_images (BatchedImages): Moving images to be transformed
            shape (Optional[Tuple[int, ...]]): Optional output shape for the transformed image.
                If None, uses the shape of the fixed image.

        Returns:
            torch.Tensor: The transformed moving image in the space of the fixed image.
                Has shape [N, C, H, W, [D]] where:
                    N = batch size
                    C = number of channels
                    H, W, D = spatial dimensions (D only for 3D)

        Note:
            The transformation is applied using bilinear interpolation with align_corners=True
            to maintain consistency with the optimization process.
        '''
        if isinstance(fixed_images, torch.Tensor):
            fixed_images = FakeBatchedImages(fixed_images, self.fixed_images)
        if isinstance(moving_images, torch.Tensor):
            moving_images = FakeBatchedImages(moving_images, self.moving_images)

        moving_arrays = moving_images()
        moved_coords = self.get_warp_parameters(fixed_images, moving_images, shape=shape)
        moved_image = fireants_interpolator(moving_arrays, **moved_coords, mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
        return moved_image