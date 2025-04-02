'''
Dispatcher for fused operations
'''
import os
from typing import Dict, Any, Callable
from fireants.interpolator.grid_sample import torch_grid_sampler_2d, torch_grid_sampler_3d, torch_warp_composer_2d, torch_warp_composer_3d, torch_affine_warp_2d, torch_affine_warp_3d
import logging
import torch
logger = logging.getLogger(__name__)

# Try to import fused operations
try:
    import fireants_fused_ops as ffo
    FFO_AVAILABLE = True
    # safe to import fused grid sampler
    from fireants.interpolator.fused_grid_sample import fused_grid_sampler_3d, fused_warp_composer_3d, fused_affine_warp_3d
except ImportError:
    logger.warning("Fused operations not available, compile the fused ops to use them")
    FFO_AVAILABLE = False
    ffo = None
    fused_grid_sampler_3d = None
    fused_warp_composer_3d = None
    fused_affine_warp_3d = None

# Get environment variable with default True
USE_FFO = os.getenv('USE_FFO', 'True').lower() == 'true'

class GridSampleDispatcher:
    """Dispatcher for grid sample operations that handles both fused and non-fused implementations."""
    
    def __init__(self):
        self._use_ffo = USE_FFO and FFO_AVAILABLE
        logger.info(f"Using FFO: {self._use_ffo}")
        self._registry: Dict[bool, Dict[str, Callable]] = {
            True: {},  # FFO backend
            False: {}  # PyTorch backend
        }
        self._setup_registry()
    
    def _setup_registry(self) -> None:
        """Set up the function registry with appropriate implementations."""
        # Set up FFO backend
        self._registry[True]['grid_sample_2d'] = torch_grid_sampler_2d  # TODO: add fused grid sampler 2d
        self._registry[True]['grid_sample_3d'] = fused_grid_sampler_3d
        self._registry[True]['warp_composer_2d'] = torch_warp_composer_2d  # TODO: add fused warp composer 2d
        self._registry[True]['warp_composer_3d'] = fused_warp_composer_3d
        self._registry[True]['affine_warp_3d'] = fused_affine_warp_3d
        self._registry[True]['affine_warp_2d'] = torch_affine_warp_2d    # TODO: add fused affine warp 2d

        # Set up PyTorch backend
        self._registry[False]['grid_sample_2d'] = torch_grid_sampler_2d
        self._registry[False]['grid_sample_3d'] = torch_grid_sampler_3d
        self._registry[False]['warp_composer_2d'] = torch_warp_composer_2d
        self._registry[False]['warp_composer_3d'] = torch_warp_composer_3d
        self._registry[False]['affine_warp_3d'] = torch_affine_warp_3d
        self._registry[False]['affine_warp_2d'] = torch_affine_warp_2d
    
    @property
    def use_ffo(self) -> bool:
        """Get the current backend state."""
        return self._use_ffo
    
    @use_ffo.setter
    def use_ffo(self, value: bool) -> None:
        """Set the backend state. Raises ValueError if trying to use FFO when not available."""
        if value and not FFO_AVAILABLE:
            raise ValueError("Cannot set use_ffo to True when fused operations are not available")
        self._use_ffo = value
    
    def __call__(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate grid sample implementation."""
        image = kwargs.get('image', None)
        if image is None:
            image = args[0]
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                return self.grid_sample_2d(*args, **kwargs)
            elif image.ndim == 5:
                return self.grid_sample_3d(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported image dimension: {image.ndim}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def warp_composer_2d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 2D warp composer implementation."""
        return self._registry[self._use_ffo]['warp_composer_2d'](*args, **kwargs)
    
    def warp_composer_3d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 3D warp composer implementation."""
        return self._registry[self._use_ffo]['warp_composer_3d'](*args, **kwargs)
    
    def grid_sample_2d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 2D grid sample implementation."""
        return self._registry[self._use_ffo]['grid_sample_2d'](*args, **kwargs)
    
    def grid_sample_3d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 3D grid sample implementation."""
        return self._registry[self._use_ffo]['grid_sample_3d'](*args, **kwargs)
    
    def __str__(self) -> str:
        """String representation of the dispatcher."""
        return f"GridSampleDispatcher(use_ffo={self._use_ffo}, registry={self._registry})"
    
    def affine_warp_3d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 3D affine warp implementation."""
        return self._registry[self._use_ffo]['affine_warp_3d'](*args, **kwargs)
    
    def affine_warp_2d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 2D affine warp implementation."""
        return self._registry[self._use_ffo]['affine_warp_2d'](*args, **kwargs)


# Create a singleton instance
fireants_interpolator = GridSampleDispatcher()
