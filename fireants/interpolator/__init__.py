# Copyright (c) 2025 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


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
        self._use_ffo = (USE_FFO and FFO_AVAILABLE)
        print(f"USE_FFO: {USE_FFO}")
        print(f"FFO_AVAILABLE: {FFO_AVAILABLE}")
        logger.info(f"Using FFO: {self._use_ffo}")
        print(f"Using FFO: {self._use_ffo}")
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
        print(f"Setting use_ffo to {value}")
        if value and not FFO_AVAILABLE:
            raise ValueError("Cannot set use_ffo to True when fused operations are not available")
        self._use_ffo = value
    
    def _get_image_dim(self, *args, image_idx: int = 0, arg_name: str = "image", **kwargs) -> int:
        ''' return spatial dimensions of image '''
        image = kwargs.get(arg_name, None)
        if image is None:
            image = args[image_idx]
        if isinstance(image, torch.Tensor):
            return image.ndim - 2
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    @staticmethod
    def augment_registry_backend(func: Callable):
        ''' For each function, we want to fallback to torch if the tensor is not on cuda '''
        def wrapper(self, *args, **kwargs):
            prev_ffo = self._use_ffo
            # get any tensor arg
            tensor_arg = [arg for arg in args if isinstance(arg, torch.Tensor)] + [arg for arg in kwargs.values() if isinstance(arg, torch.Tensor)]
            if len(tensor_arg) > 0 and any([x.device.type != 'cuda' for x in tensor_arg]):
                self._use_ffo = False
            ret = func(self, *args, **kwargs)
            self._use_ffo = prev_ffo
            return ret
        return wrapper
    
    @augment_registry_backend.__func__
    def __call__(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate grid sample implementation."""
        # print(f"Using FFO: {self._use_ffo}")
        dim = self._get_image_dim(*args, **kwargs)
        # print(f"Dispatching to {f'grid_sample_{dim}d'}")
        return self._registry[self._use_ffo][f'grid_sample_{dim}d'](*args, **kwargs)
    
    @augment_registry_backend.__func__
    def warp_composer(self, *args, **kwargs) -> Any:
        ''' Dispatch to appropriate warp composer implementation '''
        dim = self._get_image_dim(*args, **kwargs)
        return self._registry[self._use_ffo][f'warp_composer_{dim}d'](*args, **kwargs)
    
    @augment_registry_backend.__func__
    def affine_warp(self, *args, **kwargs) -> Any:
        ''' Dispatch to appropriate affine warp implementation '''
        dim = self._get_image_dim(*args, image_idx=1, **kwargs, arg_name="grid")
        return self._registry[self._use_ffo][f'affine_warp_{dim}d'](*args, **kwargs)

    # individual functions are called by the user at their discretion, we do not need to check
    # for the backend

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
