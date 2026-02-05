# Scale-Aware Stateful Loss Functions

FireANTs supports multi-scale registration where optimization proceeds through multiple resolution scales (e.g., `scales=[8, 4, 2, 1]`). Some loss functions may need to adapt their behavior at different scales, such as adjusting kernel sizes, smoothing parameters, or other hyperparameters. This guide explains how to implement **scale-aware stateful loss functions** that can dynamically adjust their parameters during multi-scale optimization.

## Overview

A scale-aware loss function implements three optional methods that are called by the registration framework:

1. **`set_scales(scales)`** - Called once during initialization to inform the loss about the scales
2. **`set_iterations(iterations)`** - Called once during initialization to inform the loss about iterations per scale
3. **`set_current_scale_and_iterations(scale, iters)`** - Called at the start of each scale to update the loss state

These methods allow the loss function to maintain state and adapt its behavior as the registration progresses through different scales.

## Example: Multi-Scale Kernel Size for Cross-Correlation

The `FusedLocalNormalizedCrossCorrelationLoss` and `LocalNormalizedCrossCorrelationLoss` classes demonstrate scale-aware behavior by allowing different kernel sizes at different scales.

### Implementation Pattern

Here's how to implement a scale-aware loss function:

```python
import torch
from torch import nn
from typing import List, Union

class ScaleAwareLoss(nn.Module):
    """
    Example scale-aware loss function that adjusts kernel size per scale.
    """
    
    def __init__(
        self,
        kernel_size: Union[int, List[int]] = 3,
        spatial_dims: int = 3,
        **kwargs
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        
        # Support both single kernel size and list of kernel sizes
        self.kernel_size_list = kernel_size if isinstance(kernel_size, (list, tuple)) else None
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        
        # Initialize state tracking
        self.scales = None
        self.iterations = None
        
        # Initialize kernel with first size
        self._initialize_kernel(self.kernel_size)
    
    def _initialize_kernel(self, kernel_size: int):
        """Initialize or update the kernel based on kernel size."""
        # Example: create a simple averaging kernel
        self.kernel = torch.ones(kernel_size) / kernel_size
        self.kernel.requires_grad = False
    
    def set_scales(self, scales: List[float]) -> None:
        """
        Called once during registration initialization.
        
        Args:
            scales: List of scale factors (e.g., [8.0, 4.0, 2.0, 1.0])
        """
        self.scales = scales
        if self.kernel_size_list:
            assert len(self.kernel_size_list) == len(self.scales), \
                f"kernel_size_list must have the same length as scales, " \
                f"got {len(self.kernel_size_list)} vs {len(self.scales)}"
    
    def set_iterations(self, iterations: List[int]) -> None:
        """
        Called once during registration initialization.
        
        Args:
            iterations: List of iteration counts per scale (e.g., [200, 100, 50, 25])
        """
        self.iterations = iterations
    
    def set_current_scale_and_iterations(self, scale: float, iters: int) -> None:
        """
        Called at the start of each scale during optimization.
        
        This is where you update the loss function's state based on the current scale.
        
        Args:
            scale: Current scale factor (e.g., 8.0, 4.0, 2.0, 1.0)
            iters: Current iteration count for this scale
        """
        if self.kernel_size_list and self.scales is not None:
            # Find the index of the current scale
            idx = self.scales.index(scale)
            new_kernel_size = self.kernel_size_list[idx]
            
            # Only update if kernel size changed
            if new_kernel_size != self.kernel_size:
                self.kernel_size = new_kernel_size
                self._initialize_kernel(self.kernel_size)
                # Optionally update other scale-dependent parameters here
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - uses current kernel_size which may have been updated
        by set_current_scale_and_iterations.
        """
        # Your loss computation here using self.kernel_size
        # ...
        return loss
```

### Real-World Example: FusedLocalNormalizedCrossCorrelationLoss

Here's the actual implementation from `FusedLocalNormalizedCrossCorrelationLoss`:

```python
class FusedLocalNormalizedCrossCorrelationLoss(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: Union[int, List[int]] = 3,
        # ... other parameters
    ):
        super().__init__()
        # Store kernel_size_list if provided, otherwise None
        self.kernel_size_list = kernel_size if isinstance(kernel_size, (list, tuple)) else None
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        
        # Initialize state
        self.scales = None
        self.iterations = None
    
    def set_scales(self, scales):
        """Called at initialization of abstract registration."""
        self.scales = scales
        if self.kernel_size_list:
            assert len(self.kernel_size_list) == len(self.scales), \
                "kernel_size must be a list of the same length as scales"
    
    def set_iterations(self, iterations):
        """Called at initialization of abstract registration."""
        self.iterations = iterations
    
    def set_current_scale_and_iterations(self, scale, iters):
        """Update kernel size based on current scale."""
        if self.kernel_size_list:
            idx = self.scales.index(scale)
            self.kernel_size = self.kernel_size_list[idx]
```

## How Registration Classes Use Scale-Aware Losses

Registration classes automatically detect and call these methods. Here's how `GreedyRegistration` handles it:

```python
class GreedyRegistration(AbstractRegistration):
    def optimize(self):
        # ... setup code ...
        
        for scale, iters in zip(self.scales, self.iterations):
            self.convergence_monitor.reset()
            
            # Notify loss function of scale change if it supports it
            if hasattr(self.loss_fn, 'set_current_scale_and_iterations'):
                self.loss_fn.set_current_scale_and_iterations(scale, iters)
            
            # ... rest of optimization loop ...
```

The registration framework:

1. **During initialization**: Calls `set_scales()` and `set_iterations()` if the loss function implements them (see `AbstractRegistration.__init__`)
2. **During optimization**: Calls `set_current_scale_and_iterations()` at the start of each scale if the method exists

## Usage Example

Here's how to use a scale-aware loss with different kernel sizes per scale:

```python
from fireants.registration.greedy import GreedyRegistration
from fireants.losses.fusedcc import FusedLocalNormalizedCrossCorrelationLoss

# Define kernel sizes for each scale
# scales = [8, 4, 2, 1] -> kernel_sizes = [3, 5, 7, 9]
kernel_sizes = [3, 5, 7, 9]

# Create registration with scale-aware kernel sizes
reg = GreedyRegistration(
    scales=[8, 4, 2, 1],
    iterations=[200, 100, 50, 25],
    fixed_images=fixed_images,
    moving_images=moving_images,
    loss_type='fusedcc',
    cc_kernel_size=kernel_sizes,  # Pass list instead of single int
    # ... other parameters
)

# The loss function will automatically adjust kernel size at each scale
reg.optimize()
```

## Best Practices

1. **Optional Methods**: All three methods (`set_scales`, `set_iterations`, `set_current_scale_and_iterations`) are optional. Only implement them if your loss function needs scale-aware behavior.

2. **Validation**: Always validate that `kernel_size_list` (or similar parameter lists) match the length of `scales` in `set_scales()`.

3. **State Updates**: Only update state in `set_current_scale_and_iterations()` if necessary. Check if values have changed before expensive operations.

4. **Backward Compatibility**: Support both single values and lists for parameters that can vary by scale:
   ```python
   self.param_list = param if isinstance(param, (list, tuple)) else None
   self.param = param[0] if isinstance(param, (list, tuple)) else param
   ```

5. **Documentation**: Clearly document which parameters support scale-aware behavior in your loss function's docstring.

## When to Use Scale-Aware Losses

Consider implementing scale-aware behavior when:

- **Kernel sizes** need to vary with image resolution (e.g., larger kernels at coarser scales)
- **Smoothing parameters** should adapt to scale (e.g., different Gaussian sigmas)
- **Regularization strength** should change per scale
- **Feature extraction** parameters need adjustment (e.g., different patch sizes)
- **Any hyperparameter** that benefits from scale-dependent tuning

## Summary

Scale-aware loss functions enable dynamic adaptation during multi-scale registration by implementing three optional methods:

- `set_scales()` - Receive scale information
- `set_iterations()` - Receive iteration information  
- `set_current_scale_and_iterations()` - Update state at each scale

The registration framework automatically detects and calls these methods, making it easy to create sophisticated loss functions that adapt to the current optimization scale.
