from typing import List, Tuple
from time import perf_counter
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from fireants.losses.cc import gaussian_1d, separable_filtering
from collections import deque
import numpy as np
import gc
import os
import inspect
import logging
from typing import Optional
from fireants.interpolator import fireants_interpolator
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_tensor_memory_details() -> List[Tuple[torch.Tensor, float, str, str]]:
    """Get details of all tensors currently in memory.
    
    Returns:
        List of tuples containing (tensor, size_in_mb, tensor_description, variable_name)
    """
    tensor_details = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor = obj if torch.is_tensor(obj) else obj.data
                size_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                description = f"{tensor.shape} {tensor.dtype} {tensor.device}"
                # Try to find the variable name
                var_name = "unknown"
                for frame in inspect.stack():
                    frame_locals = frame.frame.f_locals
                    for var, val in frame_locals.items():
                        if val is obj:
                            var_name = var
                            break
                    if var_name != "unknown":
                        break
                
                tensor_details.append((tensor, size_mb, description, var_name))
        except Exception as e:
            logger.warning(f"Error getting tensor details: {e}")
            pass
    return sorted(tensor_details, key=lambda x: x[1], reverse=True)


def get_gpu_memory(clear: bool = False):
    """Get current GPU memory usage in MB"""
    if clear:
        torch.cuda.empty_cache()
        gc.collect()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024

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


def augment_filenames(filenames: List[str], batch_size: int, permitted_ext: List[str]):
    '''
    If filenames is a single string, return a list of batch_size strings with the filename
    If filenames is a list of strings > 1, do nothing
    '''
    # do nothing if batch_size == 1
    if batch_size == 1:
        return filenames
    # if filenames is a list with single item, return a list of batch_size strings with the filename (batch > 1)
    if len(filenames) == 1:
        for ext in permitted_ext:
            if filenames[0].endswith(ext):
                return [filenames[0].replace(ext, f"_{i}{ext}") for i in range(batch_size)]
        raise ValueError(f"No permitted extension found in {filenames[0]}")

    logger.warning(f"More than one filename provided, returning the same filename for all {batch_size} images")
    return filenames

def check_correct_ext(filenames: List[str], permitted_ext: List[str]):
    '''
    Check if the filenames have the correct extension
    '''
    for filename in filenames:
        if not any(filename.endswith(ext) for ext in permitted_ext):
            raise ValueError(f"File {filename} has an incorrect extension, must be one of {permitted_ext}")
    return True

def any_extension(filename: str, permitted_ext: List[str]):
    '''
    Check if the filename has any of the permitted extensions
    '''
    return any(filename.endswith(ext) for ext in permitted_ext)

def savetxt(filename: str, A: torch.Tensor, t: torch.Tensor):
    '''
    Save the transform matrix and translation vector to a text file
    '''
    dims = t.flatten().shape[0]
    with open(filename, 'w') as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("#Transform 0\n")
        f.write("Transform: AffineTransform_float_3_3\n")
        f.write("Parameters: " + " ".join(map(str, list(A.flatten()) + list(t.flatten()))) + "\n")
        f.write("FixedParameters: " + " ".join(map(str, np.zeros((dims, 1)).flatten())) + "\n")

# def compose_warp(warp1: torch.Tensor, warp2: torch.Tensor, grid: torch.Tensor):
def compose_warp(warp1: torch.Tensor, warp2: torch.Tensor, affine: Optional[torch.Tensor] = None):
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
    warp12 = fireants_interpolator.warp_composer(warp1, affine, warp2, align_corners=True)
    # warp12 = F.grid_sample(warp1.permute(*permute_vtoimg), grid + warp2, mode='bilinear', align_corners=True).permute(*permute_imgtov)
    return warp2 + warp12

def collate_fireants_fn(batch):
    '''
    collate batch of arbitrary lists/tuples/dicts with collating the Images into BatchedImages object 
    '''
    raise NotImplementedError

def check_and_raise_cond(cond: bool, msg: str, error_type: Exception = ValueError):
    if not cond:
        raise error_type(msg)
