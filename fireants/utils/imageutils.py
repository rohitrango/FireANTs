import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from fireants.utils.util import catchtime
from typing import Optional, List
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.types import ItemOrList
import warnings
from enum import Enum
import logging
logger = logging.getLogger(__name__)
try:
    import fireants_fused_ops as ffo
except ImportError:
    logger.warn("fireants_fused_ops not found, will use torch fft instead")
    ffo = None

class TorchFloatType(Enum):
    FLOAT16 = torch.float16
    FLOAT32 = torch.float32
    FLOAT64 = torch.float64
    BFLOAT16 = torch.bfloat16

def is_torch_float_type(dtype: torch.dtype):
    return any([dtype == t.value for t in TorchFloatType])

def lie_bracket(u: torch.Tensor, v: torch.Tensor):
    if len(u.shape) == 4:
        return lie_bracket_2d(u, v)
    elif len(u.shape) == 5:
        return lie_bracket_3d(u, v)
    else:
        raise ValueError(f"lie_bracket not implemented for tensors of shape {u.shape}")

def lie_bracket_2d(u: torch.Tensor, v: torch.Tensor):
    '''
    u: displacement vector of size [N, H, W, 2]
    v: displacement vector of size [N, H, W, 2]
    '''
    B, H, W,  _ = u.shape
    newshape = [B, 2, H, W, 3]
    # Initialize Jacobian tensors
    J_u = torch.empty(newshape, dtype=u.dtype, device=u.device)
    J_v = torch.empty(newshape, dtype=u.dtype, device=u.device)
    # Compute Jacobian of u and v using image_gradient_singlechannel function
    for i in range(2):
        J_u[..., i] = image_gradient_singlechannel(u[..., i].reshape(B, 1, H, W))
        J_v[..., i] = image_gradient_singlechannel(v[..., i].reshape(B, 1, H, W))

    # Compute the dot product of the Jacobians with v and u respectively
    J_u_v = torch.einsum('bjhwi,bhwi->bhwj', J_u, v)
    J_v_u = torch.einsum('bjhwi,bhwi->bhwj', J_v, u)
    lie_bracket = J_u_v - J_v_u
    return lie_bracket

def jacobian_2d(u: torch.Tensor, normalize: bool):
    ''' u: displacement vector of size [N, H, W, 2] '''
    B, H, W, _ = u.shape
    newshape = [B, 2, H, W, 2]
    J = torch.empty(newshape, dtype=u.dtype, device=u.device)
    # Compute Jacobian of u and v using image_gradient_singlechannel function
    for i in range(2):
        J[..., i] = image_gradient_singlechannel(u[..., i].reshape(B, 1, H, W), normalize)
    return J

def jacobian_3d(u: torch.Tensor, normalize: bool):
    ''' u: displacement vector of size [N, H, W, D, 3] '''
    B, H, W, D, _ = u.shape
    newshape = [B, 3, H, W, D, 3]
    J = torch.empty(newshape, dtype=u.dtype, device=u.device)
    for i in range(3):
        J[..., i] = image_gradient_singlechannel(u[..., i].reshape(B, 1, H, W, D), normalize)
    return J

def jacobian(u: torch.Tensor, normalize=True):
    '''
    u: displacement vector of size [N, H, W, [D], dims]
    '''
    if len(u.shape) == 4:
        return jacobian_2d(u, normalize)
    elif len(u.shape) == 5:
        return jacobian_3d(u, normalize)
    else:
        raise ValueError(f"jacobian not implemented for tensors of shape {u.shape}")


def lie_bracket_3d(u: torch.Tensor, v: torch.Tensor):
    '''
    u: displacement vector of size [N, H, W, [D], dims]
    v: displacement vector of size [N, H, W, [D], dims]
    '''
    B, H, W, D, _ = u.shape
    newshape = [B, 3, H, W, D, 3]
    # Initialize Jacobian tensors
    J_u = torch.empty(newshape, dtype=u.dtype, device=u.device)
    J_v = torch.empty(newshape, dtype=u.dtype, device=u.device)
    # Compute Jacobian of u and v using image_gradient_singlechannel function
    for i in range(3):
        J_u[..., i] = image_gradient_singlechannel(u[..., i].reshape(B, 1, H, W, D))
        J_v[..., i] = image_gradient_singlechannel(v[..., i].reshape(B, 1, H, W, D))

    # Compute the dot product of the Jacobians with v and u respectively
    J_u_v = torch.einsum('bjhwdi,bhwdi->bhwdj', J_u, v)
    J_v_u = torch.einsum('bjhwdi,bhwdi->bhwdj', J_v, u)
    # Compute Lie bracket as [u,v] = J_u*v - J_v*u
    lie_bracket = J_u_v - J_v_u
    return lie_bracket

def downsample(image: torch.Tensor, size: List[int], mode: str, sigma: Optional[torch.Tensor]=None,
               gaussians: Optional[torch.Tensor] = None, use_fft=True) -> torch.Tensor:
    ''' 
    this function is to downsample the image to the given size
    but first, we need to perform smoothing 
    if sigma is provided (in voxels), then use this sigma for downsampling, otherwise infer sigma
    '''
    if use_fft and ffo is None:
        logger.warn("fireants_fused_ops is not found, will default to standard downsampling")
        use_fft = False

    if use_fft:
        return downsample_fft(image, size)

    if gaussians is None:
        if sigma is None:
            orig_size = list(image.shape[2:])
            sigma = [0.5 * orig_size[i] / size[i] for i in range(len(orig_size))]   # use sigma as the downsampling factor
        sigma = torch.tensor(sigma, dtype=image.dtype, device=image.device)
        # create gaussian convs
        gaussians = [gaussian_1d(s, truncated=2) for s in sigma]
    # otherwise gaussians is given, just downsample
    image_down = separable_filtering(image, gaussians)
    image_down = F.interpolate(image_down, size=size, mode=mode, align_corners=True)
    return image_down

def downsample_fft(image: torch.Tensor, size: List[int], padding=1) -> torch.Tensor:
    ''' downsample using fft transform instead '''
    dims = num_dims = len(image.shape) - 2
    dims = [-x for x in range(1, dims+1)]
    im_fft = torch.fft.fftn(image, dim=dims)
    im_fft = torch.fft.fftshift(im_fft, dim=dims)
    # get spatial coords
    source_dims = list(image.shape[2:])
    target_dims = size
    # print(source_dims, target_dims)
    # get start and end indices
    start_idx = [H//2 - (h//2 + padding) for H, h in zip(source_dims, target_dims)]
    end_idx = [si + sz + padding+1 for si, sz in zip(start_idx, target_dims)]
    # crop the image
    multiplier = [1.0 * sz / SZ for sz, SZ in zip(target_dims, source_dims)]
    multiplier = np.prod(multiplier)
    # print(multiplier, [-(h//2 + padding) for h in target_dims], [h - (h//2) + padding for h in target_dims])
    if num_dims == 2:
        im_fft = im_fft[:, :, start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]].contiguous()
        ffo.gaussian_blur_fft2(im_fft, *[-(h//2 + padding) for h in target_dims], *[h - (h//2) + padding for h in target_dims], multiplier)
    elif num_dims == 3:
        im_fft = im_fft[:, :, start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]].contiguous()
        ffo.gaussian_blur_fft3(im_fft, *[-(h//2 + padding) for h in target_dims], *[h - (h//2) + padding for h in target_dims], multiplier)
    else:
        raise ValueError(f"Invalid dimension: {dims}")
    
    # print("cropped", im_fft.shape)
    # crop the image
    if padding > 0:
        if num_dims == 2:
            im_fft = im_fft[:, :, padding:-padding, padding:-padding]
        elif num_dims == 3:
            im_fft = im_fft[:, :, padding:-padding, padding:-padding, padding:-padding]
    
    # print("after padding", im_fft.shape)
    im_fft = torch.fft.ifftshift(im_fft, dim=dims)
    im_fft = torch.real(torch.fft.ifftn(im_fft, dim=dims)).contiguous()
    return im_fft
    

def apply_gaussian(image: torch.Tensor, sigma: torch.Tensor, truncated: float = 2) -> torch.Tensor:
    '''
    wrap the `gaussian_1d` and `separable filtering into one`
    '''
    gaussians = [gaussian_1d(s, truncated=truncated) for s in sigma]
    image_smooth = separable_filtering(image, gaussians)
    return image_smooth

def scaling_and_squaring(u, grid, n = 6):
    """
    Apply scaling and squaring to a displacement field
    
    :param u: Input stationary velocity field, PyTorch tensor of shape [B, D, H, W, 3] or [B, H, W, 2]
    :param grid: Sampling grid of size [B, D, H, W, dims]  or [B, H, W, dims]
    :param n: Number of iterations of scaling and squaring (default: 6)
    
    :returns: Output displacement field, v, PyTorch tensor of shape [B, D, H, W, dims] or [B, H, W, dims]
    """
    dims = u.shape[-1]
    v = (1.0/2**n) * u
    if dims == 3:
        for i in range(n):
            vimg = v.permute(0, 4, 1, 2, 3)          # [1, 3, D, H, W]
            v = v + F.grid_sample(vimg, v + grid, align_corners=True).permute(0, 2, 3, 4, 1)
    elif dims == 2:
        for i in range(n):
            vimg = v.permute(0, 3, 1, 2)
            v = v + F.grid_sample(vimg, v + grid, align_corners=True).permute(0, 2, 3, 1)
    else:
        raise ValueError('Invalid dimension: {}'.format(dims))
    return v

def _find_integrator_n(u):
    raise NotImplementedError('Automatic integrator_n not implemented yet')

def image_gradient_singlechannel(image, normalize=False):
    """
    Compute the gradient of an image using central difference approximation
    :param I: input image, represented as a [B,1,D,H,W] or [B,1,H,W] tensor
    :returns: gradient of the input image, represented as a [B,C,D,H,W] or [B,C,H,W]  tensor

    :TODO: Add support for multichannel images
    """
    dims = len(image.shape) - 2
    device = image.device
    grad = None
    if dims == 2:
        B, C, H, W = image.shape
        if normalize:
            facx, facy = (W-1)/2, (H-1)/2
        else:
            facx, facy = 1, 1 
        k = torch.tensor([[-1.0, 0.0, 1.0]], device=device, dtype=image.dtype)[None, None] / 2
        gradx = F.conv2d(image, facx * k, padding=(0, 1))
        grady = F.conv2d(image, facy * k.permute(0, 1, 3, 2), padding=(1, 0))
        grad = torch.cat([gradx, grady], dim=1)
    elif dims == 3:
        B, C, D, H, W = image.shape
        if normalize:
            facx, facy, facz = (W-1)/2, (H-1)/2, (D-1)/2
        else:
            facx, facy, facz = 1, 1, 1
        k = torch.tensor([[[-1.0, 0.0, 1.0]]], device=device, dtype=image.dtype)[None, None] / 2
        gradx = F.conv3d(image, facx * k, padding=(0, 0, 1))
        grady = F.conv3d(image, facy * k.permute(0, 1, 2, 4, 3), padding=(0, 1, 0))
        gradz = F.conv3d(image, facz * k.permute(0, 1, 4, 2, 3), padding=(1, 0, 0))
        grad = torch.cat([gradx, grady, gradz], dim=1)
    else:
        raise ValueError('Invalid dimension: {}'.format(dims))
    return grad

def image_gradient(image, normalize=False):
    ''' compute the image gradient using central difference approximation '''
    c = image.shape[1]
    if c == 1:
        return image_gradient_singlechannel(image, normalize)
    else:
        raise NotImplementedError('Multichannel images not supported yet')

def integer_to_onehot(image: torch.Tensor, background_label:int=0, dtype: torch.dtype = None, max_label=None):
    ''' convert an integer map into one hot mapping
    assumed the image is of size [H, W, [D]] and we convert it into [C, H, W, [D]]

    background_label: this is the label to ignore (default: 0)
    max_label: max value of the label expected in the label segmentation, which sometimes may not be present
    we provide this as an additional option in case some images do not have the anatomy corresponding to the max label

    if None, we assume the image has that label already
    '''
    if max_label is None:
        max_label = image.max()
    if background_label >= 0 and background_label <= max_label: # we will ignore it
        num_labels = max_label
    else:
        num_labels = max_label + 1
    dtype = dtype if dtype is not None else image.dtype
    onehot = torch.zeros((num_labels, *image.shape), dtype=dtype, device=image.device)
    count = 0
    for i in range(num_labels+1):
        if i == background_label:
            continue
        onehot[count, ...] = (image == i).to(dtype)
        count += 1
    return onehot

# no_grad versions
scaling_and_squaring_nograd = torch.no_grad()(scaling_and_squaring)
image_gradient_nograd = torch.no_grad()(image_gradient)

def compute_inverse_warp_displacement(warp, grid, initial_inverse=None, iters=20, lr=1e-1, warp_gaussian=None, grad_gaussian=None, debug=False, optimizer='SGD'):
    ''' 
    Compute the inverse warp using a given warp, grid and optional initialization
    '''
    warnings.warn('This function is deprecated, use `compositive_warp_inverse` from warputils.py instead')

    permute_vtoimg = (0, 4, 1, 2, 3) if len(warp.shape) == 5 else (0, 3, 1, 2)
    permute_imgtov = (0, 2, 3, 4, 1) if len(warp.shape) == 5 else (0, 2, 3, 1)
    # in case this block is being called within a no_grad block 
    if warp_gaussian is not None:
        warp = separable_filtering(warp.permute(*permute_vtoimg), warp_gaussian).permute(*permute_imgtov)

    with torch.set_grad_enabled(True):
        if initial_inverse is None:
            invwarp = nn.Parameter(torch.zeros_like(warp.detach()))
        else:
            invwarp = nn.Parameter(initial_inverse.detach())

        if optimizer == 'SGD':
            optim = torch.optim.SGD([invwarp], lr=lr)
        elif optimizer == 'Adam':
            optim = torch.optim.Adam([invwarp], lr=lr)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        
        print("using optimizer lr ", optimizer, lr)

        for i in range(iters):
            optim.zero_grad()
            loss = invwarp + F.grid_sample(warp.permute(*permute_vtoimg), grid + invwarp, mode='bilinear', align_corners=True).permute(*permute_imgtov)
            loss2 = warp + F.grid_sample(invwarp.permute(*permute_vtoimg), grid + warp, mode='bilinear', align_corners=True).permute(*permute_imgtov)
            loss = (loss**2).sum() + (loss2**2).sum()
            if debug:
                print(f"loss: {loss.item()}")
            loss.backward()
            # apply gaussian smoothing to the gradient
            if grad_gaussian is not None:
                invwarp.grad = separable_filtering(invwarp.grad.permute(*permute_vtoimg), warp_gaussian).permute(*permute_imgtov)
            optim.step()
            # if warp_gaussian is not None:
            #     with torch.no_grad():
            #         invwarp.data = separable_filtering(invwarp.permute(*permute_vtoimg), warp_gaussian).permute(*permute_imgtov)
    return invwarp.data

def compute_inverse_warp_exp(warp, grid, lr=5e-3, iters=200, n=10):
    ''' compute warp inverse using exponential map '''

    warnings.warn('This function is deprecated, use `compositive_warp_inverse` from warputils.py instead')
    with torch.set_grad_enabled(True):
        vel = nn.Parameter(torch.zeros_like(warp))
        optim = torch.optim.Adam([vel], lr=lr)
        permute_vtoimg = (0, 4, 1, 2, 3) if len(warp.shape) == 5 else (0, 3, 1, 2)
        permute_imgtov = (0, 2, 3, 4, 1) if len(warp.shape) == 5 else (0, 2, 3, 1)
        # pbar = tqdm(range(iters))
        pbar = range(iters)
        for i in pbar:
            optim.zero_grad()
            invwarp = scaling_and_squaring(vel, grid, n=n)
            loss = invwarp + F.grid_sample(warp.permute(*permute_vtoimg), grid + invwarp, mode='bilinear', align_corners=True).permute(*permute_imgtov)
            loss2 = warp + F.grid_sample(invwarp.permute(*permute_vtoimg), grid + warp, mode='bilinear', align_corners=True).permute(*permute_imgtov)
            loss = (loss**2).sum() + (loss2**2).sum()
            loss.backward()
            optim.step()
    return scaling_and_squaring(vel.data, grid, n=n)

### Other itk like filters
class LaplacianFilter(nn.Module):
    def __init__(self, dims, device=None, spacing=None, itk_scale=True, learning_rate=1):
        super().__init__()
        self.dims = dims
        self.itk_scale = itk_scale
        self.learning_rate = learning_rate
        assert dims in [2, 3] 
        if spacing is None:
            spacing = [1.0] * dims
        self.spacing = torch.tensor(spacing, dtype=torch.float32)   # save spacing
        if dims == 2:
            laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32) / 4
        else:
            laplacian = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                      [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                      [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=torch.float32) / 6
        self.laplacian = laplacian[None, None]  # [1, 1, n, n, [n]]
        if device is not None:
            self.laplacian = self.laplacian.to(device)
        self.device = device
    
    def _scale_image(self, image):
        imgflat = image.flatten(2)
        meta = {
            'mean': imgflat.mean(2),
            'min': imgflat.min(2).values,
            'max': imgflat.max(2).values
        }
        for k in meta:
            meta[k] = meta[k][..., None, None]
            if self.dims == 3:
                meta[k] = meta[k][..., None]
        return meta
    
    def forward(self, image: torch.Tensor, itk_scale=None, learning_rate=None):
        ''' forward pass 
        input: tensor of size [B, C, D, H, [W]]
        '''
        C = image.shape[1]
        if self.device is None:
            self.laplacian = self.laplacian.to(image.device)

        if self.dims == 2:
            lap = self.laplacian.expand(C, -1, -1, -1)
            lap_image = F.pad(image, (1, 1, 1, 1), 'replicate')
            lap_image = F.conv2d(lap_image, lap, padding=0, groups=C)
        else:
            lap = self.laplacian.expand(C, -1, -1, -1, -1)
            lap_image = F.pad(image, (1, 1, 1, 1, 1, 1), 'replicate')
            lap_image = F.conv3d(lap_image, lap, padding=0, groups=C)

        itk_scale = itk_scale if itk_scale is not None else self.itk_scale
        learning_rate = learning_rate if learning_rate is not None else self.learning_rate

        if itk_scale:        
            lap_meta = self._scale_image(lap_image)
            image_meta = self._scale_image(image)
            # scale the laplacian image
            lap_image = (lap_image - lap_meta['min']) / (lap_meta['max'] - lap_meta['min']) * (image_meta['max'] - image_meta['min']) 
            scaled_image = image - learning_rate * lap_image
            # scale it again
            scaled_meta = self._scale_image(scaled_image)
            scaled_image = scaled_image - scaled_meta['min']
            return scaled_image
        else:
            return image - learning_rate * lap_image
