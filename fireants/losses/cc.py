'''
Cross correlation
'''
from time import time, sleep
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
from fireants.types import ItemOrList

# @torch.jit.script
def gaussian_1d(
    sigma: torch.Tensor, truncated: float = 4.0, approx: str = "erf", normalize: bool = True
) -> torch.Tensor:
    """
    one dimensional Gaussian kernel.
    Args:
        sigma: std of the kernel
        truncated: tail length
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            - ``erf`` approximation interpolates the error function;
            - ``sampled`` uses a sampled Gaussian kernel;
            - ``scalespace`` corresponds to
              https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
              based on the modified Bessel functions.
        normalize: whether to normalize the kernel with `kernel.sum()`.
    Raises:
        ValueError: When ``truncated`` is non-positive.
    Returns:
        1D torch tensor
    """
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device = sigma.device
        dtype = sigma.dtype

    sigma = torch.as_tensor(sigma, dtype=dtype , device=device)
    if truncated <= 0.0:
        raise ValueError(f"truncated must be positive, got {truncated}.")
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    if approx.lower() == "erf":
        x = torch.arange(-tail, tail + 1, dtype=dtype, device=device)
        t = 0.70710678 / torch.abs(sigma)
        out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
        out = out.clamp(min=0)
    elif approx.lower() == "sampled":
        x = torch.arange(-tail, tail + 1, dtype=dtype, device=device)
        out = torch.exp(-0.5 / (sigma * sigma) * x**2)
        if not normalize:  # compute the normalizer
            out = out / (2.5066282 * sigma)
    else:
        raise NotImplementedError(f"Unsupported option: approx='{approx}'.")
    return out / out.sum() if normalize else out  # type: ignore

def make_rectangular_kernel(kernel_size: int) -> torch.Tensor:
    return torch.ones(kernel_size)

def make_triangular_kernel(kernel_size: int) -> torch.Tensor:
    fsize = (kernel_size + 1) // 2
    if fsize % 2 == 0:
        fsize -= 1
    f = torch.ones((1, 1, fsize), dtype=torch.float).div(fsize)
    padding = (kernel_size - fsize) // 2 + fsize // 2
    return F.conv1d(f, f, padding=padding).reshape(-1)

def make_gaussian_kernel(kernel_size: int) -> torch.Tensor:
    sigma = torch.tensor(kernel_size / 3.0)
    kernel = gaussian_1d(sigma=sigma, truncated=(kernel_size // 2) * 1.0, approx="sampled", normalize=False) * (
        2.5066282 * sigma
    )
    return kernel[:kernel_size]

def _separable_filtering_conv(
    input_: torch.Tensor,
    kernels: List[torch.Tensor],
    pad_mode: str,
    spatial_dims: int,
    paddings: List[int],
    num_channels: int,
) -> torch.Tensor:

    # re-write from recursive to non-recursive for torch.jit to work
    # for d in range(spatial_dims-1, -1, -1):
    for d in range(spatial_dims):
        s = [1] * len(input_.shape)
        s[d + 2] = -1
        _kernel = kernels[d].reshape(s)
        # if filter kernel is unity, don't convolve
        if _kernel.numel() == 1 and _kernel[0] == 1:
            continue

        _kernel = _kernel.repeat([num_channels, 1] + [1] * spatial_dims)
        # _padding = [0] * spatial_dims
        # _padding[d] = paddings[d]
        # _reversed_padding = _padding[::-1]

        # # translate padding for input to torch.nn.functional.pad
        # _reversed_padding_repeated_twice: list[list[int]] = [[p, p] for p in _reversed_padding]
        # _sum_reversed_padding_repeated_twice: list[int] = []
        # for p in _reversed_padding_repeated_twice:
        #     _sum_reversed_padding_repeated_twice.extend(p)
        # _sum_reversed_padding_repeated_twice: list[int] = sum(_reversed_padding_repeated_twice, [])

        # padded_input = F.pad(input_, _sum_reversed_padding_repeated_twice, mode=pad_mode)
        # dont waste time in padding, let conv do it
        padded_input = input_
        # print("padded_input", padded_input.shape)
        # update input
        if spatial_dims == 1:
            input_ = F.conv1d(input=padded_input, weight=_kernel, groups=num_channels, padding='same')
        elif spatial_dims == 2:
            input_ = F.conv2d(input=padded_input, weight=_kernel, groups=num_channels, padding='same')
        elif spatial_dims == 3:
            input_ = F.conv3d(input=padded_input, weight=_kernel, groups=num_channels, padding='same')
        else:
            raise NotImplementedError(f"Unsupported spatial_dims: {spatial_dims}.")
    return input_

def separable_filtering(x: torch.Tensor, kernels: ItemOrList[torch.Tensor], mode: str = "zeros", use_separable_override: bool = True) -> torch.Tensor:
    """
    Apply 1-D convolutions along each spatial dimension of `x`.
    Args:
        x: the input image. must have shape (batch, channels, H[, W, ...]).
        kernels: kernel along each spatial dimension.
            could be a single kernel (duplicated for all spatial dimensions), or
            a list of `spatial_dims` number of kernels.
        mode (string, optional): padding mode passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``. See ``torch.nn.Conv1d()`` for more information.
    Raises:
        TypeError: When ``x`` is not a ``torch.Tensor``.
    Examples:
    .. code-block:: python
        >>> import torch
        >>> img = torch.randn(2, 4, 32, 32)  # batch_size 2, channels 4, 32x32 2D images
        # applying a [-1, 0, 1] filter along each of the spatial dimensions.
        # the output shape is the same as the input shape.
        >>> out = separable_filtering(img, torch.tensor((-1., 0., 1.)))
        # applying `[-1, 0, 1]`, `[1, 0, -1]` filters along two spatial dimensions respectively.
        # the output shape is the same as the input shape.
        >>> out = separable_filtering(img, [torch.tensor((-1., 0., 1.)), torch.tensor((1., 0., -1.))])
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")

    spatial_dims = len(x.shape) - 2
    if isinstance(kernels, torch.Tensor):
        kernels = [kernels] * spatial_dims
    
    # run one conv if kernel is small
    _kernels = [s.to(x) for s in kernels]

    # run one conv if kernel is small
    if _kernels[0].numel() < 15 and not use_separable_override:
        # create conv
        if spatial_dims == 2:
            kernel = _kernels[0].reshape(1, 1, -1, 1) * _kernels[1].reshape(1, 1, 1, -1)
            kernel = kernel.to(x).repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, padding='same', groups=x.shape[1])
        elif spatial_dims == 3:
            kernel = _kernels[0].reshape(1, 1, -1, 1, 1) * _kernels[1].reshape(1, 1, 1, -1, 1) * _kernels[2].reshape(1, 1, 1, 1, -1)
            kernel = kernel.to(x).repeat(x.shape[1], 1, 1, 1, 1)
            return F.conv3d(x, kernel, padding='same', groups=x.shape[1])
        else:
            raise NotImplementedError(f"Unsupported spatial_dims: {spatial_dims}.")

    _paddings = [(k.shape[0] - 1) // 2 for k in _kernels]
    n_chs = x.shape[1]
    pad_mode = "constant" if mode == "zeros" else mode
    return _separable_filtering_conv(x, _kernels, pad_mode, spatial_dims, _paddings, n_chs)


# dict
kernel_dict = {
    "rectangular": make_rectangular_kernel,
    "triangular": make_triangular_kernel,
    "gaussian": make_gaussian_kernel,
}

class LocalNormalizedCrossCorrelationLoss(nn.Module):
    """
    Local squared zero-normalized cross-correlation.
    The loss is based on a moving kernel/window over the y_true/y_pred,
    within the window the square of zncc is calculated.
    The kernel can be a rectangular / triangular / gaussian window.
    The final loss is the averaged loss over all windows.
    Adapted from:
        https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: int = 3,
        kernel_type: str = "rectangular",
        reduction: str = "mean",
        smooth_nr: float = 0,
        smooth_dr: float = 1e-5,
        unsigned: bool = True,
        use_separable_override: bool = True,
        checkpointing: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel spatial size, must be odd.
            kernel_type: {``"rectangular"``, ``"triangular"``, ``"gaussian"``}. Defaults to ``"rectangular"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
            split: do we want to split computation across 2 GPUs? (if pred and target are on different GPUs)
                default: False (assumes they are on same device and big enough to fit on one GPU)
        """
        super().__init__()
        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")
        self.reduction = reduction
        self.unsigned = unsigned

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        # _kernel = look_up_option(kernel_type, kernel_dict)
        _kernel = kernel_dict[kernel_type]
        self.kernel = _kernel(self.kernel_size)
        self.kernel.requires_grad = False
        self.kernel_nd, self.kernel_vol = self.get_kernel_vol()   # get nD kernel and its volume
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.checkpointing = checkpointing
        self.use_separable_override = use_separable_override

    def get_kernel_vol(self):
        vol = self.kernel
        for _ in range(self.ndim - 1):
            vol = torch.matmul(vol.unsqueeze(-1), self.kernel.unsqueeze(0))
        return vol, torch.sum(vol)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if pred.ndim - 2 != self.ndim:
            raise ValueError(f"expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}")
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")

        # sum over kernel
        def cc_checkpoint_fn(target, pred, kernel, kernel_vol, checkpointing=False):
            '''
            This function is used to compute the intermediate results of the loss.
            '''
            t2, p2, tp = target * target, pred * pred, target * pred
            kernel, kernel_vol = kernel.to(pred), kernel_vol.to(pred)
            # kernel_nd = self.kernel_nd.to(pred)
            kernels = [kernel] * self.ndim
            kernels_t = kernels_p = kernels
            kernel_vol_t = kernel_vol_p = kernel_vol
            # compute intermediates
            def sum_filter(target, kernels_t):
                t_sum = separable_filtering(target, kernels=kernels_t, use_separable_override=self.use_separable_override)
                return t_sum
            
            if checkpointing:
                t_sum = checkpoint(sum_filter, target, kernels_t, use_reentrant=False)
                p_sum = checkpoint(sum_filter, pred, kernels_p, use_reentrant=False)
            else:
                t_sum = sum_filter(target, kernels_t)
                p_sum = sum_filter(pred, kernels_p)

            if checkpointing:
                t2_sum = checkpoint(sum_filter, t2, kernels_t, use_reentrant=False)
                p2_sum = checkpoint(sum_filter, p2, kernels_p, use_reentrant=False)
                tp_sum = checkpoint(sum_filter, tp, kernels_t, use_reentrant=False)
            else:
                t2_sum = sum_filter(t2, kernels_t)
                p2_sum = sum_filter(p2, kernels_p)
                tp_sum = sum_filter(tp, kernels_t)

            # the following is actually squared ncc
            def cross_filter(tp_sum, p_sum, t_sum, kernel_vol):
                return tp_sum.to(pred) - p_sum * t_sum.to(pred)/kernel_vol  # on pred device
            
            if checkpointing:
                cross = checkpoint(cross_filter, tp_sum, p_sum, t_sum, kernel_vol, use_reentrant=False)
            else:
                cross = cross_filter(tp_sum, p_sum, t_sum, kernel_vol)
            
            def var_filter(t2sum, tsum, kernel_vol):
                return torch.max(
                    t2sum - tsum * tsum / kernel_vol, torch.as_tensor(self.smooth_dr, dtype=t2sum.dtype, device=t2sum.device)
                ).to(pred)
            
            if checkpointing:
                t_var = checkpoint(var_filter, t2_sum, t_sum, kernel_vol, use_reentrant=False)
                p_var = checkpoint(var_filter, p2_sum, p_sum, kernel_vol, use_reentrant=False)
            else:
                t_var = var_filter(t2_sum, t_sum, kernel_vol)
                p_var = var_filter(p2_sum, p_sum, kernel_vol)

            if self.unsigned:
                def ncc_filter(cross, t_var, p_var):
                    ncc: torch.Tensor = (cross * cross + self.smooth_nr) / ((t_var * p_var) + self.smooth_dr)
                    return ncc
                if checkpointing:
                    ncc = checkpoint(ncc_filter, cross, t_var, p_var, use_reentrant=False)
                else:
                    ncc = ncc_filter(cross, t_var, p_var)
            else:
                def ncc_filter(cross, t_var, p_var):
                    ncc: torch.Tensor = (cross + self.smooth_nr) / ((torch.sqrt(t_var) * torch.sqrt(p_var)) + self.smooth_dr)
                    return ncc

                if checkpointing:
                    ncc = checkpoint(ncc_filter, cross, t_var, p_var, use_reentrant=False)
                else:
                    ncc = ncc_filter(cross, t_var, p_var)
            return ncc
        
        ncc = cc_checkpoint_fn(target, pred, self.kernel, self.kernel_vol, checkpointing=self.checkpointing)
        # clamp (dont really need this because the offending pixels are very sparse)
        ncc = ncc.clamp(min=-1, max=1)

        if self.reduction == 'sum':
            return torch.sum(ncc).neg()  # sum over the batch, channel and spatial ndims
        if self.reduction == 'none':
            return ncc.neg()
        if self.reduction == 'mean':
            return torch.mean(ncc).neg()  # average over the batch, channel and spatial ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


if __name__ == '__main__':
    N = 160  
    img1 = torch.rand(1, 1, N, N, N).cuda()
    img2 = torch.rand(1, 1, N, N, N).cuda()
    # loss = torch.jit.script(LocalNormalizedCrossCorrelationLoss(3, kernel_type='rectangular', reduction='mean')).cuda()
    mem  = torch.cuda.memory_allocated()

    for use_jit_version, separable_override in [(True, True), (True, False), (False, True), (False, False)]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated()
        # 
        loss = LocalNormalizedCrossCorrelationLoss(3, kernel_type='rectangular', reduction='mean', use_separable_override=separable_override).cuda()
        if use_jit_version:
            loss = torch.compile(loss)
        total = 0
        a = time()
        for i in range(20):
            out = loss(img1, img2)
            total += out.item()
        b = time()
        print(f"Time for jit: {use_jit_version} separable: {separable_override} time: {b - a}s")
        print(f"Total loss: {total}")
        mem = torch.cuda.max_memory_allocated() - mem
        print(f"Memory for jit: {use_jit_version} separable: {separable_override} memory: {mem / 1024 / 1024} MB\n")
        print()
