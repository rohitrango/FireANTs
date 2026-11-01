'''
MIND-SSC: modality independent neighbourhood descriptor with a self-similarity context.

References:
    [1] M. P. Heinrich, M. Jenkinson, M. Bhushan, T. Matin, F. V. Gleeson, M. Brady, J. A. Schnabel.
        "MIND: Modality independent neighbourhood descriptor for multi-modal deformable registration."
        Medical Image Analysis 16(7):1423-1435, 2012.
    [2] M. P. Heinrich, M. Jenkinson, B. W. Papiez, M. Brady, J. A. Schnabel.
        "Towards realtime multimodal fusion for image-guided interventions using self-similarities."
        MICCAI 2013, LNCS 8149, pp. 187-194.
'''
from itertools import combinations
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from fireants.types import ItemOrList
from fireants.losses.maskedutils import POSSIBLE_MASKED_MODES, DEFAULT_MASK_MODE, get_tensors_and_mask, mask_loss_function


def ssc_offset_pairs(spatial_dims: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    ''' pairs of the 2*D axis-aligned unit offsets that are sqrt(2) apart: 4 pairs in 2D, 12 in 3D '''
    units = []
    for d in range(spatial_dims):
        for sign in (-1, 1):
            u = [0] * spatial_dims
            u[d] = sign
            units.append(tuple(u))
    return [(a, b) for a, b in combinations(units, 2) if sum((i - j) ** 2 for i, j in zip(a, b)) == 2]


def _box_filter(x: torch.Tensor, radius: int, spatial_dims: int) -> torch.Tensor:
    ''' mean over a (2r+1)^D window with replicate padding, as D separable 1D average pools '''
    if radius == 0:
        return x
    pool = F.avg_pool2d if spatial_dims == 2 else F.avg_pool3d
    for d in range(spatial_dims):
        pad = [0] * (2 * spatial_dims)
        pad[2 * (spatial_dims - 1 - d)] = pad[2 * (spatial_dims - 1 - d) + 1] = radius  # F.pad lists the last dim first
        kernel = [1] * spatial_dims
        kernel[d] = 2 * radius + 1
        x = pool(F.pad(x, pad, mode='replicate'), kernel_size=kernel, stride=1)
    return x


def ssc_patch_distances(image: torch.Tensor, radius: int = 2, dilation: int = 2) -> torch.Tensor:
    '''Patch distances of the self-similarity context: [B, C, *spatial] -> [B, C, K, *spatial].

    For each pair (a, b) from `ssc_offset_pairs`,
    D_ab(x) = mean_{p in [-radius, radius]^D} (I(x + dilation*a + p) - I(x + dilation*b + p))^2,
    using replicate padding at the image border.
    '''
    spatial_dims = image.ndim - 2
    if spatial_dims not in (2, 3):
        raise ValueError(f"MIND-SSC supports 2D and 3D images ([B, C, *spatial]), got shape {tuple(image.shape)}")
    shape = image.shape[2:]
    padded = F.pad(image, [dilation] * (2 * spatial_dims), mode='replicate')

    def shifted(offset):
        return padded[(..., *[slice(dilation * (1 + o), dilation * (1 + o) + n) for o, n in zip(offset, shape)])]

    pairs = ssc_offset_pairs(spatial_dims)
    dist = torch.stack([(shifted(a) - shifted(b)).square() for a, b in pairs], dim=2)  # [B, C, K, *spatial]
    C, K = dist.shape[1], dist.shape[2]
    return _box_filter(dist.flatten(1, 2), radius, spatial_dims).unflatten(1, (C, K))


def mind_ssc(image: torch.Tensor, radius: int = 2, dilation: int = 2, var_floor: float = 1e-3) -> torch.Tensor:
    '''Compute the MIND-SSC descriptor of a 2D or 3D image.

    With the patch distances D_k(x) of `ssc_patch_distances` (k runs over the K neighbour pairs),

        MIND_k(x) = exp(-(D_k(x) - min_j D_j(x)) / V(x)),   V(x) = mean_j D_j(x),

    i.e. exp(-D/V) divided by its maximum over the pairs [1], so the largest entry is 1. V is floored at
    `var_floor` times its spatial mean (per image and channel) to avoid dividing by ~0 in flat regions.
    [2] describes the 3D case (12 pairs); in 2D the same construction on the 4-neighbourhood gives 4 pairs.
    The convexAdam implementation (github.com/multimodallearning/convexAdam) computes the same patch distances
    but estimates V as mean_j (D_j - min_i D_i), which sharpens the descriptor.

    Args:
        image: [B, C, H, W] or [B, C, H, W, D]. Each channel gets its own descriptor.
        radius: patch radius in voxels (box filter of size 2*radius+1). 0 compares single voxels.
        dilation: distance in voxels of the neighbourhood points from the centre.
        var_floor: relative floor for the local variance estimate V.

    Returns:
        [B, C*K, *spatial] descriptor with entries in (0, 1], K = 4 (2D) or 12 (3D).
        float16/bfloat16 inputs are computed and returned in float32.
    '''
    if image.dtype in (torch.float16, torch.bfloat16):
        image = image.float()
    dist = ssc_patch_distances(image, radius, dilation)
    var = dist.mean(dim=2, keepdim=True)
    spatial = tuple(range(3, dist.ndim))
    floor = var_floor * var.mean(dim=spatial, keepdim=True).detach() + torch.finfo(dist.dtype).tiny
    var = torch.maximum(var, floor)
    mind = torch.exp(-(dist - dist.amin(dim=2, keepdim=True)) / var)
    return mind.flatten(1, 2)


class MINDSSCLoss(nn.Module):
    '''Distance between the MIND-SSC descriptors [1, 2] of the moved and the fixed image.

    MIND-SSC describes each voxel by how similar its neighbouring patches are to one another, so it depends on
    local structure rather than on the intensity mapping. It is exactly invariant to affine intensity changes
    (I -> a*I + b, including contrast inversion a < 0) and approximately invariant to smooth monotone ones,
    which makes it suited to multi-modal deformable registration (e.g. T1 to T2 MRI, MR to CT).

    The descriptor of the moved image is recomputed at every iteration (it is not warped), so the gradient is
    exact. With `cache_target=True` the fixed descriptor is reused while the same fixed tensor is passed in and
    does not require gradients (Greedy, affine and rigid registration); SyN warps both images, so both are recomputed.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        radius: patch radius in voxels. An int, or a list with one entry per registration scale.
        dilation: neighbourhood distance in voxels. An int, or a list with one entry per registration scale.
        distance: ``"l2"`` (squared difference, default) or ``"l1"`` (absolute difference) between descriptors.
        reduction: ``"mean"``, ``"sum"`` or ``"none"`` (``"none"`` returns the per-voxel loss, [B, 1, *spatial]).
        var_floor: relative floor for the local variance estimate (see `mind_ssc`).
        cache_target: reuse the fixed image descriptor across iterations when possible.
        masked: if True, the last channel of both inputs is a mask (set by ``loss_type='masked_mind'``).
        mask_mode: how the two masks are combined, ``"mult"`` or ``"max"``.

    The loss is 0 when the descriptors agree and lies in [0, 1].
    '''

    def __init__(
        self,
        spatial_dims: int = 3,
        radius: ItemOrList[int] = 2,
        dilation: ItemOrList[int] = 2,
        distance: str = "l2",
        reduction: str = "mean",
        var_floor: float = 1e-3,
        cache_target: bool = True,
        masked: bool = False,
        mask_mode: str = DEFAULT_MASK_MODE,
    ) -> None:
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"MINDSSCLoss supports spatial_dims 2 or 3, got {spatial_dims}")
        if distance not in ("l1", "l2"):
            raise ValueError(f"distance must be 'l1' or 'l2', got {distance}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction}")
        if masked and reduction == "none":
            raise ValueError("masked MINDSSCLoss needs reduction 'mean' or 'sum'")
        assert mask_mode in POSSIBLE_MASKED_MODES
        self.spatial_dims = spatial_dims
        self.radius_list = list(radius) if isinstance(radius, (list, tuple)) else None
        self.dilation_list = list(dilation) if isinstance(dilation, (list, tuple)) else None
        self.radius = self.radius_list[0] if self.radius_list else radius
        self.dilation = self.dilation_list[0] if self.dilation_list else dilation
        for r in self.radius_list or [self.radius]:
            if int(r) != r or r < 0:
                raise ValueError(f"radius must be a non-negative integer, got {r}")
        for d in self.dilation_list or [self.dilation]:
            if int(d) != d or d < 1:
                raise ValueError(f"dilation must be a positive integer, got {d}")
        self.distance = distance
        self.reduction = reduction
        self.var_floor = float(var_floor)
        self.cache_target = cache_target
        self.masked = masked
        self.mask_mode = mask_mode
        self.scales = None
        self.iterations = None
        self._cache = None

    # multi-scale hooks called by the registration classes
    def set_scales(self, scales: List) -> None:
        self.scales = scales
        self._cache = None
        for name, values in (("radius", self.radius_list), ("dilation", self.dilation_list)):
            if values is not None:
                assert len(values) == len(scales), f"{name} list must have one entry per scale, got {len(values)} vs {len(scales)}"

    def set_iterations(self, iterations: List[int]) -> None:
        self.iterations = iterations

    def set_current_scale_and_iterations(self, scale, iters: int) -> None:
        self._cache = None
        if self.scales is None:
            return
        idx = self.scales.index(scale)
        if self.radius_list:
            self.radius = self.radius_list[idx]
        if self.dilation_list:
            self.dilation = self.dilation_list[idx]

    def get_image_padding(self) -> int:
        return self.radius + self.dilation

    def _descriptor(self, image: torch.Tensor) -> torch.Tensor:
        return mind_ssc(image, radius=self.radius, dilation=self.dilation, var_floor=self.var_floor)

    def _target_descriptor(self, target: torch.Tensor) -> torch.Tensor:
        if not self.cache_target or target.requires_grad:
            return self._descriptor(target)
        key = (target._version, self.radius, self.dilation, self.var_floor)
        if self._cache is not None and self._cache[0] is target and self._cache[1] == key:
            return self._cache[2]
        with torch.no_grad():
            desc = self._descriptor(target)
        self._cache = (target, key, desc)  # holding `target` keeps the identity check valid
        return desc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim - 2 != self.spatial_dims:
            raise ValueError(f"expected pred with {self.spatial_dims} spatial dimensions, got shape {tuple(pred.shape)}")
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}")
        mask = None
        if self.masked:
            pred, target, mask = get_tensors_and_mask(pred, target, self.mask_mode)
        diff = self._descriptor(pred) - self._target_descriptor(target)
        loss = (diff.square() if self.distance == "l2" else diff.abs()).mean(dim=1, keepdim=True)  # [B, 1, *spatial]
        if mask is not None:
            return mask_loss_function(loss, mask.to(loss), self.reduction)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
