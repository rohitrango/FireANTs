'''
Similarity in the feature space of a frozen, user-supplied encoder.
'''
from itertools import chain
from typing import List, Optional, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F

from fireants.losses.maskedutils import POSSIBLE_MASKED_MODES, DEFAULT_MASK_MODE, get_tensors_and_mask, mask_loss_function


class FeatureSimilarityLoss(nn.Module):
    '''Compare the moved and fixed images through a frozen encoder ``E``: ``loss = d(E(moved), E(fixed))``.

    Any ``nn.Module`` that maps ``[B, C, *spatial]`` to a dense feature map ``[B, F, *spatial']`` works, e.g. a
    pretrained modality-agnostic network or a hand-crafted filter bank. The encoder's weights are frozen
    (``requires_grad_(False)``, kept in eval mode); gradients flow through it to the moved image only.
    No weights are shipped: pass your own encoder and use it with ``loss_type='custom'``.

    Args:
        encoder: an ``nn.Module``, or a list of them with one entry per registration scale. It may return a
            single feature map or a list/tuple of maps (e.g. several decoder levels); the per-level losses are
            then averaged with `level_weights`.
        distance: ``"cosine"`` (default): ``1 - cos(E(moved), E(fixed))`` along the feature dimension, per voxel.
            ``"l2"``: squared Euclidean distance between the feature vectors, per voxel. With `normalize=True`
            the vectors are L2-normalised first, so ``"l2" == 2 * "cosine"``.
        normalize: L2-normalise feature vectors before the ``"l2"`` distance (``"cosine"`` is scale-free anyway).
        level_weights: weights for multi-level encoder outputs. Defaults to equal weights.
        reduction: ``"mean"``, ``"sum"`` or ``"none"`` (``"none"`` returns the per-voxel loss of a single-level encoder).
        cache_target: reuse the fixed image features while the same fixed tensor is passed in and does not require
            gradients (Greedy, affine and rigid registration). SyN warps both images, so both are recomputed.
        masked: if True, the last channel of both inputs is a mask (it is not passed to the encoder); the mask is
            resized to each feature map with area interpolation.
        mask_mode: how the two masks are combined, ``"mult"`` or ``"max"``.
        eps: lower bound on feature-vector norms in the cosine and the normalisation.

    The encoder sees the images at each pyramid scale as FireANTs downsamples them; pass one encoder per scale
    if it should differ. The encoder is moved to the images' device on first use, and the images are cast to
    its parameter dtype. Normalise intensities the way the encoder expects (e.g. wrap it in ``nn.Sequential``).
    '''

    def __init__(
        self,
        encoder: Union[nn.Module, Sequence[nn.Module]],
        distance: str = "cosine",
        normalize: bool = True,
        level_weights: Optional[Sequence[float]] = None,
        reduction: str = "mean",
        cache_target: bool = True,
        masked: bool = False,
        mask_mode: str = DEFAULT_MASK_MODE,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        encoders = list(encoder) if isinstance(encoder, (list, tuple, nn.ModuleList)) else [encoder]
        if not all(isinstance(e, nn.Module) for e in encoders):
            raise TypeError("encoder must be an nn.Module or a list of nn.Modules")
        if distance not in ("cosine", "l2"):
            raise ValueError(f"distance must be 'cosine' or 'l2', got {distance}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction}")
        if masked and reduction == "none":
            raise ValueError("masked FeatureSimilarityLoss needs reduction 'mean' or 'sum'")
        assert mask_mode in POSSIBLE_MASKED_MODES
        self.encoders = nn.ModuleList(encoders)
        for e in self.encoders:
            e.requires_grad_(False)
        self.train(False)
        self.per_scale = len(encoders) > 1
        self.encoder_idx = 0
        self.distance = distance
        self.normalize = normalize
        self.level_weights = level_weights
        self.reduction = reduction
        self.cache_target = cache_target
        self.masked = masked
        self.mask_mode = mask_mode
        self.eps = eps
        self.scales = None
        self.iterations = None
        self._cache = None

    def train(self, mode: bool = True):
        ''' the encoder is frozen: keep it in eval mode (batchnorm statistics, dropout) '''
        super().train(mode)
        for e in self.encoders:
            e.eval()
        return self

    # multi-scale hooks called by the registration classes
    def set_scales(self, scales: List) -> None:
        self.scales = scales
        self._cache = None
        if self.per_scale:
            assert len(self.encoders) == len(scales), f"need one encoder per scale, got {len(self.encoders)} vs {len(scales)}"

    def set_iterations(self, iterations: List[int]) -> None:
        self.iterations = iterations

    def set_current_scale_and_iterations(self, scale, iters: int) -> None:
        self._cache = None
        if self.per_scale and self.scales is not None:
            self.encoder_idx = self.scales.index(scale)

    def _encode(self, image: torch.Tensor) -> List[torch.Tensor]:
        encoder = self.encoders[self.encoder_idx]
        param = next(chain(encoder.parameters(), encoder.buffers()), None)
        if param is not None:
            if param.device != image.device:
                encoder.to(image.device)
            image = image.to(param.dtype)
        feats = encoder(image)
        feats = list(feats) if isinstance(feats, (list, tuple)) else [feats]
        for f in feats:
            if f.ndim != image.ndim:
                raise ValueError(f"encoder output must be [B, F, *spatial] like its input {tuple(image.shape)}, got {tuple(f.shape)}")
        return feats

    def _target_features(self, target: torch.Tensor) -> List[torch.Tensor]:
        if not self.cache_target or target.requires_grad:
            return self._encode(target)
        key = (target._version, self.encoder_idx)
        if self._cache is not None and self._cache[0] is target and self._cache[1] == key:
            return self._cache[2]
        with torch.no_grad():
            feats = self._encode(target)
        self._cache = (target, key, feats)  # holding `target` keeps the identity check valid
        return feats

    def _voxel_distance(self, fp: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        ''' [B, F, *spatial] x 2 -> [B, 1, *spatial] '''
        if self.distance == "cosine":
            return 1 - F.cosine_similarity(fp, ft, dim=1, eps=self.eps).unsqueeze(1)
        if self.normalize:
            fp = F.normalize(fp, dim=1, eps=self.eps)
            ft = F.normalize(ft, dim=1, eps=self.eps)
        return (fp - ft).square().sum(dim=1, keepdim=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}")
        mask = None
        if self.masked:
            pred, target, mask = get_tensors_and_mask(pred, target, self.mask_mode)
        feats_p, feats_t = self._encode(pred), self._target_features(target)
        weights = self.level_weights if self.level_weights is not None else [1.0] * len(feats_p)
        if len(weights) != len(feats_p):
            raise ValueError(f"got {len(weights)} level_weights for {len(feats_p)} feature levels")
        if self.reduction == "none" and len(feats_p) > 1:
            raise ValueError("reduction='none' needs an encoder with a single feature output")
        total = 0
        for w, fp, ft in zip(weights, feats_p, feats_t):
            if fp.shape != ft.shape:
                raise ValueError(f"feature shapes differ: {tuple(fp.shape)} vs {tuple(ft.shape)}")
            loss = self._voxel_distance(fp, ft)
            if mask is not None:
                m = F.interpolate(mask.to(loss), size=loss.shape[2:], mode="area")
                loss = mask_loss_function(loss, m, self.reduction)
            elif self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()
            total = total + w * loss
        return total / sum(weights) if self.reduction == "mean" else total
