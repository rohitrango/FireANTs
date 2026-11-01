"""Tests for the multimodal similarity losses: MINDSSCLoss (loss_type='mind') and FeatureSimilarityLoss.

All tests run on CPU; the device/dtype tests also run on CUDA when it is available.
"""
import itertools

import numpy as np
import pytest
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch import nn

from fireants.io.image import BatchedImages, Image
from fireants.io.imagemask import generate_and_apply_mask_allones
from fireants.losses import FeatureSimilarityLoss, MINDSSCLoss
from fireants.losses.mind import mind_ssc, ssc_offset_pairs, ssc_patch_distances
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration

devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _smooth_noise(shape, seed=0, passes=3, dtype=torch.float32):
    ''' smooth random image [1, 1, *shape] with values in [0, 1] '''
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 1, *shape, generator=g, dtype=torch.float64)
    pool = F.avg_pool2d if len(shape) == 2 else F.avg_pool3d
    for _ in range(passes):
        x = pool(x, 3, stride=1, padding=1, count_include_pad=False)
    x = (x - x.min()) / (x.max() - x.min())
    return x.to(dtype)


def _crop(x, offset, size):
    ''' crop of `x` starting at 4 + offset along the first spatial axis and at 4 along the others '''
    idx = [slice(None), slice(None), slice(4 + offset, 4 + offset + size[0])]
    idx += [slice(4, 4 + s) for s in size[1:]]
    return x[tuple(idx)]


# ---------------------------------------------------------------------------------------------- MIND-SSC
@pytest.mark.parametrize("dims,npairs", [(2, 4), (3, 12)])
def test_ssc_pairs(dims, npairs):
    pairs = ssc_offset_pairs(dims)
    assert len(pairs) == npairs
    for a, b in pairs:   # two different axes, distance sqrt(2), centre not included
        assert sum((i - j) ** 2 for i, j in zip(a, b)) == 2
        assert sum(abs(i) for i in a) == 1 and sum(abs(i) for i in b) == 1


@pytest.mark.parametrize("shape", [(9, 8), (7, 6, 8)])
@pytest.mark.parametrize("radius,dilation", [(0, 1), (1, 1), (1, 2), (2, 3)])
def test_patch_distances_match_brute_force(shape, radius, dilation):
    img = torch.rand(1, 1, *shape, generator=torch.Generator().manual_seed(17), dtype=torch.float64)
    ours = ssc_patch_distances(img, radius, dilation)[0, 0].numpy()
    dims = len(shape)
    patch = list(itertools.product(range(-radius, radius + 1), repeat=dims))
    for k, (a, b) in enumerate(ssc_offset_pairs(dims)):
        for x in itertools.product(*[range(n) for n in shape]):
            ssd = 0.0
            for p in patch:
                # replicate padding = clamping indices: the box filter pads the squared-difference map
                # (clamp the patch position), the shifts pad the image (clamp the shifted position)
                q = [min(max(xi + pi, 0), n - 1) for xi, pi, n in zip(x, p, shape)]
                ia = tuple(min(max(qi + dilation * ai, 0), n - 1) for qi, ai, n in zip(q, a, shape))
                ib = tuple(min(max(qi + dilation * bi, 0), n - 1) for qi, bi, n in zip(q, b, shape))
                ssd += (img[0, 0][ia].item() - img[0, 0][ib].item()) ** 2
            assert ours[(k, *x)] == pytest.approx(ssd / len(patch), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("shape,channels", [((20, 24), 1), ((12, 14, 10), 2)])
def test_descriptor_shape_and_range(shape, channels):
    img = torch.cat([_smooth_noise(shape, seed=s) for s in range(channels)], dim=1).repeat(2, 1, *[1] * len(shape))
    desc = mind_ssc(img, radius=1, dilation=2)
    k = 4 if len(shape) == 2 else 12
    assert desc.shape == (2, channels * k, *shape)
    assert desc.min() > 0 and desc.max() <= 1
    # the largest entry of each descriptor is exactly 1 (normalisation by the maximum)
    per_channel = desc.unflatten(1, (channels, k)).amax(dim=2)
    assert torch.allclose(per_channel, torch.ones_like(per_channel))


@pytest.mark.parametrize("shape", [(40, 40), (24, 24, 24)])
def test_identity_alignment_is_minimum(shape):
    big = _smooth_noise([s + 8 for s in shape], seed=1)
    fixed = _crop(big, 0, shape)
    loss_fn = MINDSSCLoss(spatial_dims=len(shape), radius=1, dilation=1)
    losses = {t: loss_fn(_crop(big, t, shape), fixed).item() for t in range(-3, 4)}
    assert losses[0] == pytest.approx(0.0, abs=1e-12)
    assert all(losses[t] > 1e-4 for t in losses if t != 0)
    assert losses[1] < losses[2] < losses[3] and losses[-1] < losses[-2] < losses[-3]


@pytest.mark.parametrize("shape", [(32, 32), (16, 16, 16)])
@pytest.mark.parametrize("a,b", [(2.5, 0.3), (-1.7, 1.0), (1e-2, -5.0)])
def test_affine_intensity_invariance(shape, a, b):
    img = _smooth_noise(shape, seed=2, dtype=torch.float64)
    assert torch.allclose(mind_ssc(img), mind_ssc(a * img + b), atol=1e-8)
    assert MINDSSCLoss(spatial_dims=len(shape))(a * img + b, img).item() < 1e-12


@pytest.mark.parametrize("shape", [(48, 48), (24, 24, 24)])
def test_monotone_remap_near_invariance(shape):
    ''' smooth non-linear monotone remaps cost much less than a one-voxel misalignment (MIND is only exactly
    invariant to affine remaps; on this image the measured ratios are 0.03-0.10 for l2 and 0.15-0.27 for l1) '''
    big = _smooth_noise([s + 8 for s in shape], seed=3)
    fixed = _crop(big, 0, shape)
    remaps = {
        "gamma": lambda x: x ** 0.5,
        "sigmoid": lambda x: torch.sigmoid(6 * (x - 0.5)),
        "log": lambda x: torch.log1p(9 * x),
    }
    for distance in ("l1", "l2"):
        loss_fn = MINDSSCLoss(spatial_dims=len(shape), radius=1, dilation=1, distance=distance)
        shifted = loss_fn(_crop(big, 1, shape), fixed).item()
        for name, g in remaps.items():
            remapped = loss_fn(g(fixed), fixed).item()
            assert remapped < 0.5 * shifted, f"{name}/{distance}: {remapped:.4g} vs one-voxel shift {shifted:.4g}"


@pytest.mark.parametrize("shape", [(7, 8), (5, 6, 5)])
@pytest.mark.parametrize("distance", ["l1", "l2"])
def test_gradcheck(shape, distance):
    g = torch.Generator().manual_seed(4)
    pred = torch.rand(1, 1, *shape, generator=g, dtype=torch.float64, requires_grad=True)
    target = torch.rand(1, 1, *shape, generator=g, dtype=torch.float64)
    loss_fn = MINDSSCLoss(spatial_dims=len(shape), radius=1, dilation=1, distance=distance)
    assert torch.autograd.gradcheck(lambda p: loss_fn(p, target), (pred,), eps=1e-6, atol=1e-5)


def test_gradient_points_towards_alignment():
    ''' d loss / d translation is positive for positive shifts and negative for negative ones '''
    fixed = _smooth_noise((48, 48), seed=5, dtype=torch.float64)
    loss_fn = MINDSSCLoss(spatial_dims=2, radius=1, dilation=1)
    for shift_px in (-1.5, -0.5, 0.5, 1.5):
        t = torch.tensor(shift_px * 2 / 47, dtype=torch.float64, requires_grad=True)
        theta = torch.stack([torch.stack([torch.ones_like(t), torch.zeros_like(t), t]),
                             torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)])[None]
        grid = F.affine_grid(theta, fixed.shape, align_corners=True)
        moved = F.grid_sample(fixed, grid, align_corners=True, padding_mode="border")
        loss_fn(moved[..., 4:-4, 4:-4], fixed[..., 4:-4, 4:-4]).backward()
        assert np.sign(t.grad.item()) == np.sign(shift_px)


def test_gradient_reaches_both_inputs():
    ''' SyN warps both images, so both receive a gradient '''
    a = _smooth_noise((16, 16, 16), seed=6).requires_grad_(True)
    b = _smooth_noise((16, 16, 16), seed=7).requires_grad_(True)
    MINDSSCLoss(spatial_dims=3)(a, b).backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


def test_scale_lists():
    loss_fn = MINDSSCLoss(radius=[3, 2, 1], dilation=[2, 2, 1])
    loss_fn.set_scales([4, 2, 1])
    loss_fn.set_current_scale_and_iterations(2, 10)
    assert (loss_fn.radius, loss_fn.dilation, loss_fn.get_image_padding()) == (2, 2, 4)
    loss_fn.set_current_scale_and_iterations(1, 10)
    assert (loss_fn.radius, loss_fn.dilation, loss_fn.get_image_padding()) == (1, 1, 2)
    with pytest.raises(AssertionError):
        MINDSSCLoss(radius=[2, 1]).set_scales([4, 2, 1])


def test_masked():
    shape = (24, 24, 24)
    fixed, moving = _smooth_noise(shape, seed=8), _smooth_noise(shape, seed=9)
    plain = MINDSSCLoss(reduction="none")(moving, fixed)
    masked = MINDSSCLoss(masked=True)
    ones = torch.ones_like(fixed)
    assert torch.allclose(masked(torch.cat([moving, ones], 1), torch.cat([fixed, ones], 1)), plain.mean())
    half = torch.zeros_like(fixed)
    half[..., :12, :, :] = 1
    expected = plain[..., :12, :, :].mean()
    assert torch.allclose(masked(torch.cat([moving, half], 1), torch.cat([fixed, ones], 1)), expected)


def test_reductions_and_errors():
    shape = (16, 16, 16)
    fixed, moving = _smooth_noise(shape, seed=10), _smooth_noise(shape, seed=11)
    per_voxel = MINDSSCLoss(reduction="none")(moving, fixed)
    assert per_voxel.shape == (1, 1, *shape)
    assert torch.allclose(MINDSSCLoss(reduction="sum")(moving, fixed), per_voxel.sum())
    with pytest.raises(ValueError):
        MINDSSCLoss()(moving, fixed[..., :-1])          # shape mismatch
    with pytest.raises(ValueError):
        MINDSSCLoss(spatial_dims=2)(moving, fixed)      # wrong number of spatial dims
    with pytest.raises(ValueError):
        MINDSSCLoss(spatial_dims=1)
    with pytest.raises(ValueError):
        MINDSSCLoss(distance="l3")
    with pytest.raises(ValueError):
        MINDSSCLoss(dilation=0)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_device_and_dtype(device, dtype):
    if dtype == torch.bfloat16 and device == "cpu":
        pytest.skip("bfloat16 pooling on CPU is slow and not needed")
    fixed = _smooth_noise((20, 20, 20), seed=12).to(device, dtype)
    moving = _smooth_noise((20, 20, 20), seed=13).to(device, dtype).requires_grad_(True)
    loss = MINDSSCLoss()(moving, fixed)
    loss.backward()
    assert loss.device.type == device and torch.isfinite(loss)
    assert loss.dtype == (torch.float32 if dtype == torch.bfloat16 else dtype)
    assert moving.grad.dtype == dtype and torch.isfinite(moving.grad.float()).all()
    # flat images: no NaNs
    flat = torch.zeros(1, 1, 8, 8, 8, device=device, dtype=dtype)
    assert torch.isfinite(MINDSSCLoss()(flat, flat))


def test_target_cache():
    shape = (16, 16, 16)
    fixed, moving = _smooth_noise(shape, seed=14), _smooth_noise(shape, seed=15)
    loss_fn = MINDSSCLoss()
    first = loss_fn(moving, fixed)
    cached = loss_fn._cache[2]
    assert torch.equal(loss_fn(moving, fixed), first) and loss_fn._cache[2] is cached
    fixed.mul_(-1)                                 # in-place change -> recomputed (same value: MIND is contrast-invariant)
    loss_fn(moving, fixed)
    assert loss_fn._cache[2] is not cached
    loss_fn.set_current_scale_and_iterations(1, 10)
    assert loss_fn._cache is None
    nocache = MINDSSCLoss(cache_target=False)
    assert torch.allclose(nocache(moving, fixed), first, atol=1e-6) and nocache._cache is None


def _image(arr, device):
    return Image(sitk.GetImageFromArray(arr.astype(np.float32)), device=device)


def _to_batch(arr, device):
    return BatchedImages([_image(arr, device)])


def _remapped_pair(size=48):  # >= 34 voxels: with fused ops, scale-2 downsampling of smaller images fails
    ''' a smooth image, a translated copy, and that copy with a non-monotone (T1 -> T2-like) contrast change '''
    big = _smooth_noise((size + 8,) * 3, seed=16)[0, 0].numpy()
    fixed = big[4:4 + size, 4:4 + size, 4:4 + size]
    moving = big[6:6 + size, 4:4 + size, 4:4 + size]
    return fixed, np.sin(3 * moving) ** 2, moving


@pytest.mark.parametrize("loss_type", ["mind", "masked_mind"])
def test_dispatcher_and_registration(loss_type):
    device = devices[-1]
    fixed_np, moving_np, moving_same_contrast_np = _remapped_pair()
    fixed_img, moving_img = _image(fixed_np, device), _image(moving_np, device)
    if loss_type == "masked_mind":
        fixed_img = generate_and_apply_mask_allones(fixed_img)
        moving_img = generate_and_apply_mask_allones(moving_img)
    fixed, moving = BatchedImages([fixed_img]), BatchedImages([moving_img])
    reg = GreedyRegistration(scales=[2, 1], iterations=[40, 20], fixed_images=fixed, moving_images=moving,
                             loss_type=loss_type, loss_params={"radius": 1, "dilation": 1}, optimizer="Adam",
                             optimizer_lr=0.2, smooth_warp_sigma=0.25, smooth_grad_sigma=0.5, progress_bar=False)
    assert isinstance(reg.loss_fn, MINDSSCLoss) and reg.loss_fn.spatial_dims == 3
    assert reg.loss_fn.masked == (loss_type == "masked_mind")
    reg.optimize()
    # judge the alignment in the original contrast, which the loss never saw
    fixed_eval, moving_eval = _to_batch(fixed_np, device), _to_batch(moving_same_contrast_np, device)
    moved = reg.evaluate(fixed_eval, moving_eval).detach()
    before = F.mse_loss(moving_eval(), fixed_eval()).item()
    after = F.mse_loss(moved, fixed_eval()).item()
    assert after < 0.5 * before, (before, after)


def test_affine_registration_with_mind():
    device = devices[-1]
    fixed_np, moving_np, moving_same_contrast_np = _remapped_pair()
    fixed, moving = _to_batch(fixed_np, device), _to_batch(moving_np, device)
    reg = AffineRegistration([2, 1], [100, 50], fixed, moving, loss_type="mind",
                             loss_params={"radius": 1, "dilation": 1}, optimizer="Adam", optimizer_lr=1e-2,
                             progress_bar=False)
    reg.optimize()
    moving_eval = _to_batch(moving_same_contrast_np, device)
    moved = reg.evaluate(fixed, moving_eval).detach()
    assert F.mse_loss(moved, fixed()).item() < 0.5 * F.mse_loss(moving_eval(), fixed()).item()


# --------------------------------------------------------------------------------- FeatureSimilarityLoss
class _Filters(nn.Module):
    ''' a fixed random filter bank, optionally strided and/or returning two levels '''
    def __init__(self, dims=3, features=8, stride=1, levels=1, bn=False, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        conv = nn.Conv2d if dims == 2 else nn.Conv3d
        self.conv = conv(1, features, 3, padding=1, stride=stride)
        self.conv2 = conv(features, features, 3, padding=1, stride=2) if levels == 2 else None
        self.bn = (nn.BatchNorm2d if dims == 2 else nn.BatchNorm3d)(features) if bn else None

    def forward(self, x):
        f = torch.tanh(self.conv(x))
        if self.bn is not None:
            f = self.bn(f)
        return f if self.conv2 is None else (f, torch.tanh(self.conv2(f)))


@pytest.mark.parametrize("distance", ["cosine", "l2"])
def test_feature_identity_is_minimum(distance):
    shape = (24, 24, 24)
    big = _smooth_noise([s + 8 for s in shape], seed=20)
    fixed = _crop(big, 0, shape)
    loss_fn = FeatureSimilarityLoss(_Filters(), distance=distance)
    assert loss_fn(fixed, fixed).item() == pytest.approx(0.0, abs=1e-6)
    assert all(loss_fn(_crop(big, t, shape), fixed).item() > 1e-4 for t in (-2, -1, 1, 2))


def test_feature_l2_normalized_is_twice_cosine():
    a, b = _smooth_noise((16, 16, 16), seed=21), _smooth_noise((16, 16, 16), seed=22)
    enc = _Filters()
    cos = FeatureSimilarityLoss(enc, distance="cosine")(a, b)
    l2 = FeatureSimilarityLoss(enc, distance="l2", normalize=True)(a, b)
    assert torch.allclose(l2, 2 * cos, atol=1e-5)


def test_feature_encoder_is_frozen():
    enc = _Filters(bn=True)
    loss_fn = FeatureSimilarityLoss(enc)
    assert not any(p.requires_grad for p in enc.parameters()) and not enc.training
    loss_fn.train()                         # must not put batchnorm back into training mode
    assert not enc.training
    moving = _smooth_noise((16, 16, 16), seed=23).requires_grad_(True)
    loss_fn(moving, _smooth_noise((16, 16, 16), seed=24)).backward()
    assert moving.grad.abs().sum() > 0 and all(p.grad is None for p in enc.parameters())


@pytest.mark.parametrize("distance", ["cosine", "l2"])
def test_feature_gradcheck(distance):
    enc = _Filters(dims=2, features=4).double()
    g = torch.Generator().manual_seed(25)
    pred = torch.rand(1, 1, 6, 7, generator=g, dtype=torch.float64, requires_grad=True)
    target = torch.rand(1, 1, 6, 7, generator=g, dtype=torch.float64)
    loss_fn = FeatureSimilarityLoss(enc, distance=distance, normalize=True)
    assert torch.autograd.gradcheck(lambda p: loss_fn(p, target), (pred,), eps=1e-6, atol=1e-5)


def test_feature_multilevel_and_weights():
    a, b = _smooth_noise((16, 16, 16), seed=26), _smooth_noise((16, 16, 16), seed=27)
    enc = _Filters(levels=2)
    fa, fb = enc(a), enc(b)
    per_level = [(1 - F.cosine_similarity(x, y, dim=1)).mean() for x, y in zip(fa, fb)]
    loss = FeatureSimilarityLoss(enc, level_weights=[3.0, 1.0])(a, b)
    assert torch.allclose(loss, (3 * per_level[0] + per_level[1]) / 4, atol=1e-6)
    with pytest.raises(ValueError):
        FeatureSimilarityLoss(enc, level_weights=[1.0])(a, b)


def test_feature_per_scale_encoders():
    encs = [_Filters(seed=s) for s in range(3)]
    loss_fn = FeatureSimilarityLoss(encs)
    loss_fn.set_scales([4, 2, 1])
    a, b = _smooth_noise((16, 16, 16), seed=28), _smooth_noise((16, 16, 16), seed=29)
    for idx, scale in enumerate([4, 2, 1]):
        loss_fn.set_current_scale_and_iterations(scale, 10)
        assert torch.allclose(loss_fn(a, b), FeatureSimilarityLoss(encs[idx])(a, b))
    with pytest.raises(AssertionError):
        FeatureSimilarityLoss(encs[:2]).set_scales([4, 2, 1])


def test_feature_masked_with_strided_encoder():
    shape = (16, 16, 16)
    a, b = _smooth_noise(shape, seed=30), _smooth_noise(shape, seed=31)
    enc = _Filters(stride=2)
    mask = torch.zeros(1, 1, *shape)
    mask[..., :8, :, :] = 1
    ones = torch.ones_like(mask)
    per_voxel = FeatureSimilarityLoss(enc, reduction="none")(a, b)      # [1, 1, 8, 8, 8]
    masked = FeatureSimilarityLoss(enc, masked=True)(torch.cat([a, mask], 1), torch.cat([b, ones], 1))
    assert torch.allclose(masked, per_voxel[..., :4, :, :].mean(), atol=1e-6)


def test_feature_target_cache():
    a, b = _smooth_noise((16, 16, 16), seed=32), _smooth_noise((16, 16, 16), seed=33)
    calls = []
    enc = _Filters()
    enc.register_forward_hook(lambda m, i, o: calls.append(1))
    loss_fn = FeatureSimilarityLoss(enc)
    for _ in range(3):
        loss_fn(a, b)
    assert len(calls) == 4               # moved image every call, fixed image once
    b.add_(0.1)
    loss_fn(a, b)
    assert len(calls) == 6               # in-place change invalidates the cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_feature_moves_encoder_to_input_device():
    enc = _Filters()
    a = _smooth_noise((16, 16, 16), seed=34).cuda()
    loss = FeatureSimilarityLoss(enc)(a, a.clone())
    assert loss.device.type == "cuda" and next(enc.parameters()).device.type == "cuda"


def test_feature_errors():
    with pytest.raises(TypeError):
        FeatureSimilarityLoss(lambda x: x)
    with pytest.raises(ValueError):
        FeatureSimilarityLoss(_Filters(), distance="dot")
    flat = nn.Flatten()
    with pytest.raises(ValueError):
        FeatureSimilarityLoss(flat)(torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8))


def test_feature_loss_drives_registration():
    device = devices[-1]
    big = _smooth_noise((56, 56, 56), seed=35)[0, 0].numpy()
    fixed = _to_batch(big[4:52, 4:52, 4:52], device)
    moving = _to_batch(big[6:54, 4:52, 4:52], device)
    loss_fn = FeatureSimilarityLoss(_Filters())
    reg = GreedyRegistration(scales=[2, 1], iterations=[40, 20], fixed_images=fixed, moving_images=moving,
                             loss_type="custom", custom_loss=loss_fn, optimizer="Adam", optimizer_lr=0.2,
                             smooth_warp_sigma=0.25, smooth_grad_sigma=0.5, progress_bar=False)
    reg.optimize()
    moved = reg.evaluate(fixed, moving).detach()
    assert F.mse_loss(moved, fixed()).item() < 0.5 * F.mse_loss(moving(), fixed()).item()
