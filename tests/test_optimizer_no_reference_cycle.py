import gc

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from fireants.io.image import BatchedImages, Image
from fireants.registration.syn import SyNRegistration

# Optimizers store their smoothing wrapper on themselves; if it closes over the optimizer they form a
# reference cycle, and every warp leaks until the device OOMs.


def _one_registration(size=64, iterations=3):
    ''' a registration whose objects all go out of scope on return '''
    rng = np.random.default_rng(0)
    fixed_np = rng.random((size, size, size), dtype=np.float32)
    moving_np = np.roll(fixed_np, 3, axis=0)
    fixed = Image(sitk.GetImageFromArray(fixed_np), device="cuda")
    moving = Image(sitk.GetImageFromArray(moving_np), device="cuda")
    reg = SyNRegistration(
        scales=[1],
        iterations=[iterations],
        fixed_images=BatchedImages([fixed]),
        moving_images=BatchedImages([moving]),
        loss_type="cc",
        cc_kernel_size=3,
        deformation_type="compositive",
        optimizer="Adam",
        optimizer_lr=0.25,
    )
    reg.optimize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device to measure")
def test_repeated_registrations_do_not_leak():
    ''' gc is disabled so refcounting alone must reclaim each registration '''
    torch.cuda.init()
    _one_registration()  # warm up: kernels and lazy imports allocate once and stay
    gc.collect()
    torch.cuda.empty_cache()

    gc.disable()
    try:
        baseline = torch.cuda.memory_allocated()
        for _ in range(4):
            _one_registration()
        growth = torch.cuda.memory_allocated() - baseline
    finally:
        gc.enable()

    # a leaking build grows ~80 MB per iteration here, gigabytes at production sizes
    assert growth < 8 * 1024**2, (
        f"{growth / 1024**2:.1f} MB of CUDA memory still live after 4 registrations that were all "
        f"dropped. Reference counting should have reclaimed them; a reference cycle prevents it."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device to measure")
def test_optimizer_is_not_in_a_reference_cycle():
    ''' DEBUG_SAVEALL parks cyclic garbage in gc.garbage instead of freeing it '''
    _one_registration()
    gc.collect()

    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        _one_registration()
        gc.collect()
        cyclic = [type(o).__name__ for o in gc.garbage]
    finally:
        gc.set_debug(0)
        del gc.garbage[:]
        gc.collect()

    leaked = [name for name in cyclic if "Warp" in name and name.endswith(("Adam", "SGD", "Marquardt"))]
    assert not leaked, f"optimizers collected as cyclic garbage: {sorted(set(leaked))}"
