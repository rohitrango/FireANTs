# Fused CUDA kernels for ultra-efficient registration

To make registration faster and _really_ memory efficient for deep learning and optimization-based methods (including FireANTs), we implement a set of fused CUDA kernels that implement

- Affine and Deformable Interpolation
- Warp Composition
- Cross Correlation
- Mutual Information

The performance benefits are staggering - the interpolation, warp composition, and mutual information now become $O(1)$ operations that scale much better with image size. Training deep learning models with our fused operations can reduce training time by upto 80% and memory usage by upto 20%. More details in the [paper](https://arxiv.org/abs/2509.25044).

## Installation
Make sure that `fireants` and `fireants_fused_ops` are installed. See the [README.md](../README.md) for more details.

## Basic Usage

> [!NOTE] 
> All the examples below are for 3D images but work for 2D images as well.

> [!TIP] 
> For all the examples below, make sure that all inputs are contiguous.

### Interpolation
First, import the package:
```python
from fireants.interpolator import fireants_interpolator as fi
```

To sample from an image
```python
image = torch.randn(1, 1, 128, 128, 128).cuda()
displacement = torch.randn(1, 192, 160, 224, 3).cuda()  # displacement field u(x)
interpolated_image = fi(image, grid=displacement)   # computes I(x + u(x))
```
Note that in PyTorch convention, vector fields are stored in channel-last format, but images are stored in channel-first format (except batch dimension).

If an affine transformation is provided, you can use the `affine` flag to specify:
```python
interpolated_image = fi(image, affine=affine)   # computes I(Ax)
```

If the target shape is different from the input shape, you can specify the `out_shape` parameter:
```python
interpolated_image = fi(image, affine=affine, out_shape=(192, 160, 224))   # spatial dims are provided as a tuple
interpolated_image = fi(image, affine=affine, out_shape=(1, 1, 192, 160, 224))   # also works
```

To compute a composite warp field (affine + displacement), you can specify both the `affine` and `grid` parameters:
```python
interpolated_image = fi(image, affine=affine, grid=displacement)   # computes I(Ax + u(x))
```

If the warp field is _not_ a displacement field (i.e. it is a full warp field), you can use the `is_displacement` flag to specify:
```python
interpolated_image = fi(image, grid=displacement, is_displacement=False)   # computes I(x + v(x))
```

If you specify `is_displacement=False`, then the `affine` parameter is ignored.

### Warp Composition
A similar implementation is used for warp composition. 

Warp composition can be used to perform diffeomorphic updates or computing scaling-or-squaring (examples below). In FireANTs, we implement a generic warp composition of the form $u \circ (Ax + v)$ where $u, v$ are displacement fields, $A$ is an affine transformation. 

```python
u = torch.randn(1, 192, 160, 224, 3).cuda()
v = torch.randn(1, 192, 160, 224, 3).cuda()
affine = torch.randn(1, 3, 4).cuda()
composed_warp = fi.warp_composer(u, affine=affine, v=v)   # computes u \circ (Ax + v)
```

If `affine` is not provided, it is assumed to be the identity grid. 

Implementing scaling-and-squaring is easy:
```python
# 'v' is the velocity field v(x)
def scaling_and_squaring(v: torch.Tensor, n: int = 6):
  u = v / 2**n
  for i in range(n):
    u = fi.warp_composer(u, affine=None, v=u)
  return u
```

> [!WARNING] 
> $v$ cannot be a warp field, it must be a displacement field. If you have a warp field, subtract the identity grid from it to get a displacement field.

### Mutual Information

```python
from fireants.losses.fusedmi import FusedGlobalMutualInformationLoss as FusedMI
pred = torch.randn(1, 1, 192, 160, 224).cuda().requires_grad_(True)
target = torch.randn(1, 1, 192, 160, 224).cuda()
loss_module = FusedMI(num_bins=32, )
loss = loss_module(pred, target)
loss.backward()
```
This loss also works for multi-channel images, where it calculates the mutual information for each channel independently and averages across the channels.

### Cross Correlation

```python
from fireants.losses.fusedcc import FusedLocalNormalizedCrossCorrelationLoss as FusedLNCC
pred = torch.randn(1, 1, 192, 160, 224).cuda().requires_grad_(True)
target = torch.randn(1, 1, 192, 160, 224).cuda()
loss_module = FusedLNCC(kernel_size=7)
loss = loss_module(pred, target)
loss.backward()
```

For networks like TransMorph on Pytorch >= 1.9.1, the LNCC loss with box kernel (default) [does not work well](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/issues/55). In that case, use the gaussian kernel instead.

```python
loss_module = FusedLNCC(kernel_size=7, kernel_type='gaussian')
```

## Citation

If you use the fused CUDA kernels in your research, please cite the paper
```
@article{jena2025scalable,
  title={A Scalable Distributed Framework for Multimodal GigaVoxel Image Registration},
  author={Jena, Rohit and Zope, Vedant and Chaudhari, Pratik and Gee, James C},
  journal={arXiv preprint arXiv:2509.25044},
  year={2025}
}
```
