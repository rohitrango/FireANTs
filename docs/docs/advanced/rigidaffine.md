# Rigid and Affine Image Matching

## Introduction

Rigid and affine image matching is a common task in medical image analysis. Rigid registration is a special case of affine registration where the rotation and translation are constrained to be the same for all voxels.

Mathematically, both problems as formulated as optimization problems:

$$ \min_{T \in \mathcal{G}} \sum_{i=1}^N \mathcal{L}(I_f(x), I_m(\phi(x))) $$

where $\mathcal{L}$ is the dissimilarity/loss function, $I_f$ is the fixed image, $I_m(\phi(x))$ is the moving image warped by the transformation $\phi(x) = Ax + t$, where

* $A$ is the rotation matrix for rigid registration, and an affine transformation for affine registration.

* $t$ is the translation vector for both rigid and affine registration.

For rigid registration, the rotation matrix $A \in SO(n)$ is represented by its Lie algebra of skew-symmetric matrices $A = \exp(\omega)$, where $\omega$ is a vector in $\mathbb{R}^{n(n-1)/2}$. 
For affine registration, the transformation $\phi$ is represented by a matrix $A \in \mathbb{R}^{n \times n}$ and a translation vector $t \in \mathbb{R}^n$.
This ensures that all optimization parameters are in the Euclidean space, and standard optimization methods can be used.

## Quickstart

FireANTs supports an easy interface for rigid and affine image matching.

Running a rigid registration is as simple as:

```python
# load the images
image1 = Image.load_file("atlas_2mm_1000_3.nii.gz")
image2 = Image.load_file("atlas_2mm_1001_3.nii.gz")
# batchify them (we only have a single image per batch, but we can pass multiple images)
fixed_batch = BatchedImages([image1])
moving_batch = BatchedImages([image2])

# rigid registration
scales = [4, 2, 1]  # scales at which to perform registration
iterations = [200, 100, 50]
optim = 'Adam'
lr = 3e-4

# create rigid registration object
rigid_reg = RigidRegistration(scales, iterations, fixed_batch, moving_batch, 
                            optimizer=optim, optimizer_lr=lr,
                            cc_kernel_size=5)
# call method
rigid_reg.optimize()
```

**Affine registration** follows a similar interface:

```python

# get rigid registration matrix as optional initialization for affine registration
rigid_matrix = rigid_reg.get_rigid_matrix()

# create affine registration object
affine_reg = AffineRegistration(scales, iterations, fixed_batch, moving_batch, 
                            optimizer=optim, optimizer_lr=lr,
                            init_rigid=rigid_matrix,  # optionally initialized with rigid matrix
                            cc_kernel_size=5)

affine_reg.optimize()
```

Checkout the reference for details on more parameters.


## Initialization

You can initialize translation (and optionally the linear part) from the **center of frame** (COF) so that the physical centers of the fixed and moving images are aligned before optimization. This uses \(c_m - c_f\) in physical space (same convention as [moment matching](moments.md) with `transl_mode="cof"`).

**Rigid registration** — set initial translation to \(c_m - c_f\) and keep rotation at identity:

```python
rigid_reg = RigidRegistration(scales, iterations, fixed_batch, moving_batch,
                              init_translation="cof",  # center-of-frame: c_m - c_f
                              optimizer=optim, optimizer_lr=lr)
```

**Affine registration** — set initial translation to \(c_m - c_f\) and the linear part to identity:

```python
affine_reg = AffineRegistration(scales, iterations, fixed_batch, moving_batch,
                                init_rigid="cof",  # identity + translation c_m - c_f
                                optimizer=optim, optimizer_lr=lr)
```

For more advanced initialization (e.g. from moment matching or subspace methods), pass a tensor to `init_translation` / `init_rigid` as shown in [Composing with other transforms](moments.md#composing-with-other-transforms) and in the [Subspace2D Affine how-to](../howto/subspace2d-affine.md).


## Additional Functionality

* **Dense transformation grid**: Both rigid and affine registration extend the `get_warped_coordinates` method to return a dense coordinate grid of the warp function $\phi(x) = Ax + t$.

* **Saving as ANTs transforms**: Coming soon.
