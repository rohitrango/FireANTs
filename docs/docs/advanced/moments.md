# Moment matching for image matching

Different images reside in different physical coordinates. However, the registration algorithms like [Rigid/affine matching](./rigidaffine.md) and [Deformable matching](./deformable.md) assume that the images reside in the same physical coordinates. 

This makes it easy to perform image matching in the pixel space directly.

Moment matching roughly works in the following way:

1. Compute the specified moments of the images using _physical coordinates_.

    - The first moment is simply the weighted center of mass of the image.

    $$ \mathbf{m}_1 = \frac{\sum_{i=1}^n \mathbf{x}_i I(\mathbf{x}_i)}{\sum_{i=1}^n I(\mathbf{x}_i)} $$

    - The second moment is the weighted covariance matrix of the image.

    $$ \mathbf{M}_2 = \frac{\sum_{i=1}^n (\mathbf{x}_i - \mathbf{m}_1)(\mathbf{x}_i - \mathbf{m}_1)^T I(\mathbf{x}_i)}{\sum_{i=1}^n I(\mathbf{x}_i)} $$

2. If first-order matching is required, the transformation is simply the difference between the moments of the images.

    $$ \mathbf{T} = \mathbf{m}_1^{\text{moving}} - \mathbf{m}_1^{\text{fixed}} $$

3. If second-order matching is required, the optimal rotation is the solution to the following optimization problem:

    $$ \min_{\mathbf{R} \in SO(n)} \left\| \mathbf{M}_2^{\text{moving}} - \mathbf{R} \mathbf{M}_2^{\text{fixed}} \mathbf{R}^T \right\| $$

    This can be solved easily using SVD giving us
    
    $$ \mathbf{R} = \mathbf{U} \mathbf{V}^T $$
    
    However, the rotation matrix is underdetermined. To fix this, we try all possible rotations and select the one with the smallest error.

    $$ \mathbf{R} = \mathbf{U} D \mathbf{V}^T $$

    where $D$ is the diagonal matrix containing either $1$ or $-1$ on the diagonal to keep the determinant of the matrix to be $1$.
    if flips are requested in the transformation, $D$ is selected to have determinant $-1$. The optimal choice of $D$ is determined by minimizing a similarity loss between the fixed image and the moving image transformed by the rotation matrix and $D$.

    Given the optimal rotation matrix, the translation is given as

    $$ \mathbf{T} = \mathbf{m}_1^{\text{moving}} - \mathbf{R} \mathbf{m}_1^{\text{fixed}} $$

## An example

This example demonstrates how to perform moment matching for images. We will use an in-vivo to ex-vivo hemisphere registration as an example.
First, we visualize the images in ITK-SNAP.

![in-vivo](/assets/moment/before-moment-matching.png)

The in-vivo image (right) falls completely outside the range of the ex-vivo image's physical coordinates range, therefore the image is not visible in the ITK-SNAP window.

We run moment matching to align the images.

```python
from fireants.registration import MomentsRegistration
moments = MomentsRegistration(fixed_images=fixed_images_batch, \
                                moving_images=moving_images_batch, \
                                **dict(args))
moments.optimize(save_transformed=False)
```

![after-moment-matching](/assets/moment/after-moment-matching.png)

The images are now in a similar physical space, and amenable to rigid/affine/deformable image registration.

## Composing with other transforms

Using the results of moment matching as initialization for rigid and affine transformations is straightforward.

Rigid:

```python
# retrieve rotation and translation matrices from moment matching 
init_moment_rigid = moments.get_rigid_moment_init()
init_moment_transl = moments.get_rigid_transl_init()

rigid = RigidRegistration(fixed_images=fixed_images_batch, \
                        moving_images=moving_images_batch, \
                        init_translation=init_moment_transl, \  # initialized with translation
                        init_moment=init_moment_rigid, \        # initialized with rotation 
                        **dict(args))
```

Affine:

```python
# retrieve rigid matrix from moment matching 
init_rigid = moments.get_affine_init()      

affine = AffineRegistration(fixed_images=fixed_images_batch, \
                            moving_images=moving_images_batch, \
                            init_rigid=init_rigid, \        # initialized with rigid matrix
                            **dict(args))
```

