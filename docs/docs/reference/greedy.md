# Greedy Deformable Image Matching

Deformable image matching is the core feature of FireANTs. Greedy deformable image matching is a simple and fast method for registering two images by performing gradient descent optimization on the deformation field to directly minimize the similarity metric.

This class implements greedy deformable registration between fixed and moving images,
optionally initialized with an affine transform. 

Greedy registration finds a deformation field that minimizes the similarity metric between the fixed and moving images, by warping the moving image to the fixed image space.

$$ \varphi^* = \arg \min_{\varphi} \mathcal{L}(I_m\circ\varphi, I_f) $$

The deformation field is optimized using gradient descent to minimize a similarity metric while maintaining smoothness through optional regularization terms.

!!! caution "Why is it called Greedy?"

    Deformation image matching is a highly ill-conditioned problem. The 'greedy' way to update $\varphi(x)$ is by computing its gradient w.r.t. the similarity metric at that point, irrespective of all othe points, and using it to update $\varphi(x)$.

FireANTs supports both free-form and diffeomorphic transforms.

!!! info "Diffeomorphic Transforms"

    Diffeomorphic transforms are a special class of deformations that are both smooth and invertible. They are useful for registering images that have anatomically plausible non-linear deformations.

::: fireants.registration.greedy.GreedyRegistration

