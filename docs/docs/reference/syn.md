# Symmetric Deformable Image Matching

FireANTs also supports symmetric deformable image matching. Symmetric deformable image matching is a very popular method for matching two images by considering both the fixed and moving images '_symmetrically_' in the optimization process.
This formulation ensures that spatial gradients of both the fixed and moving images are considered in the optimization process.

Symmetric registration finds a deformation field that minimizes the similarity metric between the fixed and moving images, by warping the moving and fixed image to a 'midpoint' space.

$$ \varphi_1^*, \varphi_2^* = \arg \min_{\varphi_1, \varphi_2} \mathcal{L}(I_m\circ\varphi_1, I_f\circ\varphi_2) $$

The deformation field is optimized using gradient descent to minimize a similarity metric while maintaining smoothness through optional regularization terms.

FireANTs supports both free-form and diffeomorphic transforms for symmetric registration.

!!! info "Composing Transforms for Symmetric Registration"

    The final transform is composed of the two deformation fields, $\varphi_1$ and $\varphi_2$.

    $$ \varphi = \varphi_1 \circ \varphi_2^{-1} $$

---

::: fireants.registration.syn.SyNRegistration

