# DeformableMixin

This mixin provides common functionality for deformable registration classes.


!!! question "Why Deformable Mixin?"

    Although the [AbstractRegistration](./abstract.md) class provides a common interface for all registration classes, there are significant differences in the functionality required from deformable registration classes.
    Greedy and Symmetric registration classes have different optimizations, but share the same final transformation: a deformation field that warps the moving image to the fixed image space.

The DeformableMixin class provides common functionality for both Greedy and Symmetric registration classes, particularly for saving deformation fields in a format compatible with ANTs (Advanced Normalization Tools) and other widely used registration tools.

---

## DeformableMixin class

::: fireants.registration.deformablemixin.DeformableMixin

