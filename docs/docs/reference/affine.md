# Affine Image Matching

FireANTs supports multi-scale affine image matching between two images. 
Affine transformations are more flexible than rigid transformations, as they allow for scaling, rotation, and shearing.

**Initialization:** You can pass `init_rigid="cof"` to set the initial translation to \(c_m - c_f\) (center of moving minus center of fixed) and the linear part to identity. Pass a tensor (e.g. from rigid or moment matching) to use a custom initial affine.

::: fireants.registration.affine.AffineRegistration

