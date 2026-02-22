# Rigid Image Matching

FireANTs supports multi-scale rigid image matching between two images.

**Initialization:** You can pass `init_translation="cof"` to set the initial translation to \(c_m - c_f\) (center of moving minus center of fixed in physical space), with identity rotation. Pass a tensor to use a custom translation, or `init_moment` for a custom rotation matrix.

::: fireants.registration.rigid.RigidRegistration

