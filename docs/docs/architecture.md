# A typical real world Registration Pipeline

Most registration pipelines will require the user to ensure that the images are:

- of the same voxel size (and preferably a multiple of a power of 2)
- roughly aligned already in the voxel space, but _not_ physical space

In contrast, most real-world images are:

- typically never of the same voxel size. 
    This problem is exacerbated by applications like in-vivo to ex-vivo registration or CBCT to FBCT registration where voxel sizes are significantly different due to different physical resolutions.
- often can have different metadata, including different spacing, origin and direction matrices. We highly recommend the reader to refer to the [SimpleITK documentation](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html) for more details on the importance of image metadata.

Performing repeated resampling to bring the images to the same voxel space and physical space is a huge burden on the user. Multiple resampling operations can also introduce aliasing artifacts.


To mitigate this, FireANTs allows the user to register images by composing transforms including 
- moments matching (to bring the images to the same physical space and orientation)
- rigid/affine matching (to perform "correctional" rigid/affine transforms after ensuring the images are roughly aligned in the physical space either by verifying the metadata or by using moment matching)
- deformable matching (to perform "large" deformable transforms after ensuring the images are roughly aligned in the physical space).
This is the step that is of most quantitative importance in the registration pipeline, but the previous steps are important to ensure that the deformable matching is performed correctly.

Furthermore, FireANTs allows the user to use `feature images' instead of the intensity images in the registration pipeline.
We provide a couple of powerful feature extractors, and allow the user to define their own custom feature extractors.

The entire registration pipeline looks like the following:

<div style="text-align: center;">
    <img src="../assets/images/fireants-pipeline.png" width="80%" alt="Registration Pipeline"/>
</div>
<br>
Note that the "loss fn", "warp reg" and "disp reg" can be user-defined modules that are used to steer the registration process.

!!! warning "Active Development"

    We are actively working on adding more preconditioning techniques and feature Backends to the library.
    If you are interested in contributing more to the library, please [open an issue](https://github.com/rohitrango/fireants/issues/new) or [submit a pull request](https://github.com/rohitrango/fireants/pulls).