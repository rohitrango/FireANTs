# Image Representation

The core data structure used in FireANTs is an image.

However, images in medical imaging differ from images in computer vision (hereby referred to as 'natural images') in a fundamental way:

* Natural images are typically *projections* of a 3D scene. Each pixel lies on an *image plane*. The intensity at each pixel can possibly represent a physical location that is different from all other pixel locations. This leads to many challenges in computer vision applications - depth ambiguity, occlusions, etc. 
<!-- Correspondence matching for natural images is therefore modelled as a saliency or feature matching problem. -->

* In contrast, each pixel/voxel in a medical image typically corresponds directly to a location in the physical space rather than being a projection on an image plane. Therefore, most medical images reside in some physical space. Dense correspondence matching therefore corresponds to matching actual physical locations. 
 
However, most modern image processing libraries are built for natural images. Therefore, we must imbue our data structure with additional metadata to enable correspondence matching of *physical locations*. To do this, we need the following additional information about the physical space:

* **Origin**: For an n-D image, the origin provides the n-D location of the first pixel in the image.

* **Direction**: A rotation matrix that provides the basis vectors for rotating points from the pixel space to the physical space.

* **Spacing**: Physical spacing between adjacent pixels.

These variables are stored in the metadata for most medical image formats. In `SimpleITK`, we can access these variables using the following code:

```python
import SimpleITK as sitk

image = sitk.ReadImage('path/to/image.nii')
origin = image.GetOrigin()
direction = image.GetDirection()
spacing = image.GetSpacing()
```

These variables are required to convert coordinates from the physical space (the space where images reside, and must be registered to) to pytorch's pixel space (where computations are performed). 

FireANTs wraps the SimpleITK object in a convenient `Image` class. 

```python
from fireants.io import Image
image = Image.load_file("/path/to/image.nii")
``` 

The `Image` class eagerly loads the image data into a pytorch tensor, and optionally accepts arguments to specify whether the image is an integer segmentation image, and if so, the background label and max cutoff segmentation label. This is bundled with the option to specify a 'segmentation preprocessor' function that is applied to the segmentation image. 

Finally, the `Image` class allows the user to override the default spacing, direction, and origin values stored in the image metadata and provide them as inputs. 

After initialization, the `Image` class stores the extra variables:

* `torch2phy` : A $(n+1) \times (n+1)$ matrix that maps coordinates from pytorch's pixel space to the physical space.

* `phy2torch` : A $(n+1) \times (n+1)$ matrix that maps coordinates from the physical space to pytorch's pixel space.

