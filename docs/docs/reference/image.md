# Image

FireANTs provides a base `Image` class that handles the loading, manipulation, and transformation of medical images.
This image contains the actual data and the metadata to be aware of the physical coordinates of the image.

::: fireants.io.image.Image

---

# BatchedImages

This class is a wrapper around a list of `Image` objects. All registration algorithms in FireANTs are designed to work with `BatchedImages` objects.

::: fireants.io.image.BatchedImages