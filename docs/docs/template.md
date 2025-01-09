# Template creation using FireANTs

Yet another powerful application of FireANTs is super fast template construction.

FireANTs provides a simple starter script in `fireants/scripts/template/build_template.py` to create a template from a set of images.
The script requires a YAML config file (a default one is provided in `fireants/scripts/template/configs/oasis_deformable.yaml`) that supports the following:

## Multi-Stage Registration Pipeline
* Moments registration: Initial alignment using image moments
* Rigid registration: Rotation, scaling, and translation only
* Affine registration: Includes general affine registration
* Deformable registration: Choice between:
    * Greedy deformable registration
    * SyN (Symmetric Normalization) registration

## Template Initialization Options
* Use an existing image as initial template (`init_template_path`)
* Create average from input images if no initial template provided

## Shape Averaging
* Optional feature to maintain shape consistency
* Averages displacement fields across all registrations
* Applies inverse warp to maintain anatomical correspondence

## Template Post Processing
* Laplacian filtering for template smoothing
    * Configurable number of iterations (`num_laplacian`)
    * Adjustable learning rate and scaling parameters
* Multi-resolution optimization at each registration stage
* Optional blurring during downsampling

## Distributed Processing
* Parallel processing across multiple GPUs
* Batch processing of images
* Synchronized template updates across processes

## Customizable Parameters
* Different similarity metrics (CC, MI, MSE)
* Adjustable optimization parameters per registration stage
* Configurable kernel sizes for correlation metrics
* Smoothing parameters for warp fields and gradients

## Additional Features
* Support for saving intermediate templates
* Ability to save transformed images
* Support for processing additional related files (e.g., segmentation masks)
* Progress tracking and verbose logging options

## Memory Management
* Batch processing to handle large datasets
* Optional memory optimization for image handling
* Cleanup of intermediate results

This template construction framework is designed to be flexible and modular, allowing users to customize the registration pipeline while maintaining computational efficiency through distributed processing.

An example of how to run template construction is provided below:

```bash
#!/bin/bash
torchrun --nproc_per_node=8  build_template.py --config_name oasis_deformable
```

Change `oasis_deformable` with a config file that meets your needs.

