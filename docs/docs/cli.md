# Command-Line Interface Tools

FireANTs provides command-line interface (CLI) tools that offer ANTs-compatible functionality with the performance benefits of the FireANTs framework. These tools are designed for users who prefer working from the command line or need to integrate FireANTs into existing workflows.

## Overview

The FireANTs CLI tools aim to provide a familiar interface for users coming from ANTs while leveraging GPU acceleration and the modern PyTorch backend of FireANTs. These tools maintain compatibility with ANTs workflows, making it easy to transition existing scripts and pipelines.

## Available Tools

### `fireantsRegistration`

The main registration tool, equivalent to ANTs' `antsRegistration`. This tool provides a complete pipeline for medical image registration with support for rigid, affine, and deformable (SyN/Greedy) transformations.

#### Key Features

- **Multi-stage registration**: Support for sequential Rigid → Affine → Deformable registration
- **Multiple similarity metrics**: MSE, Cross-Correlation (CC), Mutual Information (MI), and custom loss functions
- **Multi-resolution optimization**: Configurable shrink factors and iteration schedules
- **GPU acceleration**: Runs on CUDA devices for fast processing
- **ANTs compatibility**: Designed to work with ANTs file formats and workflows
- **Moment-based initialization**: Support for center-of-mass and principal axes alignment

#### Basic Usage

```bash
fireantsRegistration \
  --output output/transform \
  --transform Rigid[0.1] \
  --metric MI[fixed.nii.gz,moving.nii.gz,gaussian,32] \
  --convergence [100x50x25,1e-6,10] \
  --shrink-factors 4x2x1
```

#### Required Arguments

**`--output [prefix,warped_image]`**

Output prefix for transforms and path for the warped image. If only the prefix is provided, the warped image will be saved as `{prefix}_warped.nii.gz`.

```bash
--output output/myregistration
# or
--output output/myregistration,output/warped.nii.gz
```

**`--transform [type,params]`**

Transform type and parameters. Can be specified multiple times for multi-stage registration. Supported types:

- `Rigid[gradient_step,optimizer=Adam,scaling=False]` - Rigid (rotation + translation) transformation
- `Affine[gradient_step,optimizer=Adam]` - Affine (rigid + scaling + shearing) transformation  
- `SyN[gradient_step,optimizer=Adam,smooth_warp,smooth_grad]` - Symmetric diffeomorphic registration
- `Greedy[gradient_step,optimizer=Adam,smooth_warp,smooth_grad]` - Greedy diffeomorphic registration

```bash
--transform Rigid[0.1] \
--transform Affine[0.05] \
--transform SyN[0.2,Adam,0.25,0.5]
```

**`--metric [type,fixed,moving,params]`**

Similarity metric for registration. Must be specified once per transform. Supported types:

- `MSE[fixed,moving]` - Mean Squared Error
- `CC[fixed,moving,kernel_size]` - Cross-Correlation with optional kernel size (default: 5)
- `MI[fixed,moving,kernel_type,num_bins]` - Mutual Information (kernel_type: gaussian, num_bins: default 32)
- `Custom[fixed,moving,module.path.ClassName,param=value,...]` - Custom loss function

```bash
--metric CC[fixed.nii.gz,moving.nii.gz,5]
```

**`--convergence [iterations,tolerance,window]`**

Convergence parameters. Must be specified once per transform.

- `iterations`: Comma-separated (x-separated) list of iterations per resolution scale
- `tolerance`: Convergence tolerance threshold
- `window`: Number of iterations to check for convergence

```bash
--convergence [100x50x25x10,1e-6,10]
```

**`--shrink-factors [factors]`**

Shrink factors for multi-resolution optimization. Must be specified once per transform and match the number of iterations in `--convergence`.

```bash
--shrink-factors 8x4x2x1
```

#### Optional Arguments

**`--initial-moving-transform [fixed,moving,type]`**

Initial transform to align images before registration using moment matching:

- `1`: Match by center of mass with rotation (1st order moments)
- `2`: Match by principal axes with rotation (2nd order moments)  
- `3`: Match by principal axes with anti-rotation (2nd order moments)
- `4`: Match by principal axes with both orientations (2nd order moments)

```bash
--initial-moving-transform [fixed.nii.gz,moving.nii.gz,2]
```

**`--winsorize-image-intensities [lower,upper]`**

Clip image intensities to specified quantiles [0,1] or percentiles [0,100] to reduce the effect of outliers.

```bash
--winsorize-image-intensities [0.01,0.99]
```

**`--normalize-image-intensities`**

Normalize image intensities to [0,1] range.

**`--mask [fixed_mask,moving_mask]`**

Mask images for registration. Must provide both fixed and moving masks.

```bash
--mask fixed_mask.nii.gz,moving_mask.nii.gz
```

**`--device [cuda:0]`**

Device to run registration on. Default is `cuda:0`.

```bash
--device cuda:1
```

**`--smooth_warp_sigma [0.25]`**

Smoothing sigma for the warp field (for SyN/Greedy registration).

**`--smooth_grad_sigma [0.5]`**

Smoothing sigma for the gradient field (for SyN/Greedy registration).

**`--verbose`**

Enable verbose output for debugging.

#### Complete Example

Here's a complete example performing multi-stage registration with moment initialization:

```bash
fireantsRegistration \
  --output results/registration \
  --device cuda:0 \
  --winsorize-image-intensities [0.005,0.995] \
  --initial-moving-transform [fixed.nii.gz,moving.nii.gz,2] \
  --transform Rigid[0.03] \
  --metric MI[fixed.nii.gz,moving.nii.gz,gaussian,16] \
  --convergence [100x50x25x10,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --transform Affine[0.03] \
  --metric CC[fixed.nii.gz,moving.nii.gz,5] \
  --convergence [100x50x25x10,1e-4,10] \
  --shrink-factors 8x4x2x1 \
  --transform SyN[0.2] \
  --metric MSE[fixed.nii.gz,moving.nii.gz] \
  --convergence [100x70x50x20,1e-4,10] \
  --shrink-factors 8x4x2x1 \
  --verbose
```

This will produce:
- `results/registration_warped.nii.gz` - The registered moving image
- `results/registration0Warp.nii.gz` - The deformable transformation field
- Transform files for each stage

#### Custom Loss Functions

You can use custom loss functions by providing a Python module path:

```bash
--metric Custom[fixed.nii.gz,moving.nii.gz,mymodule.losses.MyLoss,param1=value1,param2=value2]
```

Your custom loss class should follow the FireANTs loss interface (see [Custom Loss Functions](customloss.md) for details).

#### Notes and Best Practices

1. **Transform ordering**: Transforms must be specified in order: `Rigid` → `Affine` → `SyN/Greedy`. You can skip stages but cannot go backward.

2. **Resolution matching**: The number of shrink factors must match the number of iteration values in `--convergence`.

3. **Memory considerations**: For large images, start with higher shrink factors (8x or 16x) to reduce memory usage.

4. **Learning rates**: The gradient step (first parameter) typically needs tuning:
   - Rigid: 1e-4 - 1e-3
   - Affine: 3e-3 - 3e-2  
   - SyN/Greedy: 0.2 - 0.5

5. **Metric selection**:
   - Use **MI** for multi-modal registration (e.g., T1 to T2 MRI)
   - Use **CC** for mono-modal registration with good local contrast
   - Use **MSE** for images with similar intensity distributions

## Tools in Development

The following CLI tools are currently under development:

### `fireantsApplyTransforms`

Apply saved transformations to new images (equivalent to ANTs `antsApplyTransforms`).

### `fireantsInvertDispField`

Invert displacement field transformations post-hoc - a new feature not available in ANTs that allows inverting already computed displacement fields.

### `fireantsMultiVariateTemplateConstruction`

Multimodal template construction using FireANTs' registration framework.

## ANTs Compatibility

FireANTs CLI tools are designed to be compatible with ANTs workflows:

- **File formats**: Supports NIfTI (.nii, .nii.gz) and other common medical imaging formats
- **Transform formats**: Outputs transforms compatible with ANTs tools
- **Parameter conventions**: Command-line arguments follow ANTs conventions where possible

However, there are some differences:

- **GPU-first**: FireANTs is optimized for GPU execution
- **PyTorch backend**: Leverages modern deep learning infrastructure
- **Simplified options**: Some less-used ANTs options are not yet implemented

## Getting Help

For detailed help on any CLI tool, use the `--help` flag:

```bash
fireantsRegistration --help
```

For more information on the underlying registration methods, see:

- [Quickstart Guide](quickstart.md)
- [Rigid and Affine Registration](advanced/rigidaffine.md)
- [Deformable Registration](advanced/deformable.md)
- [Custom Loss Functions](customloss.md)

