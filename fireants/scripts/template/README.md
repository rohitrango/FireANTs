# 🧠 FireANTs Template Builder

A powerful tool for building anatomical templates from medical images using the FireANTs registration framework.

## 🚀 Overview

The template builder creates high-quality anatomical templates by iteratively registering multiple images and averaging them together. This process produces templates that represent the average anatomy of your dataset, which can be used for subsequent image registration and analysis tasks.

## 📋 Features

🔥 **Multi-modal template creation** - Support for multiple image modalities

⚡ **Distributed processing** - Parallel processing across multiple GPUs/nodes

🎯 **Multiple registration algorithms** - Rigid, affine, and deformable registration

🔧 **Flexible configuration** - YAML-based configuration for easy customization

📊 **Shape averaging** - Advanced shape averaging for better template quality

💾 **Checkpoint saving** - Save templates at specified intervals

🖼️ **Image preprocessing** - Automatic normalization and filtering options

## 🛠️ Installation

Ensure you have FireANTs installed and properly configured for your system.

## ⚙️ Configuration

The template builder uses YAML configuration files to specify all parameters. See `configs/oasis_deformable.yaml` for a complete example.

### 📁 Input Parameters

🎯 **Image List**: Specify your input images
```yaml
image_list_file: /path/to/your/image_paths.txt
num_subjects: 8  # null for all subjects
```

🏁 **Initial Template**: Start with an existing template or create from scratch
```yaml
init_template_path: /path/to/initial/template.nii.gz  # null to create from average
```

### 🔄 Registration Pipeline

The registration pipeline supports multiple stages:

#### 📐 Moments Registration
```yaml
do_moments: True
moments:
  scale: 4
  moments: 1
```

#### 🔧 Rigid Registration
```yaml
do_rigid: True
rigid:
  iterations: [200, 100, 25]
  scales: [4, 2, 1]
  loss_type: cc
```

#### 📏 Affine Registration
```yaml
do_affine: True
affine:
  iterations: [200, 100, 25]
  scales: [4, 2, 1]
  loss_type: cc
```

#### 🌊 Deformable Registration
```yaml
do_deform: True
deform_algo: greedy  # or 'syn'
deform:
  iterations: [200, 100, 50]
  scales: [4, 2, 1]
  optimizer_lr: 0.25
  smooth_warp_sigma: 0.5
  smooth_grad_sigma: 1.0
  cc_kernel_size: 5
  loss_type: fusedcc
```

### 🎨 Template Creation Options

🔢 **Template Iterations**: Number of template building iterations
```yaml
template_iterations: 6
```

📊 **Shape Averaging**: Enable advanced shape averaging
```yaml
shape_avg: true
```

🧮 **Laplacian Smoothing**: Apply smoothing filters
```yaml
num_laplacian: 2
laplace_params:
  learning_rate: 0.5
  itk_scale: true
```

🔧 **Image Processing**: Normalization and data type options
```yaml
normalize_images: True
save_as_uint8: False
```

### 💾 Output Configuration

📂 **Save Directory**: Where to store results
```yaml
save_dir: ./saved_templates
save_every: 1  # Save every N iterations
save_init_template: true
save_moved_images: false  # Save registered images from final iteration
```

## 🚀 Usage

### 📝 Basic Usage

```bash
torchrun --nproc-per-node=1 build_template.py
```

The script will use the default configuration (`oasis_deformable.yaml`) and create templates in the specified output directory.

### 🎛️ Custom Configuration

```bash
torchrun --nproc-per-node=1 build_template.py --config-name your_config.yaml
```

### 🌐 Distributed Training

For multi-GPU processing:

```bash
torchrun --nproc_per_node=8 build_template.py
```

## 📊 Monitoring Progress

The script provides detailed logging information:

🔍 **Debug Mode**: Enable verbose logging
```yaml
debug: True
verbose: True
```

📈 **Progress Tracking**: Monitor registration progress and template quality metrics

## 📁 Output Files

The template builder generates:

🎯 **Templates**: `template_0.nii.gz`, `template_1.nii.gz`, etc.

📋 **Logs**: Detailed logging of the registration process

🖼️ **Moved Images** (optional): Registered images from the final iteration

## 🔧 Advanced Options

### 🎨 Multi-modal Templates

For multi-modal template creation, provide comma-separated paths:
```yaml
init_template_path: /path/to/t1.nii.gz,/path/to/t2.nii.gz
```

### 📊 Additional Image Processing (WIP)

Save additional images or segmentations:
```yaml
save_additional:
  image_file: ["/path/to/labels.txt"]
  image_prefix: ["labels_"]
  image_suffix: [".nii.gz"]
  is_segmentation: [true]
```

### ⚡ Performance Tuning

🔢 **Batch Size**: Adjust based on available memory
```yaml
batch_size: 8
```

🗂️ **Temporary Directory**: Specify custom temp directory
```yaml
tmpdir: /path/to/temp
```

## 📚 Example Workflow

1. 📝 **Prepare your image list**: Create a text file with paths to your images
2. ⚙️ **Configure parameters**: Modify the YAML configuration file
3. 🚀 **Run template building**: Execute the script
4. 📊 **Monitor progress**: Watch the logs for convergence
5. ✅ **Validate results**: Check the final template quality

## 🆘 Troubleshooting

🔧 **Memory Issues**: Reduce batch_size or use more distributed nodes

⚡ **Slow Convergence**: Adjust learning rates or increase iterations

📁 **File Path Errors**: Ensure all paths in your image list are valid

🖥️ **GPU Issues**: Check CUDA availability and device settings

## 📖 References

For more information about the underlying algorithms and methods, see the FireANTs documentation and relevant publications.

---

🎉 **Happy Template Building!** 🎉

