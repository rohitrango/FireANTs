# Installation

Installing FireANTs is super easy! We recommend using a virtual environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [venv](https://docs.python.org/3/library/venv.html). 

We recommend getting a fresh conda environment for installing FireANTs.

```bash
conda create -n fireants python=3.10
```

Next, you can install FireANTs locally from source:

```bash
git clone https://github.com/rohitrango/fireants
cd fireants
pip install -e .
```

or install directly from PyPI (this may not be the latest iteration):

```bash
pip install fireants
```

It's as easy as that! No more superbuilds. No more cmake. No more waiting for hours to compile. No more esoteric errors. Just install and run! 🚀

## Fused CUDA Operations

FireANTs also comes with a set of fused CUDA operations to improve memory and runtime performance of core operations in the registration pipeline. To install the fused CUDA operations, install the `fireants_fused_ops` package:

```bash
cd fused_ops
python setup.py build_ext && python setup.py install
cd ..
```
This will build the required CUDA extensions and install the `fireants_fused_ops` package.
You can find the incredible memory savings and runtime improvements in the [paper](https://arxiv.org/abs/2509.25044). Note that this package is only available for NVIDIA GPUs - sorry CPU and MPS users! If you are interested in implementing these fused ops for CPUs or other hardware, please [open an issue](https://github.com/rohitrango/fireants/issues/new) or [submit a pull request](https://github.com/rohitrango/fireants/pulls).

## Docker 

For Windows users, PyTorch installation might be tricky, especially NCCL support. We recommend using the Docker image to avoid these issues. See [docker/README.md](https://github.com/rohitrango/FireANTs/blob/main/docker/README.md) for more details.

!!! warning "Troubleshooting install"

    Still facing trouble? [Open an issue](https://github.com/rohitrango/fireants/issues/new) and we will get back to you as soon as possible.