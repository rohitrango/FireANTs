# Installation

Installing FireANTs is super easy! We recommend using a virtual environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [venv](https://docs.python.org/3/library/venv.html). 

We recommend getting a fresh conda environment for installing FireANTs.

```bash
conda create -n fireants python=3.7
```

Next, you can install FireANTs locally from source:

```bash
git clone https://github.com/rohitrango/fireants
cd fireants
pip install -e .
```

or install directly from PyPI (may not be the latest iteration):

```bash
pip install fireants
```

It's as easy as that! No more superbuilds. No more cmake. No more waiting for hours to compile. No more esoteric errors. Just install and run! 🚀

!!! warning "Troubleshooting install"

    Still facing trouble? [Open an issue](https://github.com/rohitrango/fireants/issues/new) and we will get back to you as soon as possible.