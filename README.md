# :fire: FireANTs: Adaptive Riemannian Optimization for Multi-Scale Diffeomorphic Registration

The FireANTs library is a lightweight registration package for Riemannian diffeomorphic registration on GPUs.

## Installation 
To use the FireANTs package, you can either clone the repository and install the package locally or install the package directly from PyPI.
We recommend using a fresh Anaconda/Miniconda environment to install the package.
```
conda create -n fireants python=3.7
```

To install FireANTs locally:
```
git clone https://github.com/rohitrango/fireants
cd fireants
pip install -e .
```

Or install from PyPI:
```
pip install fireants
```

## Tutorial
To check out some of the tutorials, check out the `tutorials/` directory for usage.
Alternatively, to reproduce the results in the [paper](https://arxiv.org/abs/2404.01249) checkout the `fireants/scripts/` directory.

## Documentation
You can also check out the [Documentation](https://fireants.readthedocs.io/en/latest/). Feel free to reach out to me for improvements in the documentation.

## Datasets
In the paper, we use the datasets as following: 
* Klein's evaluation of 14 non-linear registration algorithms: [here](https://www.synapse.org/#!Synapse:syn3251018)
* EMPIRE10 lung registration challenge: [here](https://empire10.grand-challenge.org/)
* Expansion Microscopy dataset: [here](https://rnr-exm.grand-challenge.org/)

## Contributing
Feel free to [add issues](https://github.com/rohitrango/fireants/issues/new) or [pull requests](https://github.com/rohitrango/fireants/compare) to the repository. We welcome contributions to the package.

## License
Please refer to the [LICENSE](LICENSE) file for the license details, especially pertaining to redistribution of code and derivative works.

## Citation

If you use FireANTs in your research, please cite the following paper:

```
@article{jena2024fireants,
  title={FireANTs: Adaptive Riemannian Optimization for Multi-Scale Diffeomorphic Registration},
  author={Jena, Rohit and Chaudhari, Pratik and Gee, James C},
  journal={arXiv preprint arXiv:2404.01249},
  year={2024}
}
```

If you use FireANTs-as-a-layer ([Deep Implicit Optimization](https://www.sciencedirect.com/science/article/pii/S1361841525001240?via%3Dihub), [code](https://github.com/rohitrango/DIO)), cite the following paper:
```
@article{jena2025deep,
  title={Deep implicit optimization enables robust learnable features for deformable image registration},
  author={Jena, Rohit and Chaudhari, Pratik and Gee, James C},
  journal={Medical Image Analysis},
  volume={103},
  pages={103577},
  year={2025},
  publisher={Elsevier}
}
```
