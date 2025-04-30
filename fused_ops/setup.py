from setuptools import setup, Extension
from torch.utils import cpp_extension
import torch
import os

include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include')

setup(
    name='fireants_fused_ops',
    version='0.1.0',
    description='Fused CUDA operations for FireANTs',
    author='Rohit Jena',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='fireants_fused_ops',
            sources=[
                'src/src.cpp',
                'src/CrossCorrelation.cu',  
                'src/FusedGridSampler.cu',
                'src/FusedGridComposer.cu',
                'src/FusedGenerateGrid.cu',
                'src/AdamUtils.cu',
                'src/GaussianBlurFFT.cu',
                'src/MutualInformation.cu'
            ],
            include_dirs=[include_dir] + torch.utils.cpp_extension.include_paths(),
            library_dirs=torch.utils.cpp_extension.library_paths(),
            # extra_compile_args={
            #     'cxx': ['-O3'],
            #     'nvcc': ['-O3']
            # }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=['torch>=2.3.0'],
)
