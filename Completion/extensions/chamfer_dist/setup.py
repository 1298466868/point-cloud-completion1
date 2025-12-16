# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 10:04:25
# @Email:  cshzxie@gmail.com

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Windows特定设置
if os.name == 'nt':
    extra_compile_args = {
        'cxx': ['/MD', '/EHsc', '/O2', '/std:c++17'],
        'nvcc': [
            '-arch=sm_86',  # RTX 3060的架构
            '--ptxas-options=-v',
            '--compiler-options', "'/EHsc'",
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
    }
else:
    extra_compile_args = {
        'cxx': ['-O2', '-std=c++17'],
        'nvcc': [
            '-arch=sm_86',
            '--ptxas-options=-v',
        ]
    }

setup(name='chamfer',
      version='2.0.0',
      ext_modules=[
          CUDAExtension('chamfer', [
              'chamfer_cuda.cpp',
              'chamfer.cu',
          ], extra_compile_args=extra_compile_args),
      ],
      cmdclass={'build_ext': BuildExtension})
