# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 10:04:25
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Windows 特定的编译选项
extra_compile_args = {
    'cxx': [],
    'nvcc': [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
}

# 如果是 Windows，添加更多选项
if os.name == 'nt':
    extra_compile_args['cxx'].append('/MP')  # 多进程编译
    extra_compile_args['cxx'].append('/std:c++17')  # 使用 C++17 标准
    extra_compile_args['nvcc'].append('-Xcompiler=/wd4819')  # 禁用特定警告

setup(name='chamfer',
      version='2.0.0',
      ext_modules=[
          CUDAExtension('chamfer', [
              'chamfer_cuda.cpp',
              'chamfer.cu',
          ],
          extra_compile_args=extra_compile_args),
      ],
      cmdclass={'build_ext': BuildExtension})
