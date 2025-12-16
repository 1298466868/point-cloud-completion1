# -*- coding: utf-8 -*-
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# PyTorch 2.4 + Windows 必需的定义
extra_compile_args = {
    'cxx': [
        '/DNOMINMAX',           # Windows: 防止min/max宏冲突
        '/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS',  # 禁用C++17弃用警告
        '/wd4005',              # 禁用宏重定义警告（针对HAVE_SNPRINTF）
        '/wd4819',              # 禁用代码页警告
    ],
    'nvcc': [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__', 
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',  # PyTorch 2.x 新增
        '-D__CUDA_NO_HALF2_OPERATORS__',
        '--expt-relaxed-constexpr',  # CUDA 11.8 需要
    ]
}

setup(
    name='chamfer',
    version='2.0.0',
    ext_modules=[
        CUDAExtension('chamfer', [
            'chamfer_cuda.cpp',
            'chamfer.cu',
        ], extra_compile_args=extra_compile_args),
    ],
    cmdclass={'build_ext': BuildExtension}
)
          extra_compile_args=extra_compile_args),
      ],
      cmdclass={'build_ext': BuildExtension})
