# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 10:04:25
# @Email:  cshzxie@gmail.com

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取CUDA版本
cuda_version = torch.version.cuda
cuda_version_str = cuda_version.replace('.', '') if cuda_version else ''

# Windows特定设置
if os.name == 'nt':
    # 禁用特定警告
    extra_compile_args = {
        'cxx': [
            '/MD',  # 使用动态运行时库
            '/EHsc',  # 启用C++异常处理
            '/O2',  # 优化级别
            '/std:c++17',
            '/DWIN32',
            '/D_WINDOWS',
            '/D_CRT_SECURE_NO_WARNINGS',  # 禁用安全警告
            '/wd4624',  # 禁用4624警告
            '/wd4005',  # 禁用4005警告（宏重定义）
            '/wd4819',  # 禁用代码页警告
            '/wd4251',  # 禁用dll接口警告
        ],
        'nvcc': [
            '-arch=sm_86',  # RTX 3060的架构
            '-gencode=arch=compute_86,code=sm_86',
            '--use-local-env',
            '--cl-version=2019',
            '-Xcompiler', '/MD',
            '-Xcompiler', '/wd4624',
            '-Xcompiler', '/wd4005',
            '-Xcompiler', '/wd4819',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '--extended-lambda',
            '--expt-relaxed-constexpr',
            '-std=c++17',
        ]
    }
else:
    extra_compile_args = {
        'cxx': ['-O2', '-std=c++17'],
        'nvcc': [
            '-arch=sm_86',
            '-gencode=arch=compute_86,code=sm_86',
            '--ptxas-options=-v',
            '-std=c++17',
        ]
    }

# 如果是RTX 30系列，可能需要添加额外的架构
extra_compile_args['nvcc'].extend([
    '-gencode=arch=compute_86,code=compute_86',
])

setup(name='chamfer',
      version='2.0.0',
      ext_modules=[
          CUDAExtension('chamfer', [
              'chamfer_cuda.cpp',
              'chamfer.cu',
          ], extra_compile_args=extra_compile_args),
      ],
      cmdclass={'build_ext': BuildExtension})
