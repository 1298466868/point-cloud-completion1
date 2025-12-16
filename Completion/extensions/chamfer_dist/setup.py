# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 10:04:25
# @Email:  cshzxie@gmail.com

import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_compile_args():
    """获取平台特定的编译参数"""
    if os.name != 'nt':  # Linux/Mac
        return {}
    
    # Windows特定修复
    args = {
        'cxx': [
            # 关键：解决 _addcarry_u64 问题
            '/D__STDC_LIMIT_MACROS',
            '/D__STDC_CONSTANT_MACROS',
            
            # Windows版本
            '/D_WIN32_WINNT=0x0A00',
            '/DWINVER=0x0A00',
            
            # 禁用警告
            '/D_CRT_SECURE_NO_WARNINGS',
            '/D_SCL_SECURE_NO_WARNINGS',
            '/wd4624',  # 析构函数警告
            '/wd4005',  # 宏重定义
        ],
        'nvcc': [
            # 允许新版本编译器
            '-allow-unsupported-compiler',
            
            # 传递宏定义给C++编译器
            '-Xcompiler', '/D__STDC_LIMIT_MACROS',
            '-Xcompiler', '/D__STDC_CONSTANT_MACROS',
            '-Xcompiler', '/D_WIN32_WINNT=0x0A00',
            '-Xcompiler', '/D_CRT_SECURE_NO_WARNINGS',
        ]
    }
    
    # 如果是32位Python，添加x86架构定义
    if sys.maxsize <= 2**32:
        args['cxx'].extend(['/D_M_IX86', '/D_X86_'])
        args['nvcc'].extend(['-Xcompiler', '/D_M_IX86', '-Xcompiler', '/D_X86_'])
    else:
        args['cxx'].extend(['/D_WIN64', '/D_M_AMD64'])
        args['nvcc'].extend(['-Xcompiler', '/D_WIN64', '-Xcompiler', '/D_M_AMD64'])
    
    return args

setup(name='chamfer',
      version='2.0.0',
      ext_modules=[
          CUDAExtension('chamfer', [
              'chamfer_cuda.cpp',
              'chamfer.cu',
          ], extra_compile_args=get_compile_args()),
      ],
      cmdclass={'build_ext': BuildExtension})
