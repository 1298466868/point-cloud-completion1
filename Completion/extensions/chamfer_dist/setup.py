# -*- coding: utf-8 -*-
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
import sys

# 获取CUDA版本信息
cuda_available = False
try:
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else None
except:
    pass

# 设置MSVC编译参数（关键！）
extra_compile_args = {
    'cxx': [
        '/wd4624',    # 禁用"已将析构函数隐式定义为已删除"
        '/wd4068',    # 禁用未知pragma警告
        '/wd4819',    # 禁用编码问题警告
        '/wd4820',    # 禁用字节填充警告
        '/wd4661',    # 禁用模板特化警告
        '/wd4251',    # 禁用DLL接口警告
        '/wd4275',    # 禁用DLL基类警告
        '/wd4190',    # 禁用C-linkage警告
        '/wd5030',    # 禁用属性警告
        '/O2',        # 优化级别
        '/MD',        # 使用动态运行时
        '/DNDEBUG',   # 禁用调试
        '/D_WINDOWS', # Windows定义
    ],
    'nvcc': [
        '-Xcompiler', '/wd4624',
        '-Xcompiler', '/wd4068',
        '-Xcompiler', '/wd4819',
        '-Xcompiler', '/wd4820',
        '-Xcompiler', '/wd4661',
        '-Xcompiler', '/wd4251',
        '-Xcompiler', '/wd4275',
        '-Xcompiler', '/wd4190',
        '-Xcompiler', '/wd5030',
        '--use_fast_math',
        '-lineinfo',
        '-O3',
        '-DNDEBUG',
        '-D_WINDOWS',
        '--expt-relaxed-constexpr',
        '-gencode=arch=compute_50,code=sm_50',
        '-gencode=arch=compute_52,code=sm_52',
        '-gencode=arch=compute_60,code=sm_60',
        '-gencode=arch=compute_61,code=sm_61',
        '-gencode=arch=compute_70,code=sm_70',
        '-gencode=arch=compute_75,code=sm_75',
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
        '-gencode=arch=compute_86,code=compute_86',
    ]
}

setup(
    name='chamfer',
    version='2.0.0',
    ext_modules=[
        CUDAExtension(
            'chamfer',
            [
                'chamfer_cuda.cpp',
                'chamfer.cu',
            ],
            extra_compile_args=extra_compile_args,
            library_dirs=[],
            libraries=[],
            include_dirs=[],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(
            use_ninja=True,
            no_python_abi_suffix=False,
        )
    }
)
