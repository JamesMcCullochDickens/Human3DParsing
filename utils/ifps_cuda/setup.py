from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='farthest_point_sampling',
    ext_modules=[
        CUDAExtension(
            name='farthest_point_sampling',
            sources=['fps.cpp', 'farthest_point_sampling.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)