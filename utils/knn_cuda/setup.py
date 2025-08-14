from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="knn_cuda",
    ext_modules=[
        CUDAExtension(
            name="knn_cuda",
            sources=["knn_query.cpp", "knn_query_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": [
                    "-O2", "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__"
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

