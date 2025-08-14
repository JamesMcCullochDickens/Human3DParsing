#pragma once
#include <torch/extension.h>
#include <cuda_fp16.h>


extern "C" torch::Tensor self_knn_query(const torch::Tensor& xyz, const int k);

extern "C" torch::Tensor knn_query(const torch::Tensor& new_xyz, const torch::Tensor& xyz, const int k);

extern "C" torch::Tensor fnn_query(const torch::Tensor& new_xyz, const torch::Tensor& xyz, int k);

extern "C" void self_knn_kernel_wrapper(int b, int n, int k, int c, const float *xyz, int *idx);

extern "C" void self_knn_kernel_wrapper_fp16(int b, int n, int k, int c, const __half *xyz, int *idx);

extern "C" void knn_query_kernel_wrapper(int b, int n, int m, int k, int c,
                              const float *new_xyz,
                              const float *xyz,
                              int *idx);

extern "C" void knn_query_kernel_wrapper_fp16(int b, int n, int m, int k, int c,
                                   const __half *new_xyz,
                                   const __half *xyz,
                                   int *idx);


extern "C" void fnn_query_kernel_wrapper_fp16(int b, int n, int m, int k, int c,
                                              const __half *new_xyz,
                                              const __half *xyz,
                                              int *idx);

extern "C" void fnn_query_kernel_wrapper(int b, int n, int m, int k, int c,
                                         const float *new_xyz,
                                         const float *xyz,
                                         int *idx);
