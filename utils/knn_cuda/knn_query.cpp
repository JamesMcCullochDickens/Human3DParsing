#include <torch/extension.h>
#include "knn_query.h"
#include "utils.h"
#include <iostream>


torch::Tensor self_knn_query(const torch::Tensor& xyz, int k) {
    CHECK_CUDA(xyz);
    CHECK_CONTIGUOUS(xyz);

    int b = xyz.size(0);
    int n = xyz.size(1);
    int c = xyz.size(2);

    if (k > 256) {
        throw std::runtime_error("knn_query_kernel_wrapper: k exceeds 256");
    }

    auto idx = torch::zeros({b, n, k}, torch::dtype(torch::kInt).device(xyz.device())).contiguous();

    if (xyz.scalar_type() == torch::kFloat16) {
        self_knn_kernel_wrapper_fp16(b, n, k, c,
                                      reinterpret_cast<const __half *>(xyz.data_ptr<at::Half>()),
                                      idx.data_ptr<int>());
    } else if (xyz.scalar_type() == torch::kFloat32) {
        self_knn_kernel_wrapper(b, n, k, c,
                                 xyz.data_ptr<float>(),
                                 idx.data_ptr<int>());
    } else {
        AT_ERROR("Unsupported tensor dtype! Must be float16 or float32.");
    }

    return idx;
}


torch::Tensor knn_query(const torch::Tensor &new_xyz, const torch::Tensor &xyz, int k) {
    CHECK_CUDA(new_xyz);
    CHECK_CUDA(xyz);
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);

    int b = new_xyz.size(0);
    int m = new_xyz.size(1);
    int n = xyz.size(1);
    int c = xyz.size(2);

    if (k > 256) {
        throw std::runtime_error("knn_query_kernel_wrapper: k exceeds 256");
    }

    auto idx = torch::zeros({b, m, k}, torch::dtype(torch::kInt).device(new_xyz.device())).contiguous();

    if (new_xyz.scalar_type() == torch::kFloat16) {
        knn_query_kernel_wrapper_fp16(b, n, m, k, c,
                                      reinterpret_cast<const __half *>(new_xyz.data_ptr<at::Half>()),
                                      reinterpret_cast<const __half *>(xyz.data_ptr<at::Half>()),
                                      idx.data_ptr<int>());
    } else if (new_xyz.scalar_type() == torch::kFloat32) {
        knn_query_kernel_wrapper(b, n, m, k, c,
                                 new_xyz.data_ptr<float>(),
                                 xyz.data_ptr<float>(),
                                 idx.data_ptr<int>());
    } else {
        AT_ERROR("Unsupported tensor dtype! Must be float16 or float32.");
    }

    return idx;
}


torch::Tensor fnn_query(const torch::Tensor &new_xyz, const torch::Tensor &xyz, int k) {
    CHECK_CUDA(new_xyz);
    CHECK_CUDA(xyz);
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);

    int b = new_xyz.size(0);
    int m = new_xyz.size(1);
    int n = xyz.size(1);
    int c = xyz.size(2);

    if (k > 256) {
        throw std::runtime_error("knn_query_kernel_wrapper: k exceeds 256");
    }

    auto idx = torch::zeros({b, m, k}, torch::dtype(torch::kInt).device(new_xyz.device())).contiguous();

    if (new_xyz.scalar_type() == torch::kFloat16) {
        fnn_query_kernel_wrapper_fp16(b, n, m, k, c,
                                      reinterpret_cast<const __half *>(new_xyz.data_ptr<at::Half>()),
                                      reinterpret_cast<const __half *>(xyz.data_ptr<at::Half>()),
                                      idx.data_ptr<int>());
    } else if (new_xyz.scalar_type() == torch::kFloat32) {
        fnn_query_kernel_wrapper(b, n, m, k, c,
                                 new_xyz.data_ptr<float>(),
                                 xyz.data_ptr<float>(),
                                 idx.data_ptr<int>());
    } else {
        AT_ERROR("Unsupported tensor dtype! Must be float16 or float32.");
    }

    return idx;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_query", &knn_query, "kNN query for point clouds (CUDA)");
    m.def("fnn_query", &fnn_query, "fNN query for point clouds (CUDA)");
    m.def("self_knn_query", &self_knn_query, "KNN within one point cloud (CUDA)");
}