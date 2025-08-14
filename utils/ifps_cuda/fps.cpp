#include <torch/extension.h>

void fps_launcher(torch::Tensor dataset, torch::Tensor idxs, torch::Tensor temp, int M);

torch::Tensor farthest_point_sampling(torch::Tensor dataset, int M) {
    TORCH_CHECK(dataset.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(dataset.dim() == 3 && dataset.size(2) == 3, "Shape must be (B, N, 3)");
    TORCH_CHECK(dataset.scalar_type() == torch::kFloat32, "Only float32 supported");

    int B = dataset.size(0);
    int N = dataset.size(1);

    auto idxs = torch::empty({B, M}, dataset.options().dtype(torch::kInt32));
    auto temp = torch::empty({B, N}, dataset.options().dtype(torch::kFloat32));

    fps_launcher(dataset, idxs, temp, M);
    return idxs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sampling", &farthest_point_sampling, "Farthest Point Sampling (CUDA)");
}
