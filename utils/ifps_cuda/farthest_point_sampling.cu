#include <cuda_runtime.h>
#include <torch/extension.h>

#define THREADS_PER_BLOCK 512
#define BUFFER_SIZE 3072

__global__ void farthest_point_sampling_kernel(
    int B, int N, int M,
    const float* __restrict__ dataset,  // [B, N, 3]
    float* __restrict__ temp,          // [B, N]
    int* __restrict__ idxs             // [B, M]
) {
    if (M <= 0) return;

    __shared__ float dists[THREADS_PER_BLOCK];
    __shared__ int dists_i[THREADS_PER_BLOCK];
    __shared__ float buf[BUFFER_SIZE * 3];

    for (int i = blockIdx.x; i < B; i += gridDim.x) {
        int old = 0;
        if (threadIdx.x == 0)
            idxs[i * M + 0] = old;

        for (int j = threadIdx.x; j < N; j += blockDim.x)
            temp[i * N + j] = 1e38f; // initialize to a large value

        for (int j = threadIdx.x; j < min(BUFFER_SIZE, N) * 3; j += blockDim.x)
            buf[j] = dataset[i * N * 3 + j];

        __syncthreads();

        for (int j = 1; j < M; j++) {
            int besti = 0;
            float best = -1.0f;

            float x1 = dataset[i * N * 3 + old * 3 + 0];
            float y1 = dataset[i * N * 3 + old * 3 + 1];
            float z1 = dataset[i * N * 3 + old * 3 + 2];

            for (int k = threadIdx.x; k < N; k += blockDim.x) {
                float td = temp[i * N + k];

                float x2, y2, z2;
                if (k < BUFFER_SIZE) {
                    x2 = buf[k * 3 + 0];
                    y2 = buf[k * 3 + 1];
                    z2 = buf[k * 3 + 2];
                } else {
                    x2 = dataset[i * N * 3 + k * 3 + 0];
                    y2 = dataset[i * N * 3 + k * 3 + 1];
                    z2 = dataset[i * N * 3 + k * 3 + 2];
                }

                float dx = x2 - x1;
                float dy = y2 - y1;
                float dz = z2 - z1;
                float d = dx * dx + dy * dy + dz * dz;
                float d2 = fminf(d, td);

                if (d2 != td)
                    temp[i * N + k] = d2;

                if (d2 > best) {
                    best = d2;
                    besti = k;
                }
            }

            dists[threadIdx.x] = best;
            dists_i[threadIdx.x] = besti;
            __syncthreads();

            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (threadIdx.x < offset) {
                    if (dists[threadIdx.x] < dists[threadIdx.x + offset]) {
                        dists[threadIdx.x] = dists[threadIdx.x + offset];
                        dists_i[threadIdx.x] = dists_i[threadIdx.x + offset];
                    }
                }
                __syncthreads();
            }

            old = dists_i[0];
            if (threadIdx.x == 0)
                idxs[i * M + j] = old;
        }
    }
}

void fps_launcher(torch::Tensor dataset, torch::Tensor idxs, torch::Tensor temp, int M) {
    int B = dataset.size(0);
    int N = dataset.size(1);

    const int threads = THREADS_PER_BLOCK;
    farthest_point_sampling_kernel<<<B, threads>>>(
        B, N, M,
        dataset.data_ptr<float>(),
        temp.data_ptr<float>(),
        idxs.data_ptr<int>()
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error in fps_launcher: %s\n", cudaGetErrorString(err));
}
