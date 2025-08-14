#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <cuda_fp16.h>

#define MAX_K 128

__global__ void knn_kernel(int b, int n, int m, int k, int c,
                           const float *__restrict__ new_xyz,
                           const float *__restrict__ xyz,
                           int *__restrict__ idx) {
    int idx_query = blockIdx.x * blockDim.x + threadIdx.x;
    int total_queries = b * m;
    if (idx_query >= total_queries) return;

    int batch_index = idx_query / m;
    int j = idx_query % m;

    const float* batch_new_xyz = new_xyz + batch_index * m * c;
    const float* batch_xyz     = xyz     + batch_index * n * c;
    int* batch_idx             = idx     + batch_index * m * k;

    float best_dist[MAX_K];
    int best_idx[MAX_K];

    for (int t = 0; t < k; t++) {
        best_dist[t] = FLT_MAX;
        best_idx[t] = -1;
    }

    for (int i = 0; i < n; i++) {
        float d2 = 0;
        for (int d = 0; d < c; d++) {
            float diff = batch_new_xyz[j * c + d] - batch_xyz[i * c + d];
            d2 += diff * diff;
        }

        if (d2 < best_dist[k - 1] ||
           (d2 == best_dist[k - 1] && i < best_idx[k - 1])) {
            int insert_pos = k - 1;
            while (insert_pos > 0 &&
                   (best_dist[insert_pos - 1] > d2 ||
                   (best_dist[insert_pos - 1] == d2 && best_idx[insert_pos - 1] > i))) {
                best_dist[insert_pos] = best_dist[insert_pos - 1];
                best_idx[insert_pos] = best_idx[insert_pos - 1];
                insert_pos--;
            }
            best_dist[insert_pos] = d2;
            best_idx[insert_pos] = i;
        }
    }

    for (int t = 0; t < k; t++) {
        batch_idx[j * k + t] = best_idx[t];
    }
}

__global__ void knn_kernel_fp16(int b, int n, int m, int k, int c,
                                const __half *__restrict__ new_xyz,
                                const __half *__restrict__ xyz,
                                int *__restrict__ idx) {
    int idx_query = blockIdx.x * blockDim.x + threadIdx.x;
    int total_queries = b * m;
    if (idx_query >= total_queries) return;

    int batch_index = idx_query / m;
    int j = idx_query % m;

    const __half* batch_new_xyz = new_xyz + batch_index * m * c;
    const __half* batch_xyz     = xyz     + batch_index * n * c;
    int* batch_idx              = idx     + batch_index * m * k;

    float query_coords[32];  // assuming c <= 32
    for (int d = 0; d < c; d++) {
        query_coords[d] = __half2float(batch_new_xyz[j * c + d]);
    }

    float best_dist[MAX_K];
    int best_idx[MAX_K];

    for (int t = 0; t < k; t++) {
        best_dist[t] = 1e10f;
        best_idx[t] = -1;
    }

    for (int i = 0; i < n; i++) {
        float d2 = 0;
        for (int d = 0; d < c; d++) {
            float ref_coord = __half2float(batch_xyz[i * c + d]);
            float diff = query_coords[d] - ref_coord;
            d2 += diff * diff;
        }

        if (d2 < best_dist[k - 1] ||
           (d2 == best_dist[k - 1] && i < best_idx[k - 1])) {
            int insert_pos = k - 1;
              while (insert_pos > 0 &&
                   (best_dist[insert_pos - 1] > d2 ||
                   (best_dist[insert_pos - 1] == d2 && best_idx[insert_pos - 1] > i))) {
                best_dist[insert_pos] = best_dist[insert_pos - 1];
                best_idx[insert_pos] = best_idx[insert_pos - 1];
                insert_pos--;
            }
            best_dist[insert_pos] = d2;
            best_idx[insert_pos] = i;
        }
    }

    for (int t = 0; t < k; t++) {
        batch_idx[j * k + t] = best_idx[t];
    }
}


__global__ void self_knn_kernel(int batch_size, int num_points,
                                int num_channels, int k,
                                const float* __restrict__ pcd,
                                int* idx) {
    size_t thread_num = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    size_t total_queries = static_cast<size_t>(batch_size * num_points);
    if (thread_num >= total_queries) return;

    int batch_index = thread_num / num_points;
    int idx_query = thread_num % num_points;

    const float* batch_pcd = pcd + batch_index * num_points * num_channels;
    int* batch_idx = idx + batch_index * num_points * k;

    float best_dist[MAX_K];
    int best_idx[MAX_K];

    best_dist[0] = 0.0f;
    best_idx[0] = idx_query;
    for (int t = 0; t < k; t++) {
        best_dist[t] = FLT_MAX;;
        best_idx[t] = -1;
    }

    for (int i = 0; i < num_points; i++) {
        float d2 = 0;
        for (int d = 0; d < num_channels; d++) {
            float diff = batch_pcd[idx_query * num_channels + d] - batch_pcd[i * num_channels + d];
            d2 += diff * diff;
        }

        if (d2 < best_dist[k - 1]) {
            int insert_pos = k - 1;
            while (insert_pos > 0 && best_dist[insert_pos - 1] > d2) {
                best_dist[insert_pos] = best_dist[insert_pos - 1];
                best_idx[insert_pos] = best_idx[insert_pos - 1];
                insert_pos--;
            }
            best_dist[insert_pos] = d2;
            best_idx[insert_pos] = i;
        }
    }

    for (int t = 0; t < k; t++) {
        batch_idx[idx_query * k + t] = best_idx[t];
    }
}

__global__ void self_knn_kernel_fp16(int batch_size, int num_points,
                                     int num_channels, int k,
                                     const __half* __restrict__ pcd,
                                     int* idx) {
    size_t thread_num = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    size_t total_queries = static_cast<size_t>(batch_size * num_points);
    if (thread_num >= total_queries) return;

    int batch_index = thread_num / num_points;
    int idx_query   = thread_num % num_points;

    const __half* batch_pcd = pcd + batch_index * num_points * num_channels;
    int* batch_idx          = idx + batch_index * num_points * k;

    float best_dist[MAX_K];
    int best_idx[MAX_K];

    for (int t = 0; t < k; t++) {
        best_dist[t] = FLT_MAX;
        best_idx[t] = -1;
    }

    float query_coords[32];  // assuming num_channels <= 32
    for (int d = 0; d < num_channels; d++) {
        query_coords[d] = __half2float(batch_pcd[idx_query * num_channels + d]);
    }

    for (int i = 0; i < num_points; i++) {
        float d2 = 0.0f;
        for (int d = 0; d < num_channels; d++) {
            float ref_coord = __half2float(batch_pcd[i * num_channels + d]);
            float diff = query_coords[d] - ref_coord;
            d2 += diff * diff;
        }

        if (d2 < best_dist[k - 1]) {
            int insert_pos = k - 1;
            while (insert_pos > 0 && best_dist[insert_pos - 1] > d2) {
                best_dist[insert_pos] = best_dist[insert_pos - 1];
                best_idx[insert_pos] = best_idx[insert_pos - 1];
                insert_pos--;
            }
            best_dist[insert_pos] = d2;
            best_idx[insert_pos] = i;
        }
    }

    for (int t = 0; t < k; t++) {
        batch_idx[idx_query * k + t] = best_idx[t];
    }
}


__global__ void fnn_kernel(int b, int n, int m, int k, int c,
                                  const float *__restrict__ new_xyz,
                                  const float *__restrict__ xyz,
                                  int *__restrict__ idx) {
    int idx_query = blockIdx.x * blockDim.x + threadIdx.x;
    int total_queries = b * m;
    if (idx_query >= total_queries) return;

    int batch_index = idx_query / m;
    int j = idx_query % m;

    const float* batch_new_xyz = new_xyz + batch_index * m * c;
    const float* batch_xyz     = xyz     + batch_index * n * c;
    int* batch_idx             = idx     + batch_index * m * k;

    float worst_dist[MAX_K];
    int worst_idx[MAX_K];

    for (int t = 0; t < k; t++) {
        worst_dist[t] = -1.0f;  // small initial value
        worst_idx[t] = -1;
    }

    for (int i = 0; i < n; i++) {
        float d2 = 0;
        for (int d = 0; d < c; d++) {
            float diff = batch_new_xyz[j * c + d] - batch_xyz[i * c + d];
            d2 += diff * diff;
        }

        if (d2 > worst_dist[k - 1]) {
            int insert_pos = k - 1;
            while (insert_pos > 0 && worst_dist[insert_pos - 1] < d2) {
                worst_dist[insert_pos] = worst_dist[insert_pos - 1];
                worst_idx[insert_pos] = worst_idx[insert_pos - 1];
                insert_pos--;
            }
            worst_dist[insert_pos] = d2;
            worst_idx[insert_pos] = i;
        }
    }

    for (int t = 0; t < k; t++) {
        batch_idx[j * k + t] = worst_idx[t];
    }
}


__global__ void fnn_kernel_fp16(int b, int n, int m, int k, int c,
                                       const __half *__restrict__ new_xyz,
                                       const __half *__restrict__ xyz,
                                       int *__restrict__ idx) {
    int idx_query = blockIdx.x * blockDim.x + threadIdx.x;
    int total_queries = b * m;
    if (idx_query >= total_queries) return;

    int batch_index = idx_query / m;
    int j = idx_query % m;

    const __half* batch_new_xyz = new_xyz + batch_index * m * c;
    const __half* batch_xyz     = xyz     + batch_index * n * c;
    int* batch_idx              = idx     + batch_index * m * k;

    float query_coords[32];  // assuming c <= 32
    for (int d = 0; d < c; d++) {
        query_coords[d] = __half2float(batch_new_xyz[j * c + d]);
    }

    float worst_dist[MAX_K];
    int worst_idx[MAX_K];

    for (int t = 0; t < k; t++) {
        worst_dist[t] = -1.0f;  // init with minimum possible (dist ≥ 0)
        worst_idx[t] = -1;
    }

    for (int i = 0; i < n; i++) {
        float d2 = 0;
        for (int d = 0; d < c; d++) {
            float ref_coord = __half2float(batch_xyz[i * c + d]);
            float diff = query_coords[d] - ref_coord;
            d2 += diff * diff;
        }

        if (d2 > worst_dist[k - 1]) {
            int insert_pos = k - 1;
            while (insert_pos > 0 && worst_dist[insert_pos - 1] < d2) {
                worst_dist[insert_pos] = worst_dist[insert_pos - 1];
                worst_idx[insert_pos] = worst_idx[insert_pos - 1];
                insert_pos--;
            }
            worst_dist[insert_pos] = d2;
            worst_idx[insert_pos] = i;
        }
    }

    for (int t = 0; t < k; t++) {
        batch_idx[j * k + t] = worst_idx[t];
    }
}

extern "C" void self_knn_kernel_wrapper(int b, int n, int k, int c, const float *xyz, int *idx) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int total_queries = b * n;
    int threads = 256;
    int blocks = (total_queries + threads - 1) / threads;

    self_knn_kernel<<<blocks, threads, 0, stream>>>(b, n, c, k, xyz, idx);
    CUDA_CHECK_ERRORS();
}


extern "C" void self_knn_kernel_wrapper_fp16(int b, int n, int k, int c,
                                              const __half *xyz, int *idx) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int total_queries = b * n;
    int threads = 256;
    int blocks = (total_queries + threads - 1) / threads;
    self_knn_kernel_fp16<<<blocks, threads, 0, stream>>>(b, n, c, k, xyz, idx);
    CUDA_CHECK_ERRORS();
}



extern "C" void fnn_query_kernel_wrapper(int b, int n, int m, int k, int c,
                                         const float *new_xyz,
                                         const float *xyz,
                                         int *idx) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int total_queries = b * m;
    int threads = 256;
    int blocks = (total_queries + threads - 1) / threads;

    fnn_kernel<<<blocks, threads, 0, stream>>>(b, n, m, k, c, new_xyz, xyz, idx);
    CUDA_CHECK_ERRORS();
}


extern "C" void knn_query_kernel_wrapper(int b, int n, int m, int k, int c,
                                         const float *new_xyz,
                                         const float *xyz,
                                         int *idx) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int total_queries = b * m;
    int threads = 256;
    int blocks = (total_queries + threads - 1) / threads;

    knn_kernel<<<blocks, threads, 0, stream>>>(b, n, m, k, c, new_xyz, xyz, idx);
    CUDA_CHECK_ERRORS();
}


extern "C" void knn_query_kernel_wrapper_fp16(int b, int n, int m, int k, int c,
                                              const __half *new_xyz,
                                              const __half *xyz,
                                              int *idx) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int total_queries = b * m;
    int threads = 256;
    int blocks = (total_queries + threads - 1) / threads;

    knn_kernel_fp16<<<blocks, threads, 0, stream>>>(b, n, m, k, c, new_xyz, xyz, idx);
    CUDA_CHECK_ERRORS();
}


extern "C" void fnn_query_kernel_wrapper_fp16(int b, int n, int m, int k, int c,
                                              const __half *new_xyz,
                                              const __half *xyz,
                                              int *idx) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int total_queries = b * m;
    int threads = 256;
    int blocks = (total_queries + threads - 1) / threads;

    fnn_kernel_fp16<<<blocks, threads, 0, stream>>>(b, n, m, k, c, new_xyz, xyz, idx);
    CUDA_CHECK_ERRORS();
}
