#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                         \
do {                                                             \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(1);                                                 \
    }                                                            \
} while(0)
#define WARP_SIZE 32
#define CLUSTER_SIZE 8

__global__ void softmax_kernel(float* data, float* result, size_t n, int s_size) {
    int stride = blockDim.x;
    
    float max_val = -FLT_MAX;
    float sum = 0;

    for (size_t i = threadIdx.x; i < n; i+=stride) {
        float new_max_val = fmaxf(max_val, data[i]);
        sum = sum * expf(max_val - new_max_val) + expf(data[i] - new_max_val);
        max_val = new_max_val;
    }

    auto combine = [] (float* m, float* s, float* m2, float* s2) {
        float new_max_val = fmaxf(*m, *m2);
        *s = *s * expf(*m - new_max_val) + *s2 * expf(*m2 - new_max_val);
        *m = new_max_val;
    };

    for (int s = WARP_SIZE/2; s > 0; s >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, max_val, s);
        float s2 = __shfl_down_sync(0xffffffff, sum, s);

        combine(&max_val, &sum, &m2, &s2);
    }

    extern __shared__ float s_mem[];
    float* s_sum = s_mem;
    float* s_max = s_mem + s_size;

    if (threadIdx.x % WARP_SIZE == 0) {
        s_max[threadIdx.x /  WARP_SIZE] = max_val;
        s_sum[threadIdx.x /  WARP_SIZE] = sum;
    }

    for (int s = s_size / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            combine(&s_max[threadIdx.x], &s_sum[threadIdx.x], &s_max[threadIdx.x+s], &s_sum[threadIdx.x+s]);
        }

        __syncthreads();
    }

    sum = s_sum[0];
    max_val = s_max[0];

    for (size_t i = threadIdx.x; i < n; i+=stride) {
        result[i] = expf(data[i] - max_val) / sum;
    }
}

int main() {
    size_t n = 100000000;
    float* h_data = (float*)malloc(n * sizeof(float));
    float* h_result = (float*)malloc(n * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        h_data[i] = 1;
    }

    h_data[0] = 2;

    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    float* d_result;
    cudaMalloc(&d_result, n * sizeof(float));
    cudaMemset(d_result, 0, n * sizeof(float));

    int block_size = 1024;
    int grid_size = 1;
    int s_size = block_size;
    softmax_kernel<<<grid_size, block_size, s_size * 2 * sizeof(float)>>>(d_data, d_result, n, s_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(h_result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    float max_val = -FLT_MAX;
    float sum = 0;

    for (size_t i = 0; i < n; i++) {
        max_val = fmaxf(max_val, h_data[i]);
    }

    for (size_t i = 0; i < n; i++) {
        sum += expf(h_data[i] - max_val);
    }

    for (size_t i = 0; i < n; i++) {
        float cur_val = expf(h_data[i] - max_val) / sum;
        if (fabs(cur_val - h_result[i]) > 1e-3) {
            printf("Error at %zu: %f != %f\n", i, cur_val, h_result[i]);
        }
    }

    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
}