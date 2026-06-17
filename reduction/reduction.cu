#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WARP_SIZE 32
#define CUDA_CHECK(call)                                         \
do {                                                             \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(1);                                                 \
    }                                                            \
} while(0)

// 1. Warp Reduction
__device__ float warp_reduction(float sum) {
    for (int s = WARP_SIZE / 2; s > 0; s /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, s);
    }

    return sum;
}

__global__ void reduction_sum(float* d_data, int n, float* d_result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 单个线程读多条数据
    float4 v = i * 4 + 3 < n ? *reinterpret_cast<const float4*> (&d_data[i*4]): make_float4(0, 0, 0, 0);
    float sum = v.x + v.y + v.z + v.w;
    sum = warp_reduction(sum);

    extern __shared__ float s_sum[];
    if (threadIdx.x % WARP_SIZE == 0) {
        s_sum[threadIdx.x /  WARP_SIZE] = sum;
    }
    __syncthreads();

    // 3. block reduction
    int s_size = blockDim.x / WARP_SIZE;
    for (int s = s_size / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }


    if (threadIdx.x == 0) {
        atomicAdd(d_result, s_sum[0]);
    }
}

int main() {
    int n = 1000000;
    float h_data[n];
    for (int i = 0; i < n; i++) {
        h_data[i] = 1;
    }

    float* d_data;
    cudaMalloc((void**)&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    float* d_result;
    cudaMalloc((void**)&d_result, 1 * sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    reduction_sum<<<(n + 1024 - 1) / 1024, 1024, 8 * sizeof(float)>>>(d_data, n, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_result;
    cudaMemcpy(&h_result, d_result, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %f\n", h_result);

    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += h_data[i];
    }
    printf("Expected: %.0f\n", sum);

    if (fabs(sum - h_result) > 0.001) {
        printf("Error: %f != %.0f\n", h_result, sum);
    }

    cudaFree(d_result);
    cudaFree(d_data);
    return 0;
}