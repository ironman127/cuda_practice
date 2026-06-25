#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define CLUSTER_SIZE 8
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

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) reduction_sum(float* d_data, int n, float* d_result) {
    auto cluster = cg::this_cluster();
    auto thread = cg::this_thread_block();

    int num_float4 = n / 4;               // 完整 float4 的个数
    int stride = blockDim.x * gridDim.x;  // grid-stride 步长
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;

    const float4* d4 = reinterpret_cast<const float4*>(d_data);
    float sum = 0.0f;
    // grid-stride loop：每个线程连续读多个 float4，保持多笔 load 在途以打满带宽
    for (int i = gtid; i < num_float4; i += stride) {
        float4 v = d4[i];
        sum += v.x + v.y + v.z + v.w;
    }
    // 处理结尾不足 4 个的尾部元素
    for (int i = num_float4 * 4 + gtid; i < n; i += stride) {
        sum += d_data[i];
    }

    sum = warp_reduction(sum);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    extern __shared__ float s_sum[];
    if (lane_id == 0) {
        s_sum[warp_id] = sum;
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

    // 4. cluster
    cluster.sync();
    if (cluster.block_rank() == 0 && warp_id == 0) {
        float cluster_sum = 0;
        for (int i = lane_id; i < cluster.dim_blocks().x; i+=WARP_SIZE) {
            float* peer = cluster.map_shared_rank(s_sum, i);
            cluster_sum += *peer;
        }

        cluster_sum = warp_reduction(cluster_sum);
        if (lane_id == 0) {
            atomicAdd(d_result, cluster_sum);
        }
    }

    // 关键：保证所有 peer block 存活到 block 0 读完它们的共享内存
    cluster.sync();
}

int main() {
    int n = 800000001;
    float* h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_data[i] = 0.0001;
    }

    float* d_data;
    cudaMalloc((void**)&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    float* d_result;
    cudaMalloc((void**)&d_result, 1 * sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    size_t block = 1024;
    // 只启动足够填满 GPU 的 block 数，每个线程靠 grid-stride loop 处理多个 float4
    size_t grid = prop.multiProcessorCount * 2;
    grid = ((grid + CLUSTER_SIZE - 1) / CLUSTER_SIZE) * CLUSTER_SIZE;
    size_t smem = (block / WARP_SIZE) * sizeof(float);

    reduction_sum<<<grid, block, smem>>>(d_data, n, d_result);
    CUDA_CHECK(cudaGetLastError());        // ← 捕获启动错误（配置错误等）
    CUDA_CHECK(cudaDeviceSynchronize());   // ← 捕获执行期错误（非法访问等）

    float h_result;
    cudaMemcpy(&h_result, d_result, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %f\n", h_result);

    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += h_data[i];
    }
    printf("Expected: %.0f\n", sum);

    // 相对误差判定：差值除以数值本身，门槛与数据规模无关
    double rel_err = fabs(sum - h_result) / fabs(sum);
    if (rel_err > 1e-4) {
        printf("Error: %f != %.0f (rel err = %.2e)\n", h_result, sum, rel_err);
    } else {
        printf("OK (rel err = %.2e)\n", rel_err);
    }

    cudaFree(d_result);
    cudaFree(d_data);
    return 0;
}