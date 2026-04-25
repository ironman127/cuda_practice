#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello from block %d, thread %d!\n", blockIdx.x, threadIdx.x);
}

int main() {
    // 启动 2 个 block，每个 block 4 个 thread
    hello_kernel<<<2, 4>>>();

    // 等 GPU 跑完（printf 需要 sync 才能看到）
    cudaDeviceSynchronize();

    // 检查是否有错
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Done!\n");
    return 0;
}
