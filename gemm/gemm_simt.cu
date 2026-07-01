#include <cuda_runtime.h>
#include <stdio.h>

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

template <int BLOCK_SIZE, int BM = 128, int BN = 128, int BK = 16, int WM = 64, int WN = 32, int TM = 8, int TN = 8>
__global__ void gemm_simt(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float A_shared[BM][BK];
    __shared__ float B_shared[BK][BN];

    static_assert(BM % WM == 0 && BN % WN == 0);
    static_assert(WM % TM == 0 && WN % TN == 0);
    static_assert((WM / TM) * (WN / TN) == 32);
    static_assert((BM / WM) * (BN / WN) * 32 == BLOCK_SIZE);

    const int tid = blockDim.x * threadIdx.y + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id / (BN / WN);
    const int warp_n = warp_id % (BN / WN);
    const int thread_m = lane_id / (WN / TN);
    const int thread_n = lane_id % (WN / TN);
    
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float C_c[TN][TM];
    for (int i = 0; i < K; i += BK) {
        #pragma unroll
        for (int j = tid; j < BM * BK; j+=blockDim.x*blockDim.y) {
            int r = j / BK, c = j % BK;
            if (block_row+r < M && i+c < K) {
                A_shared[r][c] = A[(block_row+r) * K + i + c];
            }
        }

        # pragma unroll
        for (int j = tid; j < BK * BN; j+=blockDim.x*blockDim.y) {
            int r = j / BN, c = j % BN;
            if (i+r < K && block_col+c < N) {
                B_shared[r][c] = B[(i+r) * N + block_col+c];
            }
        }

        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];

            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                a[j] = A_shared[warp_m * WM + thread_m * TM + j][kk];
            }

            # pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = B_shared[kk][warp_n * WN + thread_n * TN + j];
            }

            # pragma unroll
            for (int j = 0; j < TM; ++j) {
                #pragma unroll
                for (int k = 0; k < TN; ++k) {
                    C_c[j][k] += a[j] * b[k];
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int c = blockIdx.y * BM + warp_m * WM + thread_m * TM + i;
                const int r = blockIdx.x * BN + warp_n * WN + thread_n * TN + j;
                if (c < M && r < N) {
                    C[c * N + r] = C_c[i][j];
                }
            }
        }

    }

}

int main() {
    constexpr int M = 1000, N = 1000, K = 500;
    constexpr int BM = 128, BN = 128, BK = 16, WM = 64, WN = 32, TM = 8, TN = 8;

    float *A, *B, *C;
    A = new float[M*K];
    B = new float[K*N];
    C = new float[M*N];

    for (int i = 0; i < M*K; ++i) {
        A[i] = 1;
    }

    for (int i = 0; i < K*N; ++i) {
        B[i] = 1;
    }

    float* d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    gemm_simt<16*16, BM, BN, BK, WM, WN, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float v = 0.0f;
            for (int k = 0; k < K; ++k) {
                v += A[i * K + k] * B[k * N + j];
            }

            if (fabs(v - C[i*N + j] > 1e-3)) {
                printf("Error: %f != %f\n", v, C[i*N + j]);
            }
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}