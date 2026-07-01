#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>

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

__device__ __forceinline__ int swizzle(const int row, const int col, const int cols) {
    return col ^ (row & (cols - 1)); // 需要限制在列索引的最大位宽内
}

template<int BM, int BK, int BN, int WM, int WN, int TM, int TN>
__device__ void compute(const int warp_m, const int warp_n, const int thread_m, const int thread_n, float A[][BK], const float B[][BN], float C_c[][TN]) {
    float a[TM], b[TN];
    for (int kk = 0; kk < BK; ++kk) {    
        #pragma unroll
        for (int j = 0; j < TM; ++j) {
            a[j] = A[warp_m * WM + thread_m * TM + j][swizzle(warp_m * WM + thread_m * TM + j, kk, BK)];
        }

        # pragma unroll
        for (int j = 0; j < TN; ++j) {
            b[j] = B[kk][swizzle(kk, warp_n * WN + thread_n * TN + j, BN)];
        }

        # pragma unroll
        for (int j = 0; j < TM; ++j) {
            #pragma unroll
            for (int k = 0; k < TN; ++k) {
                C_c[j][k] += a[j] * b[k];
            }
        }
    }
}

template<int BM, int BK, int BN>
__device__ void cpy2share(const int i, const int tid, const int block_row, const int block_col, cuda::pipeline<cuda::thread_scope_block> &pipe, const float *A, const float *B, int M, int K, int N, float sA[][BK], float sB[][BN]) {
    // 【生产者：申请一个流水 stage】
    // 请求一块可写的缓冲。若两级 buffer 都还被消费者占用，这里会阻塞等待，
    // 从而保证不会覆盖正在被 compute 读取的 buffer。
    pipe.producer_acquire();

    #pragma unroll
    for (int j = tid; j < BM * BK; j+=blockDim.x*blockDim.y) {
        int r = j / BK, c = j % BK;
        if (block_row+r < M && i+c < K) {
            // 发起 global->shared 的异步拷贝（cp.async），不阻塞、不经过寄存器；
            // 该拷贝被登记到当前 stage，稍后由 consumer_wait 统一等待其完成。
            cuda::memcpy_async(&sA[r][swizzle(r, c, BK)], &A[(block_row+r) * K + i + c], sizeof(float), pipe);
        } else {
            // 越界部分直接同步写 0（padding），不走异步。
            sA[r][swizzle(r, c, BK)] = 0.0f;
        }
    }

    # pragma unroll
    for (int j = tid; j < BK * BN; j+=blockDim.x*blockDim.y) {
        int r = j / BN, c = j % BN;
        if (i+r < K && block_col+c < N) {
            cuda::memcpy_async(&sB[r][swizzle(r, c, BN)], &B[(i+r) * N + block_col+c], sizeof(float), pipe);
        } else {
            sB[r][swizzle(r, c, BN)] = 0.0f;
        }
    }

    // 【提交 stage】把上面这一批异步拷贝打包成一个"批次"提交给流水线；
    // 它只是"登记完成"，并不等待拷贝真正结束。
    pipe.producer_commit();
}

template <int BLOCK_SIZE, int BM = 128, int BN = 128, int BK = 16, int WM = 64, int WN = 32, int TM = 8, int TN = 8>
__global__ void gemm_simt(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float A_shared[2][BM][BK];
    __shared__ float B_shared[2][BK][BN];

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

    // 【流水线状态】放在 shared memory，供全 block 共享。
    // thread_scope_block: 该流水由整个 block 协作推进；模板参数 2 = 两级 buffer（双缓冲）。
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> state;
    // 拿到"当前 block 全部线程"这个协作组，pipeline 用它做组内同步。
    auto block = cooperative_groups::this_thread_block();
    // 用协作组 + 共享状态构造 block 级 pipeline。
    auto pipe  = cuda::make_pipeline(block, &state);

    float C_c[TM][TN] = {0};
    // 【预取】先把第 0 块 tile 的异步拷贝发起并提交（填入 buffer 0），
    // 这样进入循环时第 0 块已在"飞行中"，为计算/搬运重叠打底。
    cpy2share<BM, BK, BN>(0, tid, block_row, block_col, pipe, A, B, M, K, N, A_shared[0], B_shared[0]);

    for (int i = 0; i < K; i += BK) {
        int cur = (i / BK) % 2;         // 本轮要计算的 buffer（已在上一轮/预取时提交）
        int next = (i / BK + 1) % 2;    // 本轮要预取的下一块 buffer

        // 【预取下一块】发起下一块的异步拷贝，与本轮的 compute 重叠执行。
        // 只有下一块存在时才预取，否则不再提交新 stage（保证 commit 与 wait 数量一致）。
        if (i + BK < K) {
            cpy2share<BM, BK, BN>(i + BK, tid, block_row, block_col, pipe, A, B, M, K, N, A_shared[next], B_shared[next]);
        }

        // 【消费者：等待当前块就绪】阻塞直到 cur 这一 stage 的异步拷贝全部完成
        // （block 级，内部含 barrier，保证所有线程写入的数据都已可见）。
        pipe.consumer_wait();
        compute<BM, BK, BN, WM, WN, TM, TN>(warp_m, warp_n, thread_m, thread_n, A_shared[cur], B_shared[cur], C_c);
        // 【释放 stage】计算读完后归还该 buffer，使其可被后续 producer_acquire 复用。
        pipe.consumer_release();

        __syncthreads();
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

int main() {
    constexpr int M = 10000, N = 10000, K = 5000;
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

            if (fabs(v - C[i*N + j]) > 1e-3) {
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