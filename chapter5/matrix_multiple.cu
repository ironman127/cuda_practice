#include<stdio.h>

// 运行时错误检查 — 每次调用 CUDA API 后都加！
#define CUDA_CHECK(call)                                         \
do {                                                             \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(1);                                                 \
    }                                                            \
} while(0)

__global__ void matrix_multiple_kernel(float *A, float *B, float *C, size_t I, size_t K, size_t J, size_t sh_size) {

    extern __shared__ char A_B[];

    float *As = (float*)A_B;
    float *Bs = (float*)(A_B + sh_size / 2);
    size_t TILE_SIZE = blockDim.x;

    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    
    float cs = 0;
    for (int i = 0; i < (K + TILE_SIZE - 1) / TILE_SIZE; ++i) {
        size_t cur_row = (i*TILE_SIZE + threadIdx.y);
        size_t cur_col = (i*TILE_SIZE + threadIdx.x);

        if (cur_col < K && row < I) {
            As[TILE_SIZE * threadIdx.y + threadIdx.x] = A[row*K + cur_col];
        } else {
            As[TILE_SIZE * threadIdx.y + threadIdx.x] = 0;
        }

        if (cur_row < K && col < J) {
            Bs[TILE_SIZE * threadIdx.y + threadIdx.x] = B[cur_row*J + col];
        } else {
            Bs[TILE_SIZE * threadIdx.y + threadIdx.x] = 0;
        }

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j) {
            cs += As[threadIdx.y * TILE_SIZE + j] * Bs[j*TILE_SIZE + threadIdx.x];
        }

        __syncthreads();
    }

    if ((row < I) && (col < J)) {
        C[row * J + col] = cs;
    }
}


size_t calculate_share_mem_size(size_t thread_per_block) {
    int dev_count;
    cudaGetDeviceCount(&dev_count);

    if (dev_count <= 0) {
        return -1;
    }

    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);

    size_t sh = dev_prop.sharedMemPerBlock;

    if (8 * thread_per_block <= sh ) {
        return thread_per_block * 2 * sizeof(float);
    }

    return -1;
}

void matirx_multiple(float *A_h, float *B_h, float *C_h, size_t I, size_t K, size_t J) {
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**)&A_d, I * K * sizeof(float));
    cudaMalloc((void**)&B_d, K * J * sizeof(float));
    cudaMalloc((void**)&C_d, I * J * sizeof(float));

    cudaMemcpy(A_d, A_h, I * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K * J * sizeof(float), cudaMemcpyHostToDevice);

    dim3 b_dim = {3, 3, 1};
    dim3 g_dim = {(J + b_dim.x - 1) / b_dim.x, (I + b_dim.y - 1) / b_dim.y, 1};
    size_t sh_size = calculate_share_mem_size(b_dim.x * b_dim.y);

    matrix_multiple_kernel<<<g_dim, b_dim, sh_size>>>(A_d, B_d, C_d, I, K, J, sh_size);

    CUDA_CHECK(cudaGetLastError());           // 检查 kernel launch 是否成功
    CUDA_CHECK(cudaDeviceSynchronize());       // 检查 kernel 执行是否出错

    cudaMemcpy(C_h, C_d, I * J * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void show_matrix(float* M, size_t m, size_t n) {
    for (size_t i=0; i < m; ++i) {
        for (size_t j=0; j < n; ++j) {
            printf("[%0.2f]\t", M[i*n + j]);
        }
        printf("\n");
    }
}


void test_result(float *A, float *B, float *C, size_t I, size_t K, size_t J) {
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {

            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i*K+k] * B[k*J + j];
            }

            if (abs(sum - C[i*J + j]) > 0.1) {
                printf("C[%d][%d]=%0.f, %0.f, fail.\n", i, j, sum, C[i*J + j]);
                return;
            }
        }
    }

    printf("success.\n");
}


int main() {
    size_t I = 1000, K = 1100, J = 1600;
    float *A, *B, *C;

    A = (float*) malloc(sizeof(float) * I * K);
    B = (float*) malloc(sizeof(float) * K * J);
    C = (float*) malloc(sizeof(float) * I * J);
    

    for (size_t i=0; i < I; ++i) {
        for (size_t j=0; j < K; ++j) {
            A[i*K + j] = i;
        }
    }

    for (size_t i=0; i < K; ++i) {
        for (size_t j=0; j < J; ++j) {
            B[i*J + j] = j;
        }
    }

    //show_matrix(A, I, K);
    //show_matrix(B, K, J);

    matirx_multiple(A, B, C, I, K, J);

    test_result(A, B, C, I, K, J);

    //show_matrix(C, I, J);

    free(A);
    free(B);
    free(C);
}