#include<malloc.h>
#include<math.h>

__global__ void matrix_multiple_element(int *A, int *B, int *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= width || col >= width) {
        return;
    }


    C[row*width + col] = 0;
    for (int i=0; i < width; ++i) {
        C[row*width + col] += A[row*width + i] * B[i*width + col];
    }

}

__global__ void matrix_multiple_row(int *A, int *B, int *C, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= width) {
        return;
    }

    for (int i=0; i < width; ++i) {
        C[row*width + i] = 0;
        for (int j=0; j < width; ++j) {
            C[row*width + i] += A[row*width + j] * B[j*width + i];
        }
    }

}

__global__ void matrix_multiple_col(int *A, int *B, int *C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= width) {
        return;
    }

    for (int i=0; i < width; ++i) {
        C[i*width + col] = 0;
        for (int j=0; j < width; ++j) {
            C[i*width + col] += A[i*width + j] * B[j*width + col];
        }
    }

}

void matirx_multiple(int *A_h, int *B_h, int *C_h, int width) {
    int *A_d, *B_d, *C_d;
    int ele_num = width * width;

    cudaMalloc((void**)&A_d, ele_num * sizeof(int));
    cudaMalloc((void**)&B_d, ele_num * sizeof(int));
    cudaMalloc((void**)&C_d, ele_num * sizeof(int));

    cudaMemcpy(A_d, A_h, ele_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, ele_num * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim = {16, 16, 1};
    dim3 grid_dim = {ceil((block_dim.x-1 + width)/block_dim.x), ceil((block_dim.x-1 + width)/block_dim.x), 1};
    //matrix_multiple_element<<<grid_dim, block_dim>>>(A_d, B_d, C_d, width);

    dim3 block_dim = {16, 1, 1};
    dim3 grid_dim = {ceil((block_dim.x-1 + width)/block_dim.x), 1, 1};
    //matrix_multiple_row<<<grid_dim, block_dim>>>(A_d, B_d, C_d, width);

    dim3 block_dim = {16, 1, 1};
    dim3 grid_dim = {ceil((block_dim.x-1 + width)/block_dim.x), 1, 1};
    matrix_multiple_col<<<grid_dim, block_dim>>>(A_d, B_d, C_d, width);

    


    cudaMemcpy(C_h, C_d, ele_num * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int width = 5;
    int (*A)[width], (*B)[width], (*C)[width];

    A = (int(*)[width]) malloc(sizeof(int[width]) * width);
    B = (int(*)[width]) malloc(sizeof(int[width]) * width);
    C = (int(*)[width]) malloc(sizeof(int[width]) * width);
    

    for (int i=0; i < width; ++i) {
        for (int j=0; j < width; ++j) {
            A[i][j] = i;
            B[i][j] = j;
        }
    }

    matirx_multiple((int*)A, (int*)B, (int*) C, width);

    for (int i=0; i < width; ++i) {
        for (int j=0; j < width; ++j) {
            printf("[%d]\t", C[i][j]);
        }
        printf("\n");
    }


    free(A);
    free(B);
    free(C);
}