#include<stdio.h>
#include<malloc.h>
#include<cuda_runtime.h>
#include<math.h>

__global__ void vec_add_kernel(float *a, float *b, float *c, int vec_len) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < vec_len) {
        c[i] = a[i] + b[i];
    }
}

void vec_add(float *a_h, float *b_h, float *c_h, int len) {
    float *a_d, *b_d, *c_d;

    cudaMalloc((void**)&a_d, len * sizeof(float));
    cudaMalloc((void**)&b_d, len * sizeof(float));
    cudaMalloc((void**)&c_d, len * sizeof(float));

    cudaMemcpy(a_d, a_h, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, len * sizeof(float), cudaMemcpyHostToDevice);

    vec_add_kernel<<<ceil(len/256.0), 256>>>(a_d, b_d, c_d, len);

    cudaMemcpy(c_h, c_d, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main() {
    int vec_len = 1000;
    float *a_h, *b_h, *c_h;

    a_h = (float*) malloc(vec_len * sizeof(float));
    b_h = (float*) malloc(vec_len * sizeof(float));
    c_h = (float*) malloc(vec_len * sizeof(float));

    for (int a_i=0, b_i=1; a_i<vec_len; ++a_i) {
        a_h[a_i] = a_i;
        b_h[a_i] = b_i;
    }

    vec_add(a_h, b_h, c_h, vec_len);

    for (int i=0; i<vec_len; ++i) {
        printf("c_h[%d]=%0.f,\n", i, c_h[i]);
    }

    return 0;
}