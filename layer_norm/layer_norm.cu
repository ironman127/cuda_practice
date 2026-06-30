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

struct WelfordState {
    int n;
    float mean;
    float m2;
};

__device__ WelfordState welford_combine(const WelfordState& s1, const WelfordState& s2) {
    WelfordState s;
    s.n = s1.n + s2.n;
    s.mean = s1.mean + (s2.mean - s1.mean) * s2.n / (s.n);
    s.m2 = s1.m2 + s2.m2 + (s1.mean - s2.mean) * (s1.mean - s2.mean) * s1.n * s2.n / (s.n);
    return s;
}

__device__ void welford_update(WelfordState& s1, float x) {
    s1.n += 1;
    float delta = x - s1.mean;
    s1.mean += delta / (s1.n);
    s1.m2 = s1.m2 + delta * (x - s1.mean);
}

template <int BLOCK_SIZE = 1024, int VEC_SIZE = 4>
__global__ void layer_norm(
    const float* __restrict__ x,  // (B, H)
    const float* __restrict__ gamma, // (H)
    const float* __restrict__ beta, // (H)
    float* __restrict__ y, // (B, H)
    int H,
    float epsilon
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* x_row = x + row * H;
    float* y_row = y + row * H;

    WelfordState wf_state = {0, 0.0f, 0.0f};
    for (int i = tid * VEC_SIZE; i < H; i += BLOCK_SIZE * VEC_SIZE) {
        float4 v;
        v = *(reinterpret_cast<const float4*>(x_row + i));
    
        welford_update(wf_state, v.x);
        welford_update(wf_state, v.y);
        welford_update(wf_state, v.z);
        welford_update(wf_state, v.w);
    }

    for (int i = WARP_SIZE / 2; i > 0; i >>= 1) {
        WelfordState o;
        o.n = __shfl_down_sync(0xFFFFFFFF, wf_state.n, i);
        o.mean = __shfl_down_sync(0xFFFFFFFF, wf_state.mean, i);
        o.m2 = __shfl_down_sync(0xFFFFFFFF, wf_state.m2, i);
        wf_state = welford_combine(wf_state, o);
    }

    constexpr int S_SIZE = BLOCK_SIZE / WARP_SIZE;
    __shared__ WelfordState s_wf_state[S_SIZE];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
        s_wf_state[warp_id] = wf_state;
    }

    __syncthreads();

    for (int s = S_SIZE / 2; s > 0; s >>= 1) {
        if (tid + s < S_SIZE) {
            s_wf_state[tid] = welford_combine(s_wf_state[tid], s_wf_state[tid + s]);
        }

        __syncthreads();
    }

    wf_state = s_wf_state[0];
    float final_mean = wf_state.mean;
    float final_rstd = rsqrtf(wf_state.m2 / wf_state.n + epsilon);

    for (int i = tid * VEC_SIZE; i < H; i += BLOCK_SIZE * VEC_SIZE) {
        float4 vx = *(reinterpret_cast<const float4*>(x_row + i));
        float4 vg = *(reinterpret_cast<const float4*>(gamma + i));
        float4 vb = *(reinterpret_cast<const float4*>(beta + i));
        float4 vy;
        vy.x = (vx.x - final_mean) * final_rstd * vg.x + vb.x;
        vy.y = (vx.y - final_mean) * final_rstd * vg.y + vb.y;
        vy.z = (vx.z - final_mean) * final_rstd * vg.z + vb.z;
        vy.w = (vx.w - final_mean) * final_rstd * vg.w + vb.w;
        *(reinterpret_cast<float4*>(y_row + i)) = vy;
    }

}

int main() {
    int b = 100, h = 1000;
    float *x = new float[b * h];
    float *y = new float[b * h];
    float *gamma = new float[h];
    float *beta = new float[h];
    float epsilon = 1e-3;

    srand(42);
    auto randf = [](float lo, float hi) {
        return lo + (hi - lo) * (rand() / (float)RAND_MAX);
    };

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < h; ++j) {
            x[i * h + j] = randf(-1.0f, 1.0f);
        }
    }
    for (int i = 0; i < h; ++i) {
        gamma[i] = randf(0.5f, 1.5f);
        beta[i]  = randf(-0.5f, 0.5f);
    }

    float *d_x, *d_y, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_x, b * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, b * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, h * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, b * h * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma, h * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta, h * sizeof(float), cudaMemcpyHostToDevice));

    layer_norm<<<b, 1024>>>(d_x, d_gamma, d_beta, d_y, h, epsilon);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y, d_y, b * h * sizeof(float), cudaMemcpyDeviceToHost));

    float mean[b], rstd[b];
    for (int i = 0; i < b; ++i) {
        mean[i] = 0.0f;
        for (int j = 0; j < h; ++j) {
            mean[i] += x[i * h + j];
        }
        mean[i] /= h;
    }

    for (int i = 0; i < b; ++i) {
        rstd[i] = 0.0f;
        for (int j = 0; j < h; ++j) {
            rstd[i] += (x[i * h + j] - mean[i]) * (x[i * h + j] - mean[i]);
        }
        rstd[i] = rsqrtf(rstd[i] / h + epsilon);
    }

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < h; ++j) {
            float y_valid = (x[i * h + j] - mean[i]) / rstd[i] * gamma[j] + beta[j];
            if (fabs(y_valid - y[i * h + j]) > 1e-3) {
                printf("Error: %f != %f\n", y_valid, y[i * h + j]);
            }
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    delete[] x;
    delete[] y;
    delete[] gamma;
    delete[] beta;
    return 0;
}