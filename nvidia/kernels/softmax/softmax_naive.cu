#include <cuda_runtime.h>
#include <float.h>

#define BLOCK_SIZE 256

// 最原始版本：每个线程直接使用atomic操作
// 这是性能最差的版本，但实现最简单

// 每个线程直接用atomicMax找最大值
__global__ void naive_max_kernel(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMax((int*)out, __float_as_int(x[idx]));
    }
}

// 每个线程直接用atomicAdd计算exp和
__global__ void naive_sum_kernel(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  const float* __restrict__ max_ptr,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float max_val = *max_ptr;
        atomicAdd(out, expf(x[idx] - max_val));
    }
}

// 计算最终的softmax值
__global__ void naive_softmax_kernel(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      const float* __restrict__ max_ptr,
                                      const float* __restrict__ sum_ptr,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float max_val = *max_ptr;
        float sum_val = *sum_ptr;
        y[idx] = expf(x[idx] - max_val) / sum_val;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* d_max;
    float* d_sum;

    cudaMallocAsync(&d_max, sizeof(float), cudaStreamDefault);
    cudaMallocAsync(&d_sum, sizeof(float), cudaStreamDefault);

    // 初始化为负无穷和0
    cudaMemsetAsync(d_max, 0xFF, sizeof(float), cudaStreamDefault);
    cudaMemsetAsync(d_sum, 0, sizeof(float), cudaStreamDefault);

    // 第一步：每个线程直接用atomicMax找最大值
    naive_max_kernel<<<grid, BLOCK_SIZE>>>(input, d_max, N);

    // 第二步：每个线程直接用atomicAdd计算exp(x-max)的和
    naive_sum_kernel<<<grid, BLOCK_SIZE>>>(input, d_sum, d_max, N);

    // 第三步：计算最终的softmax值
    naive_softmax_kernel<<<grid, BLOCK_SIZE>>>(input, output, d_max, d_sum, N);

    cudaFreeAsync(d_max, cudaStreamDefault);
    cudaFreeAsync(d_sum, cudaStreamDefault);
}
