#include <cuda_runtime.h>
#include <stdio.h>

// 实现1: 基础实现 - 每个线程处理一个元素
__global__ void vectorAddBasic(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" {
    void launchVectorAddBasic(const float* d_a, const float* d_b, float* d_c, int n, int blockSize) {
        int gridSize = (n + blockSize - 1) / blockSize;
        vectorAddBasic<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    }
}
