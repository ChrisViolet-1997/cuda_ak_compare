#include <cuda_runtime.h>
#include <stdio.h>

// 实现2: 向量化实现 - 每个线程处理多个元素
__global__ void vectorAddVectorized(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 每个线程处理多个元素，提高内存访问效率
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

extern "C" {
    void launchVectorAddVectorized(const float* d_a, const float* d_b, float* d_c, int n, int blockSize) {
        int gridSize = min((n + blockSize - 1) / blockSize, 1024);
        vectorAddVectorized<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    }
}
