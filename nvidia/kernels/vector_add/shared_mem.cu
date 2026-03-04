#include <cuda_runtime.h>
#include <stdio.h>

// 实现3: 共享内存优化 - 使用shared memory减少全局内存访问
__global__ void vectorAddSharedMem(const float* a, const float* b, float* c, int n) {
    extern __shared__ float shared[];
    float* s_a = shared;
    float* s_b = &shared[blockDim.x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        // 加载数据到共享内存
        s_a[tid] = a[idx];
        s_b[tid] = b[idx];
        __syncthreads();

        // 从共享内存读取并计算
        c[idx] = s_a[tid] + s_b[tid];
    }
}

extern "C" {
    void launchVectorAddSharedMem(const float* d_a, const float* d_b, float* d_c, int n, int blockSize) {
        int gridSize = (n + blockSize - 1) / blockSize;
        int sharedMemSize = 2 * blockSize * sizeof(float);
        vectorAddSharedMem<<<gridSize, blockSize, sharedMemSize>>>(d_a, d_b, d_c, n);
    }
}
