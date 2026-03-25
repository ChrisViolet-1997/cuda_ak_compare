#include <cuda_runtime.h>
#include <float.h>

#define BLOCK_SIZE 256

__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    // 循环直到成功更新
    do {
        assumed = old;
        // 如果新值不大于当前值，直接退出，无需更新
        if (val <= __int_as_float(assumed)) {
            break;
        }
        // 尝试原子替换：如果地址处的值仍为 assumed，则更新为 val 的整数表示
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}


// 简单的全局reduce找最大值
__global__ void simple_max_kernel(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程加载一个元素
    sdata[tid] = (idx < n) ? x[idx] : -INFINITY;
    __syncthreads();

    // 在共享内存中进行reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 第一个线程写入结果
    if (tid == 0) {
        atomicMaxFloat(out, sdata[0]);
    }
}

// 简单的全局reduce计算exp和
__global__ void simple_sum_kernel(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   const float* __restrict__ max_ptr,
                                   int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程计算exp(x - max)
    float max_val = *max_ptr;
    sdata[tid] = (idx < n) ? expf(x[idx] - max_val) : 0.0f;
    __syncthreads();

    // 在共享内存中进行reduce求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 第一个线程写入结果
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// 简单的softmax计算kernel
__global__ void simple_softmax_kernel(const float* __restrict__ x,
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

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* d_max;
    float* d_sum;

    cudaMallocAsync(&d_max, sizeof(float), cudaStreamDefault);
    cudaMallocAsync(&d_sum, sizeof(float), cudaStreamDefault);

    // 初始化为负无穷和0
    cudaMemsetAsync(d_max, 0xFF, sizeof(float), cudaStreamDefault);
    cudaMemsetAsync(d_sum, 0, sizeof(float), cudaStreamDefault);

    // 第一步：找最大值
    simple_max_kernel<<<grid, BLOCK_SIZE>>>(input, d_max, N);

    // 第二步：计算exp(x-max)的和
    simple_sum_kernel<<<grid, BLOCK_SIZE>>>(input, d_sum, d_max, N);

    // 第三步：计算最终的softmax值
    simple_softmax_kernel<<<grid, BLOCK_SIZE>>>(input, output, d_max, d_sum, N);

    cudaFreeAsync(d_max, cudaStreamDefault);
    cudaFreeAsync(d_sum, cudaStreamDefault);
}
