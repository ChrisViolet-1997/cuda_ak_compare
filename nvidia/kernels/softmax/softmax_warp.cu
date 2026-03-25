#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <limits.h>

#define BLOCK_SIZE 256
#define TM 8
#define NUM_BLOCK (BLOCK_SIZE * TM)
#define FULL_MASK 0xffffffff
#define UPPER_DIV(A, B) (((A) + (B) - 1) / (B))

__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        if (val <= __int_as_float(assumed))
            break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

__device__ float warp_reduce_max(float v) {
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, 16));
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, 8));
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, 4));
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, 2));
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, 1));
    return v;
}
__device__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    return v;
}

__global__ void reduce_max_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int n){
        int block_start = blockIdx.x * blockDim.x * TM;
        int wid = threadIdx.x / 32;
        int lane = threadIdx.x % 32;
        float local_max = -INFINITY;
        #pragma unroll
        for (int i = 0; i < TM; i++){
            int tid = block_start + i * BLOCK_SIZE + threadIdx.x;
            if(tid < n) local_max = max(local_max, x[tid]);
        }
        __syncthreads();
        //上面得到了一个线程当中的局部最大值
        float warp_max = warp_reduce_max(local_max);
        __shared__ float smem[BLOCK_SIZE / 32];
        if(lane == 0)smem[wid] = warp_max;   
        //一个warp计算一个局部最大值，存在smem数组里，元素个数等于warp个数
        __syncthreads();
        if(wid == 0){ //在第一个warp中汇总得到当前block的最大值
            float block_max = lane < BLOCK_SIZE / 32 ? smem[lane] : -INFINITY;
            //这一步就相当于将warp上存储的最大值，数量就是warp有多少个就复制多少个过来
            block_max = warp_reduce_max(block_max);
            if(lane == 0) atomicMaxFloat(out, block_max);
        }
    } 


__global__ void reduce_exp_sum_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    const float* __restrict__ max_ptr,
                                    int n){
        int block_start = blockIdx.x * blockDim.x * TM;
        int wid = threadIdx.x / 32;
        int lane = threadIdx.x % 32;
        float local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < TM; i++){
            int tid = block_start + i * BLOCK_SIZE + threadIdx.x;
            if(tid < n) local_sum += exp(x[tid] - max_ptr[0]);
        }
        __syncthreads();
        //上面得到了一个线程当中的局部和
        float warp_sum = warp_reduce_sum(local_sum);
        __shared__ float smem[BLOCK_SIZE / 32];
        if(lane == 0)smem[wid] = warp_sum;   
        //一个warp计算一个局部最大值，存在smem数组里，元素个数等于warp个数
        __syncthreads();
        if(wid == 0){ //在第一个warp中汇总得到当前block的最大值
            float block_sum = lane < BLOCK_SIZE / 32 ? smem[lane] : 0;
            //这一步就相当于将warp上存储的最大值，数量就是warp有多少个就复制多少个过来
            block_sum = warp_reduce_sum(block_sum);
            if(lane == 0) atomicAdd(out, block_sum);
        }
    } 

__global__ void softmax_tail_kernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    const float* __restrict__ max_ptr,
                                    const float* __restrict__ sum_ptr,
                                    int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float m   = *max_ptr;
        float sum = *sum_ptr;
        y[idx] = __expf(x[idx] - m) / sum;
    }
}
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
  int dev = 0, sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);

  int grid = UPPER_DIV(N, BLOCK_SIZE * TM); //block 数量，kernel针对的是block进行编程

  float* d_max;
  float* d_sum;

  cudaMallocAsync(&d_max, sizeof(float), cudaStreamDefault);
  cudaMallocAsync(&d_sum, sizeof(float), cudaStreamDefault);

  // 初始化为负无穷和0
  cudaMemsetAsync(d_max, 0xFF, sizeof(float), cudaStreamDefault);
  cudaMemsetAsync(d_sum, 0, sizeof(float), cudaStreamDefault);

 //求最大值的kernel，输入就是一个数组，一个N，结果存储在d_max中就行了
  reduce_max_kernel<<<grid, BLOCK_SIZE>>>(input, d_max, N);
  //得到最大值了，带入到求次方和的kernel中，结果存储在d_sum中
  reduce_exp_sum_kernel<<<grid, BLOCK_SIZE>>>(input,d_sum, d_max, N);
  //直接映射得到结果数组，不需要做warp或者线程优化
  softmax_tail_kernel<<<UPPER_DIV(N, BLOCK_SIZE), BLOCK_SIZE>>>(input, output, d_max, d_sum, N);

  cudaFreeAsync(d_max, cudaStreamDefault);
  cudaFreeAsync(d_sum, cudaStreamDefault);
}

