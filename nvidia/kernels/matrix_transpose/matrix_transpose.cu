#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //输入x，最大不超过cols
    int idy = blockIdx.y * blockDim.y + threadIdx.y; //输出y， 最大不超过rows
    if(idy < rows && idx < cols){
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}
