#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    const int block_size = BLOCK_SIZE;
    //定义block行为，尽量不访问global memory
    __shared__ float cube[BLOCK_SIZE][BLOCK_SIZE + 1];
    //当前访问元素的坐标
    const int idx = blockIdx.x * block_size + threadIdx.x;
    const int idy = blockIdx.y * block_size + threadIdx.y;

    const int in_cols = cols;
    const int in_rows = rows;
    //二维转一维，列数 * 纵坐标 + 横坐标
    const int input_idx =  idx + in_cols * idy;

    if(input_idx < in_cols * in_rows){
        //存储的时候直接转置
        cube[threadIdx.y][threadIdx.x] = input[input_idx];
    }
    
    __syncthreads();
    //当前元素的坐标为（idx, idy）,输出元素的坐标理论上是（idy, idx）.
    // (0,1) -> (1, 0) , (0, 2) -> (2, 0) 这样对于输出数组，就是不连续的了
    //所以将读取的数据和写入的数据解耦，没必要一定要将当前进程读取的数字存到结果里面，仍然访问结果数据的当前坐标就可以
    //block 维度已经发生了翻转
    const int out_x = blockIdx.y * block_size + threadIdx.x;
    const int out_y = blockIdx.x * block_size + threadIdx.y;
    const int out_cols = rows;
    const int out_rows = cols;
    if(out_x < out_cols && out_y < out_rows){
        output[out_x + out_y * out_cols] = cube[threadIdx.x][threadIdx.y];
    }
}
