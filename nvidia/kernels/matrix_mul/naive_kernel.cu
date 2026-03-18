#include <cuda_runtime.h>

/**
 * @brief Naive matrix multiplication kernel
 * Each thread computes one element of the output matrix C
 * C[i,j] = sum(A[i,k] * B[k,j]) for k in [0, K)
 */
__global__ void matrix_mul_naive_kernel(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matrix_mul_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
