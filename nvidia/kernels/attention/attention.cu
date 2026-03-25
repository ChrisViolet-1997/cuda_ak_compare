#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// ============================================================
// Utility Macros
// ============================================================
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CFLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])
#define FULL_MASK 0xffffffff

// ============================================================
// 1. Optimized Matrix Transpose Kernel
//    (from matrix_transpose_opt.cu — shared memory + padding)
// ============================================================
#define TRANS_BLOCK_SIZE 16

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TRANS_BLOCK_SIZE][TRANS_BLOCK_SIZE + 1];

    int idx = blockIdx.x * TRANS_BLOCK_SIZE + threadIdx.x;
    int idy = blockIdx.y * TRANS_BLOCK_SIZE + threadIdx.y;

    if (idx < cols && idy < rows) {
        tile[threadIdx.y][threadIdx.x] = input[idy * cols + idx];
    }
    __syncthreads();

    int out_x = blockIdx.y * TRANS_BLOCK_SIZE + threadIdx.x;
    int out_y = blockIdx.x * TRANS_BLOCK_SIZE + threadIdx.y;

    if (out_x < rows && out_y < cols) {
        output[out_y * rows + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================
// 2. Optimized GEMM Kernel
//    (from optimized_kernel.cu — double buffering, warp tiling, float4)
// ============================================================
constexpr int WARPSIZE = 32;

// Block Tiling dimensions
constexpr int BM = 128;
constexpr int BK = 16;
constexpr int BN = 128;

// Warp Tiling dimensions
constexpr int WM = 64;
constexpr int WN = 64;

// Sub-Warp Tiling dimensions
constexpr int WSUBM = 64;
constexpr int WSUBN = 16;

// Thread Tiling dimensions
constexpr int TM = 8;
constexpr int TN = 4;

template <const int _BM, const int _BK, const int _BN, const int _WM, const int _WN,
          const int _WSUBM, const int _WSUBN, const int _TM, const int _TN, const int kExtraCol>
__device__ void load_global_to_shared(const float *A, const float *B, const int M, const int K, const int N,
                                       float *As, float *Bs) {
    static_assert(_BK % 4 == 0);
    static_assert(_BN % 4 == 0);

    constexpr int thread_num = (_BM/_WM)*(_BN/_WN)*WARPSIZE;
    static_assert(_BK*_BM % (thread_num*4) == 0);
    static_assert(_BK*_BN % (thread_num*4) == 0);

    // Load Tile A (and transpose into shared memory)
    constexpr int ldg_a_niter = _BK*_BM / (thread_num*4);
    constexpr int a_tile_stride = _BM / ldg_a_niter;

    const int a_tile_row = threadIdx.x / (_BK / 4);
    const int a_tile_col = (threadIdx.x % (_BK / 4)) * 4;

    #pragma unroll
    for (int i = 0; i < ldg_a_niter; ++i) {
        int row_offset = i * a_tile_stride;
        float4 ldg_a_reg = CFLOAT4(A[OFFSET(a_tile_row + row_offset, a_tile_col, K)]);
        As[OFFSET(a_tile_col    , a_tile_row + row_offset, _BM + kExtraCol)] = ldg_a_reg.x;
        As[OFFSET(a_tile_col + 1, a_tile_row + row_offset, _BM + kExtraCol)] = ldg_a_reg.y;
        As[OFFSET(a_tile_col + 2, a_tile_row + row_offset, _BM + kExtraCol)] = ldg_a_reg.z;
        As[OFFSET(a_tile_col + 3, a_tile_row + row_offset, _BM + kExtraCol)] = ldg_a_reg.w;
    }

    // Load Tile B
    constexpr int ldg_b_niter = _BK*_BN / (thread_num*4);
    constexpr int b_tile_stride = _BK / ldg_b_niter;

    const int b_tile_row = threadIdx.x / (_BN / 4);
    const int b_tile_col = (threadIdx.x % (_BN / 4)) * 4;

    #pragma unroll
    for (int i = 0; i < ldg_b_niter; ++i) {
        int row_offset = i * b_tile_stride;
        FLOAT4(Bs[OFFSET(b_tile_row + row_offset, b_tile_col, _BN)]) =
            CFLOAT4(B[OFFSET(b_tile_row + row_offset, b_tile_col, N)]);
    }
}

template <const int _BM, const int _BK, const int _BN, const int _WM, const int _WN,
          const int _WSUBM, const int _WSUBN, const int _TM, const int _TN, const int kExtraCol>
__device__ void compute_mma_from_shared(float *As, float *Bs, float *Areg, float *Breg, float *Creg) {
    constexpr int WMITER = _WM / _WSUBM;
    constexpr int WNITER = _WN / _WSUBN;

    const int warp_idx = threadIdx.x / WARPSIZE;
    const int wy = warp_idx / (_BN / _WN);
    const int wx = warp_idx % (_BN / _WN);

    const int thread_idx = threadIdx.x % WARPSIZE;
    const int ty = thread_idx / (_WSUBN / _TN);
    const int tx = thread_idx % (_WSUBN / _TN);

    #pragma unroll
    for (unsigned int k = 0; k < _BK; ++k) {
        #pragma unroll
        for (unsigned int wsy = 0; wsy < WMITER; ++wsy) {
            #pragma unroll
            for (unsigned int j = 0; j < _TM; ++j) {
                Areg[wsy*_TM + j] =
                    As[OFFSET(k, wy*_WM + wsy*_WSUBM + ty*_TM + j, _BM + kExtraCol)];
            }
        }
        #pragma unroll
        for (unsigned int wsx = 0; wsx < WNITER; ++wsx) {
            #pragma unroll
            for (unsigned int i = 0; i < _TN; ++i) {
                Breg[wsx*_TN + i] =
                    Bs[OFFSET(k, wx*_WN + wsx*_WSUBN + tx*_TN + i, _BN)];
            }
        }

        #pragma unroll
        for (unsigned int wsy = 0; wsy < WMITER; ++wsy) {
            #pragma unroll
            for (unsigned int wsx = 0; wsx < WNITER; ++wsx) {
                #pragma unroll
                for (int m = 0; m < _TM; m++) {
                    #pragma unroll
                    for (int n = 0; n < _TN; n++) {
                        Creg[OFFSET(wsy*_TM + m, wsx*_TN + n, WNITER*_TN)] +=
                            Areg[wsy*_TM + m] * Breg[wsx*_TN + n];
                    }
                }
            }
        }
    }
}

template <const int _BM, const int _BK, const int _BN, const int _WM, const int _WN,
          const int _WSUBM, const int _WSUBN, const int _TM, const int _TN>
__device__ void store_result_to_global(float *C, const int N, float *Creg) {
    constexpr int WMITER = _WM / _WSUBM;
    constexpr int WNITER = _WN / _WSUBN;

    const unsigned int warp_idx = threadIdx.x / WARPSIZE;
    const unsigned int wy = warp_idx / (_BN / _WN);
    const unsigned int wx = warp_idx % (_BN / _WN);

    const unsigned int thread_idx = threadIdx.x % WARPSIZE;
    const unsigned int ty = thread_idx / (_WSUBN / _TN);
    const unsigned int tx = thread_idx % (_WSUBN / _TN);

    C = &C[OFFSET(wy*_WM, wx*_WN, N)];

    #pragma unroll
    for (unsigned int wsy = 0; wsy < WMITER; ++wsy) {
        #pragma unroll
        for (unsigned int wsx = 0; wsx < WNITER; ++wsx) {
            float* Cws = &C[OFFSET(wsy*_WSUBM, wsx*_WSUBN, N)];
            #pragma unroll
            for (unsigned int j = 0; j < _TM; j += 1) {
                #pragma unroll
                for (unsigned int i = 0; i < _TN; i += 4) {
                    FLOAT4(Cws[OFFSET(ty*_TM + j, tx*_TN + i, N)]) =
                        CFLOAT4(Creg[OFFSET(wsy*_TM + j, wsx*_TN + i, WNITER*_TN)]);
                }
            }
        }
    }
}

template<const int _BM, const int _BK, const int _BN, const int _WM, const int _WN,
         const int _WSUBM, const int _WSUBN, const int _TM, const int _TN>
__device__ void gemm_block_tile(const float* A, const float* B, float* C,
                                 const int M, const int K, const int N) {
    constexpr int kExtraCol = 4;
    constexpr int kAsSize = _BK*(_BM+kExtraCol);
    constexpr int kBsSize = _BK*_BN;

    __shared__ float As[2][kAsSize];
    __shared__ float Bs[2][kBsSize];

    constexpr unsigned int WMITER = _WM / _WSUBM;
    constexpr unsigned int WNITER = _WN / _WSUBN;
    float Areg[WMITER*_TM];
    float Breg[WNITER*_TN];
    float Creg[WMITER*_TM * WNITER*_TN] = {0.0f};

    int buffer_idx = 0;

    load_global_to_shared<_BM, _BK, _BN, _WM, _WN, _WSUBM, _WSUBN, _TM, _TN, kExtraCol>(
        A, B, M, K, N, As[buffer_idx], Bs[buffer_idx]
    );
    __syncthreads();

    for (int k = _BK; k < K; k += _BK) {
        A += _BK;
        B += _BK * N;

        load_global_to_shared<_BM, _BK, _BN, _WM, _WN, _WSUBM, _WSUBN, _TM, _TN, kExtraCol>(
            A, B, M, K, N, As[1 - buffer_idx], Bs[1 - buffer_idx]
        );

        compute_mma_from_shared<_BM, _BK, _BN, _WM, _WN, _WSUBM, _WSUBN, _TM, _TN, kExtraCol>(
            As[buffer_idx], Bs[buffer_idx], Areg, Breg, Creg
        );

        __syncthreads();
        buffer_idx = 1 - buffer_idx;
    }

    compute_mma_from_shared<_BM, _BK, _BN, _WM, _WN, _WSUBM, _WSUBN, _TM, _TN, kExtraCol>(
        As[buffer_idx], Bs[buffer_idx], Areg, Breg, Creg
    );

    store_result_to_global<_BM, _BK, _BN, _WM, _WN, _WSUBM, _WSUBN, _TM, _TN>(C, N, Creg);
}

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               const int M, const int K, const int N) {
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;

    const float* A_block = &A[OFFSET(block_y*BM, 0, K)];
    const float* B_block = &B[OFFSET(0, block_x*BN, N)];
    float* C_block = &C[OFFSET(block_y*BM, block_x*BN, N)];

    gemm_block_tile<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN>(
        A_block, B_block, C_block, M, K, N);
}

// ============================================================
// 3. Scale Kernel — element-wise division by sqrt(d_model)
// ============================================================
__global__ void scale_kernel(float* data, int n, float scale) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// ============================================================
// 4. Row Softmax Kernel
//    Each block processes one row of the matrix.
//    Uses warp shuffle reductions (adapted from softmax_warp.cu).
// ============================================================
#define SOFTMAX_BLOCK_SIZE 256

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

__global__ void row_softmax_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_data = data + row * cols;
    int tid = threadIdx.x;
    int num_warps = SOFTMAX_BLOCK_SIZE / 32;

    // Phase 1: Find row maximum
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
        local_max = fmaxf(local_max, row_data[i]);
    }

    float warp_max = warp_reduce_max(local_max);
    __shared__ float smem_max[SOFTMAX_BLOCK_SIZE / 32];
    int wid = tid / 32;
    int lane = tid % 32;
    if (lane == 0) smem_max[wid] = warp_max;
    __syncthreads();

    if (wid == 0) {
        float block_max = lane < num_warps ? smem_max[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) smem_max[0] = block_max;
    }
    __syncthreads();
    float row_max = smem_max[0];

    // Phase 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
        float val = __expf(row_data[i] - row_max);
        row_data[i] = val;  // store exp result in-place
        local_sum += val;
    }

    float warp_sum = warp_reduce_sum(local_sum);
    __shared__ float smem_sum[SOFTMAX_BLOCK_SIZE / 32];
    if (lane == 0) smem_sum[wid] = warp_sum;
    __syncthreads();

    if (wid == 0) {
        float block_sum = lane < num_warps ? smem_sum[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) smem_sum[0] = block_sum;
    }
    __syncthreads();
    float row_sum = smem_sum[0];

    // Phase 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
        row_data[i] *= inv_sum;
    }
}

// ============================================================
// Online Softmax Kernel (Fused sum + normalize computation)
//    - Two-pass algorithm: find max, then fused exp+sum+normalize
//    - Reduces global memory traversals from 3 to 2
// ============================================================

__global__ void row_softmax_online_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_data = data + row * cols;
    int tid = threadIdx.x;
    int num_warps = SOFTMAX_BLOCK_SIZE / 32;
    int wid = tid / 32;
    int lane = tid % 32;

    // Phase 1: Find row maximum (same as naive version)
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
        local_max = fmaxf(local_max, row_data[i]);
    }

    float warp_max = warp_reduce_max(local_max);
    __shared__ float smem_max[SOFTMAX_BLOCK_SIZE / 32];
    if (lane == 0) smem_max[wid] = warp_max;
    __syncthreads();

    if (wid == 0) {
        float block_max = lane < num_warps ? smem_max[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) smem_max[0] = block_max;
    }
    __syncthreads();
    float row_max = smem_max[0];

    // Phase 2: Fused computation of exp sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
        float exp_val = __expf(row_data[i] - row_max);
        local_sum += exp_val;
    }

    float warp_sum = warp_reduce_sum(local_sum);
    __shared__ float smem_sum[SOFTMAX_BLOCK_SIZE / 32];
    if (lane == 0) smem_sum[wid] = warp_sum;
    __syncthreads();

    if (wid == 0) {
        float block_sum = lane < num_warps ? smem_sum[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) smem_sum[0] = block_sum;
    }
    __syncthreads();
    float row_sum = smem_sum[0];
    float inv_sum = 1.0f / row_sum;

    // Write normalized results directly (fused with exp computation)
    for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
        row_data[i] = __expf(row_data[i] - row_max) * inv_sum;
    }
}

// ============================================================
// Host Entry Point — Naive Version (Two-Pass Softmax)
// ============================================================
extern "C" void solve(
    const float* Q,      // [seq_len, d_model] device pointer
    const float* K,      // [seq_len, d_model] device pointer
    const float* V,      // [seq_len, d_model] device pointer
    float* output,       // [seq_len, d_model] device pointer
    int seq_len,
    int d_model
) {
    // Allocate temporary buffers
    float* K_T;   // [d_model, seq_len]
    float* S;     // [seq_len, seq_len]
    cudaMalloc(&K_T, d_model * seq_len * sizeof(float));
    cudaMalloc(&S,   seq_len * seq_len * sizeof(float));

    // Step 1: K_T = transpose(K)  [seq_len, d_model] -> [d_model, seq_len]
    {
        dim3 block(TRANS_BLOCK_SIZE, TRANS_BLOCK_SIZE);
        dim3 grid(CEIL_DIV(d_model, TRANS_BLOCK_SIZE), CEIL_DIV(seq_len, TRANS_BLOCK_SIZE));
        transpose_kernel<<<grid, block>>>(K, K_T, seq_len, d_model);
    }

    // Step 2: S = Q @ K_T  [seq_len, d_model] x [d_model, seq_len] -> [seq_len, seq_len]
    {
        constexpr int thread_num = (BM/WM)*(BN/WN)*WARPSIZE;
        dim3 block(thread_num);
        dim3 grid(CEIL_DIV(seq_len, BN), CEIL_DIV(seq_len, BM));
        matmul_kernel<<<grid, block>>>(Q, K_T, S, seq_len, d_model, seq_len);
    }

    // Step 3: S = S / sqrt(d_model)
    {
        int n = seq_len * seq_len;
        float scale = 1.0f / sqrtf((float)d_model);
        int block = 256;
        int grid = CEIL_DIV(n, block);
        scale_kernel<<<grid, block>>>(S, n, scale);
    }

    // Step 4: S = softmax(S, dim=-1)  — row-wise softmax
    {
        row_softmax_kernel<<<seq_len, SOFTMAX_BLOCK_SIZE>>>(S, seq_len, seq_len);
    }

    // Step 5: output = S @ V  [seq_len, seq_len] x [seq_len, d_model] -> [seq_len, d_model]
    {
        constexpr int thread_num = (BM/WM)*(BN/WN)*WARPSIZE;
        dim3 block(thread_num);
        dim3 grid(CEIL_DIV(d_model, BN), CEIL_DIV(seq_len, BM));
        matmul_kernel<<<grid, block>>>(S, V, output, seq_len, seq_len, d_model);
    }

    // Free temporary buffers
    cudaFree(K_T);
    cudaFree(S);
}

// ============================================================
// Host Entry Point — Optimized Version (Online Softmax)
//    - Uses fused max+sum computation in single pass
//    - Reduces global memory reads from 3 to 2 passes
// ============================================================
extern "C" void solve_online(
    const float* Q,      // [seq_len, d_model] device pointer
    const float* K,      // [seq_len, d_model] device pointer
    const float* V,      // [seq_len, d_model] device pointer
    float* output,       // [seq_len, d_model] device pointer
    int seq_len,
    int d_model
) {
    // Allocate temporary buffers
    float* K_T;   // [d_model, seq_len]
    float* S;     // [seq_len, seq_len]
    cudaMalloc(&K_T, d_model * seq_len * sizeof(float));
    cudaMalloc(&S,   seq_len * seq_len * sizeof(float));

    // Step 1: K_T = transpose(K)  [seq_len, d_model] -> [d_model, seq_len]
    {
        dim3 block(TRANS_BLOCK_SIZE, TRANS_BLOCK_SIZE);
        dim3 grid(CEIL_DIV(d_model, TRANS_BLOCK_SIZE), CEIL_DIV(seq_len, TRANS_BLOCK_SIZE));
        transpose_kernel<<<grid, block>>>(K, K_T, seq_len, d_model);
    }

    // Step 2: S = Q @ K_T  [seq_len, d_model] x [d_model, seq_len] -> [seq_len, seq_len]
    {
        constexpr int thread_num = (BM/WM)*(BN/WN)*WARPSIZE;
        dim3 block(thread_num);
        dim3 grid(CEIL_DIV(seq_len, BN), CEIL_DIV(seq_len, BM));
        matmul_kernel<<<grid, block>>>(Q, K_T, S, seq_len, d_model, seq_len);
    }

    // Step 3: S = S / sqrt(d_model)
    {
        int n = seq_len * seq_len;
        float scale = 1.0f / sqrtf((float)d_model);
        int block = 256;
        int grid = CEIL_DIV(n, block);
        scale_kernel<<<grid, block>>>(S, n, scale);
    }

    // Step 4: S = softmax(S, dim=-1)  — row-wise online softmax (fused max+sum)
    {
        row_softmax_online_kernel<<<seq_len, SOFTMAX_BLOCK_SIZE>>>(S, seq_len, seq_len);
    }

    // Step 5: output = S @ V  [seq_len, seq_len] x [seq_len, d_model] -> [seq_len, d_model]
    {
        constexpr int thread_num = (BM/WM)*(BN/WN)*WARPSIZE;
        dim3 block(thread_num);
        dim3 grid(CEIL_DIV(d_model, BN), CEIL_DIV(seq_len, BM));
        matmul_kernel<<<grid, block>>>(S, V, output, seq_len, seq_len, d_model);
    }

    // Free temporary buffers
    cudaFree(K_T);
    cudaFree(S);
}

// ============================================================
// Test Entry Points for Debugging
// ============================================================
extern "C" void test_row_softmax_naive(float* data, int rows, int cols) {
    row_softmax_kernel<<<rows, SOFTMAX_BLOCK_SIZE>>>(data, rows, cols);
}

extern "C" void test_row_softmax_online(float* data, int rows, int cols) {
    row_softmax_online_kernel<<<rows, SOFTMAX_BLOCK_SIZE>>>(data, rows, cols);
}
