#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cooperative_groups.h>

// --- Utility Macros ---
#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)
#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define LIMIT2(a, ra, b, rb) (((a)<(ra)) && ((b)<(rb)))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CFLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// --- Kernel Configuration ---
constexpr int WARPSIZE = 32;

// Block Tiling dimensions
constexpr int BM = 128;
constexpr int BK = 16;
constexpr int BN = 128;

// Warp Tiling dimensions
constexpr int WM = 64;
constexpr int WN = 64;

// Sub-Warp Tiling dimensions (for register blocking)
constexpr int WSUBM = 64;
constexpr int WSUBN = 16;

// Thread Tiling dimensions
constexpr int TM = 8;
constexpr int TN = 4;

// --- Static Asserts for Configuration Validation ---
static_assert(BM % WM == 0, "Block M must be a multiple of Warp M.");
static_assert(BN % WN == 0, "Block N must be a multiple of Warp N.");
static_assert(WM % WSUBM == 0, "Warp M must be a multiple of Sub-Warp M.");
static_assert(WN % WSUBN == 0, "Warp N must be a multiple of Sub-Warp N.");
static_assert(WSUBM % TM == 0, "Sub-Warp M must be a multiple of Thread M.");
static_assert(WSUBN % TN == 0, "Sub-Warp N must be a multiple of Thread N.");
static_assert(TN % 4 == 0, "Thread N must be a multiple of 4 for float4 vectorization.");
static_assert((WSUBM / TM) * (WSUBN / TN) == WARPSIZE, "Thread mapping within a warp is incorrect.");


/**
 * @brief Loads tiles from global memory into shared memory for matrices A and B.
 * Matrix A is transposed on-the-fly into shared memory to facilitate coalesced memory access
 * during the computation phase. Vectorized float4 loads are used for efficiency.
 */
template <const int BM, const int BK, const int BN, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN, const int kExtraCol>
__device__ void load_global_to_shared(const float *A, const float *B, const int M, const int K, const int N, float *As, float *Bs) {
    // Ensure vectorization is possible
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);

    constexpr int thread_num = (BM/WM)*(BN/WN)*WARPSIZE;
    static_assert(BK*BM % (thread_num*4) == 0, "A tile size not divisible by thread load capacity.");
    static_assert(BK*BN % (thread_num*4) == 0, "B tile size not divisible by thread load capacity.");

    // --- Load Tile A (and transpose) ---
    constexpr int ldg_a_niter = BK*BM / (thread_num*4);
    constexpr int a_tile_stride = BM / ldg_a_niter;
    static_assert(BM % ldg_a_niter == 0, "A tile stride calculation error.");

    const int a_tile_row = threadIdx.x / (BK / 4);
    const int a_tile_col = (threadIdx.x % (BK / 4)) * 4;

    #pragma unroll
    for (int i = 0; i < ldg_a_niter; ++i) {
        int row_offset = i * a_tile_stride;
        float4 ldg_a_reg = CFLOAT4(A[OFFSET(a_tile_row + row_offset, a_tile_col, K)]);
        As[OFFSET(a_tile_col    , a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.x;
        As[OFFSET(a_tile_col + 1, a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.y;
        As[OFFSET(a_tile_col + 2, a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.z;
        As[OFFSET(a_tile_col + 3, a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.w;
    }

    // --- Load Tile B ---
    constexpr int ldg_b_niter = BK*BN / (thread_num*4);
    constexpr int b_tile_stride = BK / ldg_b_niter;
    static_assert(BK % ldg_b_niter == 0, "B tile stride calculation error.");

    const int b_tile_row = threadIdx.x / (BN / 4);
    const int b_tile_col = (threadIdx.x % (BN / 4)) * 4;

    #pragma unroll
    for (int i = 0; i < ldg_b_niter; ++i) {
        int row_offset = i * b_tile_stride;
        FLOAT4(Bs[OFFSET(b_tile_row + row_offset, b_tile_col, BN)]) = CFLOAT4(B[OFFSET(b_tile_row + row_offset, b_tile_col, N)]);
    }
}

/**
 * @brief Performs matrix multiplication on tiles in shared memory.
 * Each warp computes a WMxWN portion of the output block. Results are accumulated in registers.
 */
template <const int BM, const int BK, const int BN, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN, const int kExtraCol>
__device__ void compute_mma_from_shared(float *As, float *Bs, float* Areg, float* Breg, float* Creg) {
    constexpr int WMITER = WM / WSUBM;
    constexpr int WNITER = WN / WSUBN;

    const int warp_idx = threadIdx.x / WARPSIZE;
    const int wy = warp_idx / (BN / WN);
    const int wx = warp_idx % (BN / WN);

    const int thread_idx = threadIdx.x % WARPSIZE;
    const int ty = thread_idx / (WSUBN / TN);
    const int tx = thread_idx % (WSUBN / TN);

    // Loop over the K-dimension of the tile
    #pragma unroll
    for (unsigned int k = 0; k < BK; ++k) {
        // Load sub-tiles from shared memory into registers
        #pragma unroll
        for (unsigned int wsy = 0; wsy < WMITER; ++wsy) {
            #pragma unroll
            for (unsigned int j = 0; j < TM; ++j) {
                Areg[wsy*TM + j] =
                    As[OFFSET(k, wy*WM + wsy*WSUBM + ty*TM + j, BM + kExtraCol)];
            }
        }
        #pragma unroll
        for (unsigned int wsx = 0; wsx < WNITER; ++wsx) {
            #pragma unroll
            for (unsigned int i = 0; i < TN; ++i) {
                Breg[wsx*TN + i] =
                    Bs[OFFSET(k, wx*WN + wsx*WSUBN + tx*TN + i, BN)];
            }
        }

        // Perform matrix multiplication using register data
        #pragma unroll
        for (unsigned int wsy = 0; wsy < WMITER; ++wsy) {
            #pragma unroll
            for (unsigned int wsx = 0; wsx < WNITER; ++wsx) {
                #pragma unroll
                for (int m = 0; m < TM; m++) {
                    #pragma unroll
                    for (int n = 0; n < TN; n++) {
                        Creg[OFFSET(wsy*TM + m, wsx*TN + n, WNITER*TN)] += Areg[wsy*TM + m] * Breg[wsx*TN + n];
                    }
                }
            }
        }
    }
}

/**
 * @brief Stores the computed tile from registers back to global memory.
 * Uses vectorized float4 stores for efficiency.
 */
template <const int BM, const int BK, const int BN, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void store_result_to_global(float *C, const int N, float *Creg) {
    constexpr int WMITER = WM / WSUBM;
    constexpr int WNITER = WN / WSUBN;

    const unsigned int warp_idx = threadIdx.x / WARPSIZE;
    const unsigned int wy = warp_idx / (BN / WN);
    const unsigned int wx = warp_idx % (BN / WN);

    const unsigned int thread_idx = threadIdx.x % WARPSIZE;
    const unsigned int ty = thread_idx / (WSUBN / TN);
    const unsigned int tx = thread_idx % (WSUBN / TN);

    // Pointer to the top-left of the warp's output tile
    C = &C[OFFSET(wy*WM, wx*WN, N)];

    #pragma unroll
    for (unsigned int wsy = 0; wsy < WMITER; ++wsy) {
        #pragma unroll
        for (unsigned int wsx = 0; wsx < WNITER; ++wsx) {
            float* Cws = &C[OFFSET(wsy*WSUBM, wsx*WSUBN, N)];
            #pragma unroll
            for (unsigned int j = 0; j < TM; j += 1) {
                // Since TN is a multiple of 4, this loop is safe
                #pragma unroll
                for (unsigned int i = 0; i < TN; i += 4) {
                    FLOAT4(Cws[OFFSET(ty*TM + j, tx*TN + i, N)])
                        = CFLOAT4(Creg[OFFSET(wsy*TM + j, wsx*TN + i, WNITER*TN)]);
                }
            }
        }
    }
}

/**
 * @brief Main logic for a thread block, orchestrating the GEMM computation.
 * Uses double buffering to hide global memory latency by overlapping computation with data fetching.
 */
template<const int BM, const int BK, const int BN, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void gemm_block_tile(const float* A, const float* B, float* C, const int M, const int K, const int N) {
    // Padding to avoid shared memory bank conflicts when transposing A
    constexpr int kExtraCol = 4;
    constexpr int kAsSize = BK*(BM+kExtraCol);
    constexpr int kBsSize = BK*BN;

    // Double buffer in shared memory
    __shared__ float As[2][kAsSize];
    __shared__ float Bs[2][kBsSize];

    // Register arrays for computation
    constexpr unsigned int WMITER = WM / WSUBM;
    constexpr unsigned int WNITER = WN / WSUBN;
    float Areg[WMITER*TM];
    float Breg[WNITER*TN];
    float Creg[WMITER*TM * WNITER*TN] = {0.0f};

    // --- Double Buffering Main Loop ---
    int buffer_idx = 0;

    // 1. Load the first tile (k=0) into buffer 0
    load_global_to_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
        A, B, M, K, N, As[buffer_idx], Bs[buffer_idx]
    );
    __syncthreads(); // Ensure first load is complete before starting compute

    // 2. Loop over K dimension, computing on the current tile while fetching the next
    for (int k = BK; k < K; k += BK) {
        // Advance pointers to the next tile in global memory
        A += BK;
        B += BK * N;

        // Start loading the next tile (k+1) into the alternate buffer (1-buffer_idx)
        load_global_to_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
            A, B, M, K, N, As[1 - buffer_idx], Bs[1 - buffer_idx]
        );

        // Compute using the current tile (k) from buffer (buffer_idx)
        compute_mma_from_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
            As[buffer_idx], Bs[buffer_idx], Areg, Breg, Creg
        );

        __syncthreads();

        // Swap buffers for the next iteration
        buffer_idx = 1 - buffer_idx;
    }

    // 3. Compute the final tile which was pre-fetched in the last loop iteration
    compute_mma_from_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
        As[buffer_idx], Bs[buffer_idx], Areg, Breg, Creg
    );

    // 4. Store final results from registers to global memory
    store_result_to_global<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN>(C, N, Creg);
}

/**
 * @brief Kernel entry point for the high-performance path (matrix sizes are divisible by tile sizes).
 */
template<const int BM, const int BK, const int BN, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN>
__global__ void matrix_multiplication_kernel(
    const float* A, const float* B, float* C, const int M, const int K, const int N) {
    // Calculate global block indices
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;

    // Set up pointers to the top-left of the matrices for this block
    A = &A[OFFSET(block_y*BM, 0, K)];
    B = &B[OFFSET(0, block_x*BN, N)];
    C = &C[OFFSET(block_y*BM, block_x*BN, N)];

    gemm_block_tile<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN>(A, B, C, M, K, N);
}

// Explicit instantiation for the default configuration
extern "C" __global__
void matrix_mul_optimized_kernel(
    const float* A, const float* B, float* C, const int M, const int K, const int N) {
    // Calculate global block indices
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;

    // Set up pointers to the top-left of the matrices for this block
    const float* A_block = &A[OFFSET(block_y*BM, 0, K)];
    const float* B_block = &B[OFFSET(0, block_x*BN, N)];
    float* C_block = &C[OFFSET(block_y*BM, block_x*BN, N)];

    gemm_block_tile<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN>(A_block, B_block, C_block, M, K, N);
}
