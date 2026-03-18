#!/usr/bin/env python3
import numpy as np
import cupy as cp
import time
from itertools import product
import pandas as pd

def generate_kernel_code(BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN):
    """Generate optimized kernel code with specific parameters"""
    kernel_code = f'''
#include <cuda_runtime.h>

// --- Utility Macros ---
#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)
#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define LIMIT2(a, ra, b, rb) (((a)<(ra)) && ((b)<(rb)))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CFLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// --- Kernel Configuration ---
constexpr int WARPSIZE = 32;
constexpr int BM = {BM};
constexpr int BK = {BK};
constexpr int BN = {BN};
constexpr int WM = {WM};
constexpr int WN = {WN};
constexpr int WSUBM = {WSUBM};
constexpr int WSUBN = {WSUBN};
constexpr int TM = {TM};
constexpr int TN = {TN};

template <const int BM, const int BK, const int BN, const int WM, const int WN,
          const int WSUBM, const int WSUBN, const int TM, const int TN, const int kExtraCol>
__device__ void load_global_to_shared(const float *A, const float *B, const int M,
                                      const int K, const int N, float *As, float *Bs) {{
    constexpr int thread_num = (BM/WM)*(BN/WN)*WARPSIZE;

    // Load Tile A (and transpose)
    constexpr int ldg_a_niter = BK*BM / (thread_num*4);
    constexpr int a_tile_stride = BM / ldg_a_niter;

    const int a_tile_row = threadIdx.x / (BK / 4);
    const int a_tile_col = (threadIdx.x % (BK / 4)) * 4;

    #pragma unroll
    for (int i = 0; i < ldg_a_niter; ++i) {{
        int row_offset = i * a_tile_stride;
        float4 ldg_a_reg = CFLOAT4(A[OFFSET(a_tile_row + row_offset, a_tile_col, K)]);
        As[OFFSET(a_tile_col    , a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.x;
        As[OFFSET(a_tile_col + 1, a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.y;
        As[OFFSET(a_tile_col + 2, a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.z;
        As[OFFSET(a_tile_col + 3, a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.w;
    }}

    // Load Tile B
    constexpr int ldg_b_niter = BK*BN / (thread_num*4);
    constexpr int b_tile_stride = BK / ldg_b_niter;

    const int b_tile_row = threadIdx.x / (BN / 4);
    const int b_tile_col = (threadIdx.x % (BN / 4)) * 4;

    #pragma unroll
    for (int i = 0; i < ldg_b_niter; ++i) {{
        int row_offset = i * b_tile_stride;
        FLOAT4(Bs[OFFSET(b_tile_row + row_offset, b_tile_col, BN)]) =
            CFLOAT4(B[OFFSET(b_tile_row + row_offset, b_tile_col, N)]);
    }}
}}

template <const int BM, const int BK, const int BN, const int WM, const int WN,
          const int WSUBM, const int WSUBN, const int TM, const int TN, const int kExtraCol>
__device__ void compute_mma_from_shared(float *As, float *Bs, float* Areg, float* Breg, float* Creg) {{
    constexpr int WMITER = WM / WSUBM;
    constexpr int WNITER = WN / WSUBN;

    const int warp_idx = threadIdx.x / WARPSIZE;
    const int wy = warp_idx / (BN / WN);
    const int wx = warp_idx % (BN / WN);

    const int thread_idx = threadIdx.x % WARPSIZE;
    const int ty = thread_idx / (WSUBN / TN);
    const int tx = thread_idx % (WSUBN / TN);

    #pragma unroll
    for (unsigned int k = 0; k < BK; ++k) {{
        #pragma unroll
        for (unsigned int wsy = 0; wsy < WMITER; ++wsy) {{
            #pragma unroll
            for (unsigned int j = 0; j < TM; ++j) {{
                Areg[wsy*TM + j] = As[OFFSET(k, wy*WM + wsy*WSUBM + ty*TM + j, BM + kExtraCol)];
            }}
        }}
        #pragma unroll
        for (unsigned int wsx = 0; wsx < WNITER; ++wsx) {{
            #pragma unroll
            for (unsigned int i = 0; i < TN; ++i) {{
                Breg[wsx*TN + i] = Bs[OFFSET(k, wx*WN + wsx*WSUBN + tx*TN + i, BN)];
            }}
        }}

        #pragma unroll
        for (unsigned int wsy = 0; wsy < WMITER; ++wsy) {{
            #pragma unroll
            for (unsigned int wsx = 0; wsx < WNITER; ++wsx) {{
                #pragma unroll
                for (int m = 0; m < TM; m++) {{
                    #pragma unroll
                    for (int n = 0; n < TN; n++) {{
                        Creg[OFFSET(wsy*TM + m, wsx*TN + n, WNITER*TN)] +=
                            Areg[wsy*TM + m] * Breg[wsx*TN + n];
                    }}
                }}
            }}
        }}
    }}
}}

template <const int BM, const int BK, const int BN, const int WM, const int WN,
          const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void store_result_to_global(float *C, const int N, float *Creg) {{
    constexpr int WMITER = WM / WSUBM;
    constexpr int WNITER = WN / WSUBN;

    const unsigned int warp_idx = threadIdx.x / WARPSIZE;
    const unsigned int wy = warp_idx / (BN / WN);
    const unsigned int wx = warp_idx % (BN / WN);

    const unsigned int thread_idx = threadIdx.x % WARPSIZE;
    const unsigned int ty = thread_idx / (WSUBN / TN);
    const unsigned int tx = thread_idx % (WSUBN / TN);

    C = &C[OFFSET(wy*WM, wx*WN, N)];

    #pragma unroll
    for (unsigned int wsy = 0; wsy < WMITER; ++wsy) {{
        #pragma unroll
        for (unsigned int wsx = 0; wsx < WNITER; ++wsx) {{
            float* Cws = &C[OFFSET(wsy*WSUBM, wsx*WSUBN, N)];
            #pragma unroll
            for (unsigned int j = 0; j < TM; j += 1) {{
                #pragma unroll
                for (unsigned int i = 0; i < TN; i += 4) {{
                    FLOAT4(Cws[OFFSET(ty*TM + j, tx*TN + i, N)]) =
                        CFLOAT4(Creg[OFFSET(wsy*TM + j, wsx*TN + i, WNITER*TN)]);
                }}
            }}
        }}
    }}
}}

template<const int BM, const int BK, const int BN, const int WM, const int WN,
         const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void gemm_block_tile(const float* A, const float* B, float* C,
                                const int M, const int K, const int N) {{
    constexpr int kExtraCol = 4;
    constexpr int kAsSize = BK*(BM+kExtraCol);
    constexpr int kBsSize = BK*BN;

    __shared__ float As[2][kAsSize];
    __shared__ float Bs[2][kBsSize];

    constexpr unsigned int WMITER = WM / WSUBM;
    constexpr unsigned int WNITER = WN / WSUBN;
    float Areg[WMITER*TM];
    float Breg[WNITER*TN];
    float Creg[WMITER*TM * WNITER*TN] = {{0.0f}};

    int buffer_idx = 0;

    load_global_to_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
        A, B, M, K, N, As[buffer_idx], Bs[buffer_idx]
    );
    __syncthreads();

    for (int k = BK; k < K; k += BK) {{
        A += BK;
        B += BK * N;

        load_global_to_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
            A, B, M, K, N, As[1 - buffer_idx], Bs[1 - buffer_idx]
        );

        compute_mma_from_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
            As[buffer_idx], Bs[buffer_idx], Areg, Breg, Creg
        );

        __syncthreads();
        buffer_idx = 1 - buffer_idx;
    }}

    compute_mma_from_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
        As[buffer_idx], Bs[buffer_idx], Areg, Breg, Creg
    );

    store_result_to_global<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN>(C, N, Creg);
}}

extern "C" __global__
void matrix_mul_optimized_kernel(const float* A, const float* B, float* C,
                                 const int M, const int K, const int N) {{
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;

    const float* A_block = &A[OFFSET(block_y*BM, 0, K)];
    const float* B_block = &B[OFFSET(0, block_x*BN, N)];
    float* C_block = &C[OFFSET(block_y*BM, block_x*BN, N)];

    gemm_block_tile<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN>(A_block, B_block, C_block, M, K, N);
}}
'''
    return kernel_code

def validate_config(BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN):
    """Validate if a configuration is valid"""
    WARPSIZE = 32

    # Basic divisibility checks
    if BM % WM != 0: return False, "BM not divisible by WM"
    if BN % WN != 0: return False, "BN not divisible by WN"
    if WM % WSUBM != 0: return False, "WM not divisible by WSUBM"
    if WN % WSUBN != 0: return False, "WN not divisible by WSUBN"
    if WSUBM % TM != 0: return False, "WSUBM not divisible by TM"
    if WSUBN % TN != 0: return False, "WSUBN not divisible by TN"
    if TN % 4 != 0: return False, "TN not divisible by 4"

    # Thread mapping check
    if (WSUBM // TM) * (WSUBN // TN) != WARPSIZE:
        return False, "Thread mapping incorrect"

    # Vectorization checks
    if BK % 4 != 0: return False, "BK not divisible by 4"
    if BN % 4 != 0: return False, "BN not divisible by 4"

    # Thread count check
    thread_num = (BM//WM)*(BN//WN)*WARPSIZE
    if thread_num > 1024: return False, "Too many threads per block"
    if thread_num < 32: return False, "Too few threads per block"

    # Load balance checks
    if (BK*BM) % (thread_num*4) != 0: return False, "A tile load imbalance"
    if (BK*BN) % (thread_num*4) != 0: return False, "B tile load imbalance"

    # Shared memory check (rough estimate)
    kExtraCol = 4
    shared_mem = (BK*(BM+kExtraCol) + BK*BN) * 2 * 4  # 2 buffers, 4 bytes per float
    if shared_mem > 48*1024: return False, f"Too much shared memory: {shared_mem} bytes"

    return True, "OK"

def benchmark_config(config, M, K, N, warmup=3, iterations=10):
    """Benchmark a specific configuration"""
    BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN = config

    # Validate configuration
    valid, msg = validate_config(BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN)
    if not valid:
        return None, msg

    try:
        # Generate and compile kernel
        kernel_code = generate_kernel_code(BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN)
        module = cp.RawModule(code=kernel_code, options=('--std=c++17',))
        kernel = module.get_function('matrix_mul_optimized_kernel')

        # Prepare data
        A_gpu = cp.random.randn(M, K, dtype=cp.float32)
        B_gpu = cp.random.randn(K, N, dtype=cp.float32)
        C_gpu = cp.zeros((M, N), dtype=cp.float32)

        # Calculate grid dimensions
        WARPSIZE = 32
        threads_per_block = ((BN//WN)*(BM//WM)*WARPSIZE,)
        blocks_per_grid = (N // BN, M // BM)

        # Warmup
        for _ in range(warmup):
            kernel(blocks_per_grid, threads_per_block, (A_gpu, B_gpu, C_gpu, M, K, N))
            cp.cuda.Stream.null.synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = cp.cuda.Event()
            end = cp.cuda.Event()

            start.record()
            kernel(blocks_per_grid, threads_per_block, (A_gpu, B_gpu, C_gpu, M, K, N))
            end.record()
            end.synchronize()

            times.append(cp.cuda.get_elapsed_time(start, end))

        avg_time = np.mean(times)

        # Verify correctness (quick check)
        A_host = cp.asnumpy(A_gpu)
        B_host = cp.asnumpy(B_gpu)
        C_ref = A_host @ B_host
        C_result = cp.asnumpy(C_gpu)
        max_diff = np.max(np.abs(C_ref - C_result))
        rel_error = max_diff / (np.max(np.abs(C_ref)) + 1e-10)

        if rel_error > 1e-3:
            return None, f"Correctness check failed: rel_error={rel_error:.2e}"

        # Calculate GFLOPS
        gflops = (2.0 * M * N * K) / (avg_time * 1e-3) / 1e9

        return {
            'time_ms': avg_time,
            'gflops': gflops,
            'threads': threads_per_block[0],
            'blocks': blocks_per_grid[0] * blocks_per_grid[1]
        }, "OK"

    except Exception as e:
        return None, str(e)

def main():
    print("=" * 80)
    print("Matrix Multiplication Kernel Parameter Tuning for A100")
    print("=" * 80)

    # Test matrix size
    M, K, N = 4096, 4096, 4096
    print(f"\nTest size: M={M}, K={K}, N={N}")
    print(f"Total FLOPs: {2.0 * M * N * K / 1e9:.2f} GFLOP\n")

    # Define parameter search space
    # Format: (BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN)
    # Constraint: (WSUBM/TM) * (WSUBN/TN) = 32 (warp size)
    configs = [
        # Original configuration
        (128, 16, 128, 64, 64, 64, 16, 8, 4),  # (64/8)*(16/4) = 8*4 = 32 ✓

        # Vary BK (K-dimension blocking)
        (128, 8, 128, 64, 64, 64, 16, 8, 4),
        (128, 32, 128, 64, 64, 64, 16, 8, 4),

        # Different thread tile sizes (maintaining warp constraint)
        (128, 16, 128, 64, 64, 64, 8, 8, 4),   # (64/8)*(8/4) = 8*2 = 16 ✗
        (128, 16, 128, 64, 64, 64, 32, 8, 4),  # (64/8)*(32/4) = 8*8 = 64 ✗
        (128, 16, 128, 64, 64, 64, 16, 4, 4),  # (64/4)*(16/4) = 16*4 = 64 ✗
        (128, 16, 128, 64, 64, 64, 16, 16, 4), # (64/16)*(16/4) = 4*4 = 16 ✗
        (128, 16, 128, 64, 64, 64, 8, 4, 4),   # (64/4)*(8/4) = 16*2 = 32 ✓
        (128, 16, 128, 64, 64, 64, 32, 16, 4), # (64/16)*(32/4) = 4*8 = 32 ✓

        # Vary block sizes (keeping warp tiling valid)
        (64, 16, 64, 32, 32, 32, 16, 8, 4),    # (32/8)*(16/4) = 4*4 = 16 ✗
        (64, 16, 64, 32, 32, 32, 8, 4, 4),     # (32/4)*(8/4) = 8*2 = 16 ✗
        (64, 16, 64, 64, 64, 64, 16, 8, 4),    # (64/8)*(16/4) = 8*4 = 32 ✓
        (256, 16, 256, 128, 128, 128, 16, 8, 4), # (128/8)*(16/4) = 16*4 = 64 ✗
        (256, 16, 256, 128, 128, 128, 32, 16, 4), # (128/16)*(32/4) = 8*8 = 64 ✗
        (256, 16, 256, 64, 64, 64, 16, 8, 4),  # (64/8)*(16/4) = 8*4 = 32 ✓

        # Different warp tiling strategies
        (128, 16, 128, 32, 64, 32, 16, 8, 4),  # (32/8)*(16/4) = 4*4 = 16 ✗
        (128, 16, 128, 64, 32, 64, 16, 8, 4),  # (64/8)*(16/4) = 8*4 = 32 ✓
        (128, 16, 128, 32, 32, 32, 16, 8, 4),  # (32/8)*(16/4) = 4*4 = 16 ✗
        (128, 16, 128, 32, 32, 32, 8, 4, 4),   # (32/4)*(8/4) = 8*2 = 16 ✗

        # More TM/TN variations
        (128, 16, 128, 64, 64, 32, 16, 8, 4),  # (32/8)*(16/4) = 4*4 = 16 ✗
        (128, 16, 128, 64, 64, 32, 8, 4, 4),   # (32/4)*(8/4) = 8*2 = 16 ✗
        (128, 16, 128, 64, 64, 32, 16, 4, 4),  # (32/4)*(16/4) = 8*4 = 32 ✓
        (128, 16, 128, 64, 64, 16, 8, 4, 4),   # (16/4)*(8/4) = 4*2 = 8 ✗
        (128, 16, 128, 64, 64, 16, 8, 2, 4),   # (16/2)*(8/4) = 8*2 = 16 ✗

        # Rectangular block tiles
        (128, 16, 256, 64, 128, 64, 16, 8, 4), # (64/8)*(16/4) = 8*4 = 32 ✓
        (256, 16, 128, 128, 64, 128, 16, 8, 4), # (128/8)*(16/4) = 16*4 = 64 ✗
        (256, 16, 128, 64, 64, 64, 16, 8, 4),  # (64/8)*(16/4) = 8*4 = 32 ✓
        (128, 16, 256, 64, 64, 64, 16, 8, 4),  # (64/8)*(16/4) = 8*4 = 32 ✓

        # Smaller BK values
        (128, 4, 128, 64, 64, 64, 16, 8, 4),
        (128, 12, 128, 64, 64, 64, 16, 8, 4),

        # Larger blocks with valid thread mapping
        (192, 16, 192, 64, 64, 64, 16, 8, 4),  # (64/8)*(16/4) = 8*4 = 32 ✓
        (160, 16, 160, 64, 64, 64, 16, 8, 4),  # (64/8)*(16/4) = 8*4 = 32 ✓
    ]

    results = []

    print(f"Testing {len(configs)} configurations...\n")
    print("-" * 80)

    for i, config in enumerate(configs, 1):
        BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN = config
        config_str = f"BM={BM:3d} BK={BK:2d} BN={BN:3d} WM={WM:3d} WN={WN:3d} " \
                     f"WSUBM={WSUBM:3d} WSUBN={WSUBN:2d} TM={TM:2d} TN={TN:2d}"

        print(f"[{i:2d}/{len(configs)}] {config_str}")

        result, msg = benchmark_config(config, M, K, N)

        if result is not None:
            print(f"         ✓ {result['time_ms']:.3f} ms, {result['gflops']:.2f} GFLOPS, "
                  f"{result['threads']} threads/block, {result['blocks']} blocks")
            results.append({
                'config': config_str,
                'BM': BM, 'BK': BK, 'BN': BN,
                'WM': WM, 'WN': WN,
                'WSUBM': WSUBM, 'WSUBN': WSUBN,
                'TM': TM, 'TN': TN,
                **result
            })
        else:
            print(f"         ✗ Failed: {msg}")

    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)

    if len(results) > 0:
        df = pd.DataFrame(results)
        df = df.sort_values('gflops', ascending=False)

        print("\nTop 10 Configurations by Performance:")
        print("-" * 80)
        for i, row in df.head(10).iterrows():
            print(f"{row['config']}")
            print(f"  Time: {row['time_ms']:.3f} ms, GFLOPS: {row['gflops']:.2f}, "
                  f"Threads: {row['threads']}, Blocks: {row['blocks']}")

        print("\n" + "=" * 80)
        print("Best Configuration:")
        print("=" * 80)
        best = df.iloc[0]
        print(f"BM={best['BM']}, BK={best['BK']}, BN={best['BN']}")
        print(f"WM={best['WM']}, WN={best['WN']}")
        print(f"WSUBM={best['WSUBM']}, WSUBN={best['WSUBN']}")
        print(f"TM={best['TM']}, TN={best['TN']}")
        print(f"\nPerformance: {best['gflops']:.2f} GFLOPS ({best['time_ms']:.3f} ms)")
        print(f"Speedup vs worst: {df.iloc[0]['gflops'] / df.iloc[-1]['gflops']:.2f}x")

        # Save results
        df.to_csv('tuning_results.csv', index=False)
        print(f"\n✓ Results saved to tuning_results.csv")
    else:
        print("\n✗ No successful configurations found")

if __name__ == '__main__':
    main()
