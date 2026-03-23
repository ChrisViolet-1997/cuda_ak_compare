#!/usr/bin/env python3
"""
Test BK parameter impact on large matrix sizes (8192 and 16384)
"""
import numpy as np
import cupy as cp
import pandas as pd
from tune_parameters import generate_kernel_code, validate_config

print("=" * 80)
print("BK Parameter Impact on Large Matrix Sizes")
print("=" * 80)

# Base configuration (best from tuning)
BASE_CONFIG = {
    'BM': 128, 'BN': 128,
    'WM': 64, 'WN': 64,
    'WSUBM': 64, 'WSUBN': 16,
    'TM': 8, 'TN': 4
}

# Test different BK values
BK_VALUES = [8, 16, 32]

# Test matrix sizes
TEST_SIZES = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
]

print(f"\nBase Configuration:")
print(f"  Block Tiling: BM={BASE_CONFIG['BM']}, BN={BASE_CONFIG['BN']}")
print(f"  Warp Tiling:  WM={BASE_CONFIG['WM']}, WN={BASE_CONFIG['WN']}")
print(f"  Sub-Warp:     WSUBM={BASE_CONFIG['WSUBM']}, WSUBN={BASE_CONFIG['WSUBN']}")
print(f"  Thread:       TM={BASE_CONFIG['TM']}, TN={BASE_CONFIG['TN']}")
print(f"\nTesting BK values: {BK_VALUES}")
print(f"Matrix sizes: {TEST_SIZES}\n")

def benchmark_bk(BK, M, K, N, warmup=3, iterations=10):
    """Benchmark a specific BK value"""
    config = {**BASE_CONFIG, 'BK': BK}

    # Validate
    valid, msg = validate_config(
        config['BM'], BK, config['BN'],
        config['WM'], config['WN'],
        config['WSUBM'], config['WSUBN'],
        config['TM'], config['TN']
    )

    if not valid:
        return None, msg

    try:
        # Generate and compile kernel
        kernel_code = generate_kernel_code(**config)
        module = cp.RawModule(code=kernel_code, options=('--std=c++17',))
        kernel = module.get_function('matrix_mul_optimized_kernel')

        # Prepare data
        A_gpu = cp.random.randn(M, K, dtype=cp.float32)
        B_gpu = cp.random.randn(K, N, dtype=cp.float32)
        C_gpu = cp.zeros((M, N), dtype=cp.float32)

        # Calculate grid dimensions
        BM, BN = config['BM'], config['BN']
        WM, WN = config['WM'], config['WN']
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
        std_time = np.std(times)

        # Calculate GFLOPS
        gflops = (2.0 * M * N * K) / (avg_time * 1e-3) / 1e9

        # Correctness check using CuBLAS as reference
        C_ref = cp.zeros((M, N), dtype=cp.float32)
        # Use CuBLAS for reference (C = A @ B)
        C_ref = cp.matmul(A_gpu, B_gpu)

        # Compare results
        max_diff = float(cp.max(cp.abs(C_ref - C_gpu)))
        rel_error = max_diff / (float(cp.max(cp.abs(C_ref))) + 1e-10)

        if rel_error > 1e-3:
            return None, f"Correctness check failed: rel_error={rel_error:.2e}, max_diff={max_diff:.2e}"

        # Calculate shared memory usage
        kExtraCol = 4
        shared_mem = (BK*(BM+kExtraCol) + BK*BN) * 2 * 4  # 2 buffers, 4 bytes per float

        return {
            'time_ms': avg_time,
            'std_ms': std_time,
            'gflops': gflops,
            'shared_mem_bytes': shared_mem,
            'threads': threads_per_block[0],
            'blocks': blocks_per_grid[0] * blocks_per_grid[1]
        }, "OK"

    except Exception as e:
        return None, str(e)

results = []

for M, K, N in TEST_SIZES:
    print("=" * 80)
    print(f"Matrix Size: {M} x {K} x {N}")
    print(f"Total FLOPs: {2.0 * M * N * K / 1e12:.2f} TFLOP")
    print("=" * 80)

    size_results = []

    for BK in BK_VALUES:
        print(f"\nTesting BK={BK}...")

        result, msg = benchmark_bk(BK, M, K, N)

        if result is not None:
            print(f"  ✓ Time: {result['time_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print(f"    GFLOPS: {result['gflops']:.2f}")
            print(f"    Shared Memory: {result['shared_mem_bytes']} bytes ({result['shared_mem_bytes']/1024:.1f} KB)")
            print(f"    Threads/Block: {result['threads']}, Blocks: {result['blocks']}")

            size_results.append({
                'size': f"{M}x{K}x{N}",
                'M': M, 'K': K, 'N': N,
                'BK': BK,
                **result
            })
        else:
            print(f"  ✗ Failed: {msg}")

    if len(size_results) > 0:
        # Find best for this size
        best = max(size_results, key=lambda x: x['gflops'])
        worst = min(size_results, key=lambda x: x['gflops'])

        print(f"\n{'─' * 80}")
        print(f"Summary for {M}x{K}x{N}:")
        print(f"  Best:  BK={best['BK']:2d} → {best['gflops']:.2f} GFLOPS ({best['time_ms']:.3f} ms)")
        print(f"  Worst: BK={worst['BK']:2d} → {worst['gflops']:.2f} GFLOPS ({worst['time_ms']:.3f} ms)")
        print(f"  Performance variation: {(best['gflops'] / worst['gflops'] - 1) * 100:.1f}%")

        results.extend(size_results)

# Overall summary
print("\n" + "=" * 80)
print("Overall Results")
print("=" * 80)

if len(results) > 0:
    df = pd.DataFrame(results)

    print("\nAll Results:")
    print("-" * 80)
    print(f"{'Size':<20} {'BK':<5} {'Time (ms)':<12} {'GFLOPS':<12} {'Shared Mem (KB)'}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['size']:<20} {row['BK']:<5} {row['time_ms']:<12.3f} "
              f"{row['gflops']:<12.2f} {row['shared_mem_bytes']/1024:<.1f}")

    # Group by size and analyze
    print("\n" + "=" * 80)
    print("Analysis by Matrix Size")
    print("=" * 80)

    for size in df['size'].unique():
        size_df = df[df['size'] == size].sort_values('gflops', ascending=False)
        print(f"\n{size}:")
        print(f"  Best BK: {size_df.iloc[0]['BK']} ({size_df.iloc[0]['gflops']:.2f} GFLOPS)")
        print(f"  Ranking by performance:")
        for i, (_, row) in enumerate(size_df.iterrows(), 1):
            speedup = size_df.iloc[0]['gflops'] / row['gflops']
            print(f"    {i}. BK={row['BK']:2d}: {row['gflops']:8.2f} GFLOPS "
                  f"({row['time_ms']:7.3f} ms) - {speedup:.3f}x vs best")

    # Save results
    df.to_csv('bk_tuning_large_matrices.csv', index=False)
    print(f"\n✓ Results saved to bk_tuning_large_matrices.csv")

    # Key findings
    print("\n" + "=" * 80)
    print("Key Findings")
    print("=" * 80)

    for size in df['size'].unique():
        size_df = df[df['size'] == size]
        best_bk = size_df.loc[size_df['gflops'].idxmax(), 'BK']
        best_gflops = size_df['gflops'].max()
        worst_gflops = size_df['gflops'].min()
        variation = (best_gflops / worst_gflops - 1) * 100

        print(f"\n{size}:")
        print(f"  • Optimal BK: {best_bk}")
        print(f"  • Performance range: {worst_gflops:.2f} - {best_gflops:.2f} GFLOPS")
        print(f"  • BK impact: {variation:.1f}% performance variation")

        # Shared memory analysis
        for _, row in size_df.iterrows():
            if row['BK'] == best_bk:
                print(f"  • Shared memory usage (BK={best_bk}): {row['shared_mem_bytes']/1024:.1f} KB")

    print("\nConclusion:")
    print("  BK parameter affects the balance between:")
    print("  - Shared memory usage (larger BK = more shared memory)")
    print("  - Number of iterations in K-loop (smaller BK = more iterations)")
    print("  - Data reuse efficiency (larger BK = more reuse per load)")

else:
    print("\n✗ No successful benchmarks")
