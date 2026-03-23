#!/usr/bin/env python3
import numpy as np
import cupy as cp
import time
from pathlib import Path
import matplotlib.pyplot as plt

def load_kernel_from_file(cu_file, kernel_name):
    """Load and compile CUDA kernel from .cu file"""
    with open(cu_file, 'r') as f:
        code = f.read()

    # Extract kernel code (remove extern "C" wrapper function)
    # We'll use the kernel directly
    kernel = cp.RawKernel(code, kernel_name)
    return kernel

def benchmark_naive_kernel(A, B, C, M, K, N, warmup=5, iterations=20):
    """Benchmark naive kernel"""
    kernel_code = r'''
extern "C" __global__
void matrix_mul_naive_kernel(const float* A, const float* B, float* C,
                             int M, int K, int N) {
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
'''
    kernel = cp.RawKernel(kernel_code, 'matrix_mul_naive_kernel')
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (N + threads_per_block[0] - 1) // threads_per_block[0],
        (M + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Warmup
    for _ in range(warmup):
        kernel(blocks_per_grid, threads_per_block, (A, B, C, M, K, N))
        cp.cuda.Stream.null.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        kernel(blocks_per_grid, threads_per_block, (A, B, C, M, K, N))
        end.record()
        end.synchronize()

        times.append(cp.cuda.get_elapsed_time(start, end))

    return np.mean(times), np.std(times)

def benchmark_optimized_kernel(A, B, C, M, K, N, warmup=5, iterations=20):
    """Benchmark optimized kernel"""
    # Read the optimized kernel code
    with open('../../kernels/matrix_mul/optimized_kernel.cu', 'r') as f:
        kernel_code = f.read()

    # Compile the module
    module = cp.RawModule(code=kernel_code, options=('--std=c++17',))

    # Get the kernel function
    kernel = module.get_function('matrix_mul_optimized_kernel')

    # Calculate grid dimensions
    BM, BN = 128, 128
    WARPSIZE = 32
    WM, WN = 64, 64
    threads_per_block = ((BN//WN)*(BM//WM)*WARPSIZE,)  # Should be 128
    blocks_per_grid = (N // BN, M // BM)

    print(f"    Grid: {blocks_per_grid}, Block: {threads_per_block}")

    # Warmup
    for _ in range(warmup):
        kernel(blocks_per_grid, threads_per_block, (A, B, C, M, K, N))
        cp.cuda.Stream.null.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        kernel(blocks_per_grid, threads_per_block, (A, B, C, M, K, N))
        end.record()
        end.synchronize()

        times.append(cp.cuda.get_elapsed_time(start, end))

    return np.mean(times), np.std(times)

def verify_correctness(A_host, B_host, C_gpu, M, K, N):
    """Verify kernel output against NumPy"""
    C_ref = A_host @ B_host
    C_gpu_host = cp.asnumpy(C_gpu)

    max_diff = np.max(np.abs(C_ref - C_gpu_host))
    rel_error = max_diff / (np.max(np.abs(C_ref)) + 1e-10)

    return max_diff, rel_error

def main():
    print("=" * 70)
    print("Matrix Multiplication Kernel Comparison")
    print("=" * 70)

    # Test configurations (use sizes divisible by 128 for optimized kernel)
    test_sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    results = {
        'sizes': [],
        'naive_time': [],
        'optimized_time': [],
        'speedup': []
    }

    for M, K, N in test_sizes:
        print(f"\n[Testing] M={M}, K={K}, N={N}")

        # Generate test data
        A_host = np.random.randn(M, K).astype(np.float32)
        B_host = np.random.randn(K, N).astype(np.float32)

        A_gpu = cp.asarray(A_host)
        B_gpu = cp.asarray(B_host)
        C_gpu = cp.zeros((M, N), dtype=np.float32)

        # Benchmark naive kernel
        print("  - Benchmarking naive kernel...")
        naive_time, naive_std = benchmark_naive_kernel(
            A_gpu, B_gpu, C_gpu, M, K, N
        )

        # Verify correctness
        max_diff, rel_error = verify_correctness(A_host, B_host, C_gpu, M, K, N)
        print(f"    Time: {naive_time:.3f} ± {naive_std:.3f} ms")
        print(f"    Max diff: {max_diff:.2e}, Rel error: {rel_error:.2e}")

        if rel_error > 1e-3:
            print("    ✗ Naive kernel FAILED correctness check!")
            continue

        # Reset output
        C_gpu.fill(0)

        # Benchmark optimized kernel
        print("  - Benchmarking optimized kernel...")
        try:
            opt_time, opt_std = benchmark_optimized_kernel(
                A_gpu, B_gpu, C_gpu, M, K, N
            )

            # Verify correctness
            max_diff, rel_error = verify_correctness(A_host, B_host, C_gpu, M, K, N)
            print(f"    Time: {opt_time:.3f} ± {opt_std:.3f} ms")
            print(f"    Max diff: {max_diff:.2e}, Rel error: {rel_error:.2e}")

            if rel_error > 1e-3:
                print("    ✗ Optimized kernel FAILED correctness check!")
                continue

            speedup = naive_time / opt_time
            print(f"  - Speedup: {speedup:.2f}x")

            results['sizes'].append(f"{M}x{K}x{N}")
            results['naive_time'].append(naive_time)
            results['optimized_time'].append(opt_time)
            results['speedup'].append(speedup)

        except Exception as e:
            print(f"    ✗ Error running optimized kernel: {e}")
            continue

    # Print summary
    if len(results['sizes']) > 0:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"{'Size':<15} {'Naive (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10}")
        print("-" * 70)
        for i in range(len(results['sizes'])):
            print(f"{results['sizes'][i]:<15} "
                  f"{results['naive_time'][i]:<15.3f} "
                  f"{results['optimized_time'][i]:<15.3f} "
                  f"{results['speedup'][i]:<10.2f}x")

        # Plot results
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        x = range(len(results['sizes']))
        plt.bar([i - 0.2 for i in x], results['naive_time'], width=0.4, label='Naive', alpha=0.8)
        plt.bar([i + 0.2 for i in x], results['optimized_time'], width=0.4, label='Optimized', alpha=0.8)
        plt.xlabel('Matrix Size')
        plt.ylabel('Time (ms)')
        plt.title('Execution Time Comparison')
        plt.xticks(x, results['sizes'], rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(results['sizes'], results['speedup'], marker='o', linewidth=2, markersize=8)
        plt.xlabel('Matrix Size')
        plt.ylabel('Speedup')
        plt.title('Speedup (Naive / Optimized)')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('matrix_mul_comparison.png', dpi=150)
        print(f"\n✓ Results saved to matrix_mul_comparison.png")
    else:
        print("\n✗ No successful benchmarks to display")

if __name__ == '__main__':
    main()