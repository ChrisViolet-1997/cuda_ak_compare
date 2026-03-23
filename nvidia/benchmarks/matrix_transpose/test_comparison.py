#!/usr/bin/env python3
import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
import os

def load_kernel_code(kernel_file):
    """Load kernel code from file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(script_dir, '../../kernels/matrix_transpose', kernel_file)

    with open(kernel_path, 'r') as f:
        kernel_code = f.read()

    # Wrap with extern "C" for CuPy
    kernel_code = f'extern "C" {{\n{kernel_code}\n}}'

    return kernel_code

def benchmark_kernel(kernel_code, kernel_name, input_gpu, output_gpu,
                     rows, cols, block_size, warmup=5, iterations=20):
    """Benchmark a matrix transpose kernel"""
    try:
        # Compile with BLOCK_SIZE macro for optimized kernel
        compile_options = (f'-DBLOCK_SIZE={block_size}',)
        module = cp.RawModule(code=kernel_code, options=compile_options)
        kernel = module.get_function(kernel_name)
    except Exception as e:
        print(f"    ✗ Failed to compile kernel: {e}")
        return None, None

    threads_per_block = (block_size, block_size)
    blocks_per_grid = (
        (cols + block_size - 1) // block_size,
        (rows + block_size - 1) // block_size
    )

    # Warmup
    for _ in range(warmup):
        kernel(blocks_per_grid, threads_per_block,
               (input_gpu, output_gpu, rows, cols))
        cp.cuda.Stream.null.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        output_gpu.fill(0)
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        kernel(blocks_per_grid, threads_per_block,
               (input_gpu, output_gpu, rows, cols))
        end.record()
        end.synchronize()

        times.append(cp.cuda.get_elapsed_time(start, end))

    return np.mean(times), np.std(times)

def verify_correctness(input_host, output_gpu, rows, cols):
    """Verify kernel output against NumPy"""
    output_ref = input_host.T
    output_result = cp.asnumpy(output_gpu)

    max_diff = np.max(np.abs(output_ref - output_result))
    rel_error = max_diff / (np.max(np.abs(output_ref)) + 1e-10)

    return max_diff, rel_error

def main():
    print("=" * 80)
    print("Matrix Transpose Kernel Comparison")
    print("=" * 80)

    # Test configurations
    test_sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ]

    block_sizes = [8, 16, 32]
    kernels = [
        ('matrix_transpose.cu', 'Basic'),
        ('matrix_transpose_opt.cu', 'Optimized')
    ]

    # Results structure: {kernel_name: {block_size: [times]}}
    results = {}
    for _, kernel_name in kernels:
        results[kernel_name] = {bs: {'sizes': [], 'times': [], 'stds': []}
                                for bs in block_sizes}

    for rows, cols in test_sizes:
        print(f"\n{'='*80}")
        print(f"Testing Matrix size: {rows} x {cols}")
        print(f"{'='*80}")

        # Generate test data
        input_host = np.random.randn(rows, cols).astype(np.float32)
        input_gpu = cp.asarray(input_host)
        output_gpu = cp.zeros((cols, rows), dtype=np.float32)

        for kernel_file, kernel_name in kernels:
            print(f"\n[{kernel_name} Kernel]")

            for block_size in block_sizes:
                print(f"  Block size {block_size}x{block_size}:", end=" ")

                # Load kernel code
                kernel_code = load_kernel_code(kernel_file)

                # Benchmark
                mean_time, std_time = benchmark_kernel(
                    kernel_code, 'matrix_transpose_kernel',
                    input_gpu, output_gpu, rows, cols, block_size
                )

                if mean_time is None:
                    continue

                # Verify correctness
                max_diff, rel_error = verify_correctness(input_host, output_gpu, rows, cols)

                if rel_error > 1e-5:
                    print(f"✗ FAILED (error: {rel_error:.2e})")
                    continue

                print(f"{mean_time:.3f} ± {std_time:.3f} ms")

                # Store results
                results[kernel_name][block_size]['sizes'].append(f"{rows}x{cols}")
                results[kernel_name][block_size]['times'].append(mean_time)
                results[kernel_name][block_size]['stds'].append(std_time)

                # Reset output for next test
                output_gpu.fill(0)

    # Print summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    for kernel_file, kernel_name in kernels:
        print(f"\n{kernel_name} Kernel:")
        print(f"{'Size':<15}", end="")
        for bs in block_sizes:
            print(f"Block={bs} (ms)".ljust(18), end="")
        print()
        print("-" * 80)

        # Find common sizes across all block sizes
        common_sizes = set(results[kernel_name][block_sizes[0]]['sizes'])
        for bs in block_sizes[1:]:
            common_sizes &= set(results[kernel_name][bs]['sizes'])

        for size in test_sizes:
            size_str = f"{size[0]}x{size[1]}"
            if size_str not in common_sizes:
                continue

            print(f"{size_str:<15}", end="")
            for bs in block_sizes:
                idx = results[kernel_name][bs]['sizes'].index(size_str)
                time_val = results[kernel_name][bs]['times'][idx]
                print(f"{time_val:<18.3f}", end="")
            print()

    # Bandwidth analysis
    print(f"\n{'='*80}")
    print("Bandwidth Analysis (GB/s)")
    print(f"{'='*80}")

    for kernel_file, kernel_name in kernels:
        print(f"\n{kernel_name} Kernel:")
        print(f"{'Size':<15}", end="")
        for bs in block_sizes:
            print(f"Block={bs}".ljust(18), end="")
        print()
        print("-" * 80)

        common_sizes = set(results[kernel_name][block_sizes[0]]['sizes'])
        for bs in block_sizes[1:]:
            common_sizes &= set(results[kernel_name][bs]['sizes'])

        for size in test_sizes:
            size_str = f"{size[0]}x{size[1]}"
            if size_str not in common_sizes:
                continue

            rows, cols = size
            bytes_transferred = rows * cols * 4 * 2  # read + write

            print(f"{size_str:<15}", end="")
            for bs in block_sizes:
                idx = results[kernel_name][bs]['sizes'].index(size_str)
                time_val = results[kernel_name][bs]['times'][idx]
                bw = bytes_transferred / (time_val * 1e-3) / 1e9
                print(f"{bw:<18.2f}", end="")
            print()

    # Plot results
    plot_results(results, kernels, block_sizes, test_sizes)

def plot_results(results, kernels, block_sizes, test_sizes):
    """Generate visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, (kernel_file, kernel_name) in enumerate(kernels):
        ax = axes[idx, 0]

        # Plot execution time
        for i, bs in enumerate(block_sizes):
            sizes = results[kernel_name][bs]['sizes']
            times = results[kernel_name][bs]['times']

            if len(sizes) > 0:
                x = range(len(sizes))
                ax.plot(x, times, marker='o', label=f'Block={bs}',
                       color=colors[i], linewidth=2, markersize=6)

        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{kernel_name} Kernel - Execution Time')
        ax.set_xticks(range(len(test_sizes)))
        ax.set_xticklabels([f"{s[0]}x{s[1]}" for s in test_sizes], rotation=45)
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot bandwidth
        ax = axes[idx, 1]

        for i, bs in enumerate(block_sizes):
            sizes = results[kernel_name][bs]['sizes']
            times = results[kernel_name][bs]['times']

            if len(sizes) > 0:
                bandwidths = []
                for j, size_str in enumerate(sizes):
                    rows, cols = map(int, size_str.split('x'))
                    bytes_transferred = rows * cols * 4 * 2
                    bw = bytes_transferred / (times[j] * 1e-3) / 1e9
                    bandwidths.append(bw)

                x = range(len(sizes))
                ax.plot(x, bandwidths, marker='s', label=f'Block={bs}',
                       color=colors[i], linewidth=2, markersize=6)

        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Bandwidth (GB/s)')
        ax.set_title(f'{kernel_name} Kernel - Memory Bandwidth')
        ax.set_xticks(range(len(test_sizes)))
        ax.set_xticklabels([f"{s[0]}x{s[1]}" for s in test_sizes], rotation=45)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('matrix_transpose_comparison.png', dpi=150)
    print(f"\n✓ Results saved to matrix_transpose_comparison.png")

if __name__ == '__main__':
    main()
