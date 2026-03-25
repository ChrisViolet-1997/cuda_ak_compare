import numpy as np
import cupy as cp
import torch
import torch.nn.functional as F
import time
import subprocess
import ctypes
import os
from scipy.special import softmax as scipy_softmax


def compile_cuda(source_file, output_file):
    """Compile CUDA source to shared library"""
    cmd = [
        '/usr/local/cuda-12.8/bin/nvcc',
        '-shared',
        '-Xcompiler', '-fPIC',
        '-o', output_file,
        source_file,
        '-O3',
        '--std=c++17',
        '-lcudart',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation error:\n{result.stderr}")
        raise RuntimeError("Compilation failed")


def benchmark_custom_attention(lib, Q, K, V, output, seq_len, d_model, n_runs=100, use_online=False):
    """Benchmark the custom CUDA attention kernel"""
    q_ptr = Q.data.ptr
    k_ptr = K.data.ptr
    v_ptr = V.data.ptr
    o_ptr = output.data.ptr

    solve_func = lib.solve_online if use_online else lib.solve

    # Warmup
    for _ in range(10):
        solve_func(q_ptr, k_ptr, v_ptr, o_ptr, seq_len, d_model)
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        solve_func(q_ptr, k_ptr, v_ptr, o_ptr, seq_len, d_model)
    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / n_runs * 1000  # ms
    return avg_time


def benchmark_pytorch_attention(Q_torch, K_torch, V_torch, d_model, n_runs=100):
    """Benchmark PyTorch's scaled_dot_product_attention (cuDNN/FlashAttention)"""
    # PyTorch SDPA expects (batch, num_heads, seq_len, d_model)
    # We treat our single-head [seq_len, d_model] as (1, 1, seq_len, d_model)
    Q_4d = Q_torch.unsqueeze(0).unsqueeze(0)
    K_4d = K_torch.unsqueeze(0).unsqueeze(0)
    V_4d = V_torch.unsqueeze(0).unsqueeze(0)

    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(Q_4d, K_4d, V_4d)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = F.scaled_dot_product_attention(Q_4d, K_4d, V_4d)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / n_runs * 1000  # ms
    output = F.scaled_dot_product_attention(Q_4d, K_4d, V_4d)
    torch.cuda.synchronize()
    return avg_time, output.squeeze(0).squeeze(0).cpu().numpy()


def benchmark_pytorch_manual_attention(Q_torch, K_torch, V_torch, d_model, n_runs=100):
    """Benchmark PyTorch manual attention: softmax(Q @ K^T / sqrt(d)) @ V using cuBLAS"""
    scale = 1.0 / (d_model ** 0.5)

    # Warmup
    for _ in range(10):
        S = torch.matmul(Q_torch, K_torch.T) * scale
        S = torch.softmax(S, dim=-1)
        _ = torch.matmul(S, V_torch)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        S = torch.matmul(Q_torch, K_torch.T) * scale
        S = torch.softmax(S, dim=-1)
        out = torch.matmul(S, V_torch)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / n_runs * 1000  # ms
    return avg_time, out.cpu().numpy()


def reference_attention(Q_np, K_np, V_np, d_model):
    """Compute reference attention on CPU using NumPy/SciPy"""
    S = Q_np @ K_np.T
    S = S / np.sqrt(d_model)
    S = scipy_softmax(S, axis=-1)
    output = S @ V_np
    return output


def compute_flops(seq_len, d_model):
    """Compute total FLOPS for attention"""
    #   matmul1 (Q @ K^T):  2*seq_len*seq_len*d_model
    #   matmul2 (S @ V):    2*seq_len*seq_len*d_model
    #   scale:              seq_len*seq_len
    #   softmax:            ~5*seq_len*seq_len (exp, sub, div, sum, max)
    #   transpose:          seq_len*d_model
    total_flops = (2 * 2 * seq_len * seq_len * d_model +
                   seq_len * seq_len * 6 +
                   seq_len * d_model)
    return total_flops


def main():
    # Test configurations
    seq_lens = [128, 256, 512, 1024]
    d_models = [128, 256]
    n_runs = 100

    # Compile custom kernel
    kernel_dir = '/root/cuda_ak_compare/nvidia/kernels/attention'
    attention_so = '/tmp/attention.so'

    print("Compiling custom attention kernel...")
    compile_cuda(f'{kernel_dir}/attention.cu', attention_so)
    print("Compilation successful.\n")

    # Load shared library
    lib = ctypes.CDLL(attention_so)
    lib.solve.argtypes = [
        ctypes.c_void_p,  # Q
        ctypes.c_void_p,  # K
        ctypes.c_void_p,  # V
        ctypes.c_void_p,  # output
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # d_model
    ]
    lib.solve_online.argtypes = [
        ctypes.c_void_p,  # Q
        ctypes.c_void_p,  # K
        ctypes.c_void_p,  # V
        ctypes.c_void_p,  # output
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # d_model
    ]

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"CuPy: {cp.__version__}\n")

    SEP_WIDTH = 120
    print("=" * SEP_WIDTH)
    print("Scaled Dot-Product Attention Benchmark")
    print("  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_model)) @ V")
    print("  Custom Naive:   5 kernel launches with two-pass softmax (3 memory traversals)")
    print("  Custom Online:  5 kernel launches with online softmax (2 memory traversals, fused max+sum)")
    print("  PyTorch SDPA:   F.scaled_dot_product_attention (FlashAttention / cuDNN backend)")
    print("  PyTorch Manual: torch.matmul + torch.softmax (cuBLAS + cuDNN)")
    print("=" * SEP_WIDTH)

    header = (f"{'seq_len':<10} {'d_model':<10} {'Method':<20} "
              f"{'Time(ms)':<12} {'GFLOPS':<12} {'Max Diff':<14} {'Status'}")
    print(header)
    print("-" * SEP_WIDTH)

    for seq_len in seq_lens:
        for d_model in d_models:
            # Generate random input data
            Q_np = np.random.randn(seq_len, d_model).astype(np.float32)
            K_np = np.random.randn(seq_len, d_model).astype(np.float32)
            V_np = np.random.randn(seq_len, d_model).astype(np.float32)

            # CPU reference result
            output_ref = reference_attention(Q_np, K_np, V_np, d_model)

            # Total FLOPS for this configuration
            total_flops = compute_flops(seq_len, d_model)

            # --- Custom CUDA Kernel (Naive) ---
            Q_gpu = cp.asarray(Q_np)
            K_gpu = cp.asarray(K_np)
            V_gpu = cp.asarray(V_np)
            output_gpu = cp.zeros((seq_len, d_model), dtype=cp.float32)

            custom_time = benchmark_custom_attention(
                lib, Q_gpu, K_gpu, V_gpu, output_gpu, seq_len, d_model, n_runs, use_online=False)
            custom_result = cp.asnumpy(output_gpu)
            custom_diff = np.max(np.abs(custom_result - output_ref))
            custom_gflops = total_flops / (custom_time / 1000) / 1e9
            custom_status = "PASS" if custom_diff < 1e-3 else "FAIL"

            print(f"{seq_len:<10} {d_model:<10} {'Custom Naive':<20} "
                  f"{custom_time:<12.4f} {custom_gflops:<12.1f} {custom_diff:<14.2e} {custom_status}")

            # --- Custom CUDA Kernel (Online Softmax) ---
            output_gpu.fill(0)  # Reset output buffer

            online_time = benchmark_custom_attention(
                lib, Q_gpu, K_gpu, V_gpu, output_gpu, seq_len, d_model, n_runs, use_online=True)
            online_result = cp.asnumpy(output_gpu)
            online_diff = np.max(np.abs(online_result - output_ref))
            online_gflops = total_flops / (online_time / 1000) / 1e9
            online_status = "PASS" if online_diff < 1e-3 else "FAIL"

            print(f"{'':<10} {'':<10} {'Custom Online':<20} "
                  f"{online_time:<12.4f} {online_gflops:<12.1f} {online_diff:<14.2e} {online_status}")

            # --- PyTorch SDPA (FlashAttention / cuDNN) ---
            Q_torch = torch.from_numpy(Q_np).cuda()
            K_torch = torch.from_numpy(K_np).cuda()
            V_torch = torch.from_numpy(V_np).cuda()

            sdpa_time, sdpa_result = benchmark_pytorch_attention(
                Q_torch, K_torch, V_torch, d_model, n_runs)
            sdpa_diff = np.max(np.abs(sdpa_result - output_ref))
            sdpa_gflops = total_flops / (sdpa_time / 1000) / 1e9
            sdpa_status = "PASS" if sdpa_diff < 1e-3 else "FAIL"

            print(f"{'':<10} {'':<10} {'PyTorch SDPA':<20} "
                  f"{sdpa_time:<12.4f} {sdpa_gflops:<12.1f} {sdpa_diff:<14.2e} {sdpa_status}")

            # --- PyTorch Manual (cuBLAS + cuDNN softmax) ---
            manual_time, manual_result = benchmark_pytorch_manual_attention(
                Q_torch, K_torch, V_torch, d_model, n_runs)
            manual_diff = np.max(np.abs(manual_result - output_ref))
            manual_gflops = total_flops / (manual_time / 1000) / 1e9
            manual_status = "PASS" if manual_diff < 1e-3 else "FAIL"

            print(f"{'':<10} {'':<10} {'PyTorch Manual':<20} "
                  f"{manual_time:<12.4f} {manual_gflops:<12.1f} {manual_diff:<14.2e} {manual_status}")

            # --- Speedup summary ---
            speedup_online_vs_naive = custom_time / online_time
            speedup_vs_sdpa = sdpa_time / online_time
            speedup_vs_manual = manual_time / online_time
            print(f"{'':<10} {'':<10} {'>> Speedup':<20} "
                  f"{'Online vs Naive:':<12} {speedup_online_vs_naive:<12.2f}x")
            print(f"{'':<10} {'':<10} {'':<20} "
                  f"{'vs SDPA:':<12} {speedup_vs_sdpa:<12.2f}x {'vs Manual:':<14} {speedup_vs_manual:.2f}x")
            print("-" * SEP_WIDTH)

    print("=" * SEP_WIDTH)


if __name__ == '__main__':
    main()
