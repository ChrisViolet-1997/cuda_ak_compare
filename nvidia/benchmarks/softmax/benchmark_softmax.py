import numpy as np
import cupy as cp
import time
import subprocess
import ctypes
import os

def compile_cuda(source_file, output_file):
    """编译CUDA文件为共享库"""
    cmd = [
        '/usr/local/cuda-12.8/bin/nvcc',
        '-shared',
        '-Xcompiler', '-fPIC',
        '-o', output_file,
        source_file,
        '-O3',
        '-lcudart'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"编译错误: {result.stderr}")
        raise RuntimeError("编译失败")

def benchmark_softmax_ctypes(lib, input_data, output_data, n_runs=100):
    """使用ctypes调用CUDA函数进行性能测试"""
    # 获取设备指针
    input_ptr = input_data.data.ptr
    output_ptr = output_data.data.ptr
    size = input_data.size

    # 预热
    for _ in range(10):
        lib.solve(input_ptr, output_ptr, size)
    cp.cuda.Stream.null.synchronize()

    # 正式测试
    start = time.perf_counter()
    for _ in range(n_runs):
        lib.solve(input_ptr, output_ptr, size)
    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / n_runs * 1000  # 转换为毫秒
    return avg_time

def verify_correctness(output, reference):
    """验证结果正确性"""
    max_diff = cp.max(cp.abs(output - reference))
    mean_diff = cp.mean(cp.abs(output - reference))
    return max_diff, mean_diff

def main():
    # 测试不同的数据规模
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]

    # 编译CUDA文件
    kernel_dir = '/root/cuda_ak_compare/nvidia/kernels/softmax'
    naive_so = '/tmp/softmax_naive.so'
    block_so = '/tmp/softmax_block.so'
    warp_so = '/tmp/softmax_warp.so'

    print("编译CUDA内核...")
    print("  - 编译naive版本（每个线程直接atomic）...")
    compile_cuda(f'{kernel_dir}/softmax_naive.cu', naive_so)
    print("  - 编译block版本（block级reduce + atomic）...")
    compile_cuda(f'{kernel_dir}/softmax_block.cu', block_so)
    print("  - 编译warp版本（warp shuffle + block级reduce + atomic）...")
    compile_cuda(f'{kernel_dir}/softmax_warp.cu', warp_so)

    # 加载共享库
    naive_lib = ctypes.CDLL(naive_so)
    block_lib = ctypes.CDLL(block_so)
    warp_lib = ctypes.CDLL(warp_so)

    # 设置函数参数类型
    naive_lib.solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    block_lib.solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    warp_lib.solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

    print("\n" + "="*110)
    print("Softmax性能对比测试 - 三种优化级别")
    print("="*110)
    print(f"{'数据规模':<12} {'Naive(ms)':<12} {'Block(ms)':<12} {'Warp(ms)':<12} {'Block/Naive':<13} {'Warp/Naive':<13} {'Warp/Block':<12} {'正确性'}")
    print("-"*110)

    for size in sizes:
        # 生成测试数据
        input_data = cp.random.randn(size, dtype=cp.float32)
        output_naive = cp.zeros(size, dtype=cp.float32)
        output_block = cp.zeros(size, dtype=cp.float32)
        output_warp = cp.zeros(size, dtype=cp.float32)

        # 使用CuPy计算参考结果
        reference = cp.exp(input_data - cp.max(input_data))
        reference = reference / cp.sum(reference)

        # 测试三个版本
        time_naive = benchmark_softmax_ctypes(naive_lib, input_data, output_naive)
        time_block = benchmark_softmax_ctypes(block_lib, input_data, output_block)
        time_warp = benchmark_softmax_ctypes(warp_lib, input_data, output_warp)

        # 计算加速比
        speedup_block_vs_naive = time_naive / time_block
        speedup_warp_vs_naive = time_naive / time_warp
        speedup_warp_vs_block = time_block / time_warp

        # 验证正确性
        max_diff_naive, _ = verify_correctness(output_naive, reference)
        max_diff_block, _ = verify_correctness(output_block, reference)
        max_diff_warp, _ = verify_correctness(output_warp, reference)

        correctness = "✓" if max_diff_warp < 1e-5 else "✗"

        print(f"{size:<12} {time_naive:<12.4f} {time_block:<12.4f} {time_warp:<12.4f} "
              f"{f'{speedup_block_vs_naive:.2f}x':<14}{f'{speedup_warp_vs_naive:.2f}x':<14}{f'{speedup_warp_vs_block:.2f}x':<13}{correctness}")

    print("="*110)

if __name__ == '__main__':
    main()
