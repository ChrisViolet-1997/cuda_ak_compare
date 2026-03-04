import numpy as np
import cupy as cp
import ctypes
import time

class VectorAddBenchmark:
    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)

        for func_name in ['launchVectorAddBasic', 'launchVectorAddVectorized', 'launchVectorAddSharedMem']:
            func = getattr(self.lib, func_name)
            func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            func.restype = None

    def benchmark(self, name, launcher_func, d_a, d_b, d_c, n, block_size, iterations=100):
        # 预热
        for _ in range(10):
            launcher_func(
                ctypes.c_void_p(d_a.data.ptr),
                ctypes.c_void_p(d_b.data.ptr),
                ctypes.c_void_p(d_c.data.ptr),
                n, block_size
            )
        cp.cuda.Stream.null.synchronize()

        # 计时
        start = time.time()
        for _ in range(iterations):
            launcher_func(
                ctypes.c_void_p(d_a.data.ptr),
                ctypes.c_void_p(d_b.data.ptr),
                ctypes.c_void_p(d_c.data.ptr),
                n, block_size
            )
        cp.cuda.Stream.null.synchronize()
        end = time.time()

        avg_time = (end - start) / iterations * 1000
        bandwidth = (3 * n * 4) / (avg_time / 1000) / 1e9  # GB/s
        return avg_time, bandwidth

    def run_benchmark(self, n=100000000):
        print("="*70)
        print(f"Vector Add 性能基准测试 (数组大小: {n:,})")
        print("="*70)

        h_a = np.random.rand(n).astype(np.float32)
        h_b = np.random.rand(n).astype(np.float32)

        d_a = cp.asarray(h_a)
        d_b = cp.asarray(h_b)
        d_c = cp.zeros(n, dtype=cp.float32)

        block_sizes = [64, 128, 256, 512, 1024]

        print(f"\n{'Block Size':<12} {'Basic (ms)':<15} {'Vectorized (ms)':<18} {'SharedMem (ms)':<15}")
        print("-"*70)

        for bs in block_sizes:
            basic_time, basic_bw = self.benchmark('Basic', self.lib.launchVectorAddBasic, d_a, d_b, d_c, n, bs)
            vec_time, vec_bw = self.benchmark('Vectorized', self.lib.launchVectorAddVectorized, d_a, d_b, d_c, n, bs)
            shared_time, shared_bw = self.benchmark('SharedMem', self.lib.launchVectorAddSharedMem, d_a, d_b, d_c, n, bs)

            print(f"{bs:<12} {basic_time:>6.3f} ({basic_bw:>5.1f}GB/s)  {vec_time:>6.3f} ({vec_bw:>5.1f}GB/s)  {shared_time:>6.3f} ({shared_bw:>5.1f}GB/s)")

        print("="*70)

if __name__ == '__main__':
    import sys
    lib_path = './libvector_add.so'
    n = 100000000

    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    benchmark = VectorAddBenchmark(lib_path)
    benchmark.run_benchmark(n)
