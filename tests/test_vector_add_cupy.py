import numpy as np
import cupy as cp
import ctypes
import time

class VectorAddTester:
    def __init__(self, lib_path):
        """加载编译好的共享库"""
        self.lib = ctypes.CDLL(lib_path)

        # 设置函数参数类型
        for func_name in ['launchVectorAddBasic', 'launchVectorAddVectorized', 'launchVectorAddSharedMem']:
            func = getattr(self.lib, func_name)
            func.argtypes = [
                ctypes.c_void_p,  # d_a
                ctypes.c_void_p,  # d_b
                ctypes.c_void_p,  # d_c
                ctypes.c_int,     # n
                ctypes.c_int      # blockSize
            ]
            func.restype = None

    def test_implementation(self, name, launcher_func, h_a, h_b, n, block_size=256):
        """测试单个实现"""
        print(f"\n测试 {name} 实现:")

        # 分配GPU内存并拷贝数据
        d_a = cp.asarray(h_a)
        d_b = cp.asarray(h_b)
        d_c = cp.zeros(n, dtype=cp.float32)

        # 获取GPU指针
        a_ptr = ctypes.c_void_p(d_a.data.ptr)
        b_ptr = ctypes.c_void_p(d_b.data.ptr)
        c_ptr = ctypes.c_void_p(d_c.data.ptr)

        # 预热
        launcher_func(a_ptr, b_ptr, c_ptr, n, block_size)
        cp.cuda.Stream.null.synchronize()

        # 执行kernel并计时
        start = time.time()
        launcher_func(a_ptr, b_ptr, c_ptr, n, block_size)
        cp.cuda.Stream.null.synchronize()
        end = time.time()

        # 验证结果
        h_c = cp.asnumpy(d_c)
        expected = h_a + h_b
        if np.allclose(h_c, expected, rtol=1e-5):
            print(f"✓ 结果正确")
            print(f"  执行时间: {(end-start)*1000:.3f} ms")
        else:
            print(f"✗ 结果错误")
            max_diff = np.max(np.abs(h_c - expected))
            print(f"  最大误差: {max_diff}")

        return end - start

    def run_all_tests(self, n=1000000, block_size=256):
        """运行所有测试"""
        print("="*50)
        print("Vector Add 性能测试")
        print(f"数组大小: {n}")
        print(f"Block大小: {block_size}")
        print("="*50)

        # 准备测试数据（在CPU上）
        h_a = np.arange(n, dtype=np.float32)
        h_b = np.arange(n, dtype=np.float32) * 2

        # 测试三种实现
        results = {}
        results['Basic'] = self.test_implementation(
            'Basic', self.lib.launchVectorAddBasic, h_a, h_b, n, block_size
        )
        results['Vectorized'] = self.test_implementation(
            'Vectorized', self.lib.launchVectorAddVectorized, h_a, h_b, n, block_size
        )
        results['SharedMem'] = self.test_implementation(
            'SharedMem', self.lib.launchVectorAddSharedMem, h_a, h_b, n, block_size
        )

        # 输出性能对比
        print("\n" + "="*50)
        print("性能对比:")
        baseline = results['Basic']
        for name, time_cost in results.items():
            speedup = baseline / time_cost if time_cost > 0 else 0
            print(f"  {name:12s}: {time_cost*1000:8.3f} ms  (加速比: {speedup:.2f}x)")
        print("="*50)

if __name__ == '__main__':
    import sys

    # 默认库路径
    lib_path = './libvector_add.so'
    n = 1000000
    block_size = 256

    if len(sys.argv) > 1:
        lib_path = sys.argv[1]
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    if len(sys.argv) > 3:
        block_size = int(sys.argv[3])

    tester = VectorAddTester(lib_path)
    tester.run_all_tests(n, block_size)
