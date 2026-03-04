import numpy as np
import ctypes
import time
from pathlib import Path

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

    def test_implementation(self, name, launcher_func, a, b, n, block_size=256):
        """测试单个实现"""
        print(f"\n测试 {name} 实现:")

        # 准备结果数组
        c = np.zeros(n, dtype=np.float32)

        # 获取数据指针
        a_ptr = a.ctypes.data_as(ctypes.c_void_p)
        b_ptr = b.ctypes.data_as(ctypes.c_void_p)
        c_ptr = c.ctypes.data_as(ctypes.c_void_p)

        # 执行kernel
        start = time.time()
        launcher_func(a_ptr, b_ptr, c_ptr, n, block_size)
        end = time.time()

        # 验证结果
        expected = a + b
        if np.allclose(c, expected, rtol=1e-5):
            print(f"✓ 结果正确")
            print(f"  执行时间: {(end-start)*1000:.3f} ms")
        else:
            print(f"✗ 结果错误")
            max_diff = np.max(np.abs(c - expected))
            print(f"  最大误差: {max_diff}")

        return end - start

    def run_all_tests(self, n=1000000, block_size=256):
        """运行所有测试"""
        print("="*50)
        print("Vector Add 性能测试")
        print(f"数组大小: {n}")
        print(f"Block大小: {block_size}")
        print("="*50)

        # 准备测试数据
        a = np.arange(n, dtype=np.float32)
        b = np.arange(n, dtype=np.float32) * 2

        # 测试三种实现
        results = {}
        results['Basic'] = self.test_implementation(
            'Basic', self.lib.launchVectorAddBasic, a, b, n, block_size
        )
        results['Vectorized'] = self.test_implementation(
            'Vectorized', self.lib.launchVectorAddVectorized, a, b, n, block_size
        )
        results['SharedMem'] = self.test_implementation(
            'SharedMem', self.lib.launchVectorAddSharedMem, a, b, n, block_size
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
