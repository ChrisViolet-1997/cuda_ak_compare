import matplotlib.pyplot as plt
import numpy as np

# 测试数据
block_sizes = [64, 128, 256, 512, 1024]

# 执行时间 (ms)
basic_times = [2.628, 1.499, 1.495, 1.498, 1.504]
vectorized_times = [1.812, 1.654, 1.577, 2.030, 1.917]
shared_mem_times = [2.442, 1.503, 1.498, 1.502, 1.509]

# 带宽 (GB/s)
basic_bw = [456.6, 800.6, 802.7, 801.1, 797.7]
vectorized_bw = [662.3, 725.7, 761.1, 591.1, 626.0]
shared_mem_bw = [491.4, 798.2, 800.8, 799.1, 795.4]

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 子图1: 执行时间对比
ax1.plot(block_sizes, basic_times, 'o-', label='Basic', linewidth=2, markersize=8)
ax1.plot(block_sizes, vectorized_times, 's-', label='Vectorized', linewidth=2, markersize=8)
ax1.plot(block_sizes, shared_mem_times, '^-', label='SharedMem', linewidth=2, markersize=8)
ax1.set_xlabel('Block Size', fontsize=12)
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('Vector Add Performance: Execution Time', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# 子图2: 带宽对比
ax2.plot(block_sizes, basic_bw, 'o-', label='Basic', linewidth=2, markersize=8)
ax2.plot(block_sizes, vectorized_bw, 's-', label='Vectorized', linewidth=2, markersize=8)
ax2.plot(block_sizes, shared_mem_bw, '^-', label='SharedMem', linewidth=2, markersize=8)
ax2.set_xlabel('Block Size', fontsize=12)
ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax2.set_title('Vector Add Performance: Memory Bandwidth', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)
ax2.axhline(y=900, color='r', linestyle='--', alpha=0.5, label='V100 Peak (~900 GB/s)')

plt.tight_layout()
plt.savefig('vector_add_performance.png', dpi=300, bbox_inches='tight')
print("图表已保存为 vector_add_performance.png")
