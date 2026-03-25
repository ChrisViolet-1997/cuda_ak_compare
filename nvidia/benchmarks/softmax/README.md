# Softmax CUDA 性能对比测试

对比三种不同优化级别的 Softmax CUDA kernel 实现，展示从基础到高级的 GPU 并行优化思路。

## 内核实现

### 1. Naive — 全局原子操作

每个线程独立通过 `atomicMaxFloat` / `atomicAdd` 操作全局内存，实现最简单但竞争最严重。

**三阶段流水线：**
- `naive_max_kernel`：每个线程对全局变量执行 `atomicMaxFloat` 求最大值
- `naive_sum_kernel`：每个线程计算 `exp(x - max)` 并 `atomicAdd` 到全局 sum
- `naive_softmax_kernel`：每个线程计算 `exp(x - max) / sum`

**瓶颈：** 所有线程争抢同一个全局地址，atomic 竞争随数据规模线性增长。

### 2. Block — 共享内存 Block 级归约

在 block 内使用 shared memory 做树形归约，每个 block 只产生一次 atomic 操作。

**优化要点：**
- 每个 block 加载数据到 `__shared__ float sdata[256]`
- 二叉树归约（步长逐次减半，配合 `__syncthreads()`）
- 仅 block 首线程执行一次 `atomicMaxFloat` / `atomicAdd`

**效果：** atomic 次数从 N 降低到 `ceil(N / BLOCK_SIZE)`，大幅减少全局竞争。

### 3. Warp — Warp Shuffle + 多级归约

最高优化级别，利用 warp shuffle 指令和每线程多元素处理（TM=8）。

**优化要点：**
- 每个线程处理 8 个元素（`NUM_BLOCK = 256 × 8 = 2048`），减少 kernel launch 开销
- warp 内使用 `__shfl_down_sync()` 做无 shared memory 的快速归约
- block 内仅需 `256/32 = 8` 个 float 的 shared memory 聚合各 warp 结果
- 最终每 block 一次 atomic 写回

**效果：** warp shuffle 延迟极低，shared memory 用量最小，吞吐量最高。

## 文件结构

```
nvidia/
├── kernels/softmax/
│   ├── softmax_naive.cu      # Naive 全局原子操作版本
│   ├── softmax_block.cu      # Block 级共享内存归约版本
│   └── softmax_warp.cu       # Warp shuffle 多级归约版本
└── benchmarks/softmax/
    ├── benchmark_softmax.py   # 性能测试脚本
    └── README.md
```

## 运行方式

**依赖：** CUDA 12.x、Python 3、CuPy、NumPy

```bash
python benchmark_softmax.py
```

脚本会自动编译三个 kernel 为 `.so`，通过 ctypes 调用并执行 100 次取平均。

## Benchmark 结果

测试环境：CUDA 12.8，100 次平均

| 数据规模 | Naive (ms) | Block (ms) | Warp (ms) | Block/Naive | Warp/Naive | Warp/Block |
|----------|-----------|-----------|----------|------------|-----------|-----------|
| 1,024 | 0.0203 | 0.0200 | 0.0201 | 1.02x | 1.01x | 0.99x |
| 4,096 | 0.0262 | 0.0198 | 0.0202 | 1.33x | 1.30x | 0.98x |
| 16,384 | 0.0706 | 0.0201 | 0.0202 | 3.51x | 3.50x | 1.00x |
| 65,536 | 0.1872 | 0.0199 | 0.0301 | 9.43x | 6.22x | 0.66x |
| 262,144 | 0.7027 | 0.0245 | 0.0223 | 28.71x | 31.46x | 1.10x |
| 1,048,576 | 2.4837 | 0.0484 | 0.0341 | 51.32x | 72.81x | 1.42x |

所有规模下三个版本的计算结果均通过正确性验证（误差 < 1e-5）。

## 结果分析

- **Naive 版本**随数据增长性能急剧下降（1K→1M 耗时增长 122 倍），瓶颈在全局 atomic 竞争。
- **Block 版本**通过 shared memory 归约将 atomic 次数降至 block 数量级，1M 数据下获得 **51x** 加速。
- **Warp 版本**在大数据量（≥256K）时优势明显，1M 数据下相比 Naive 达到 **73x** 加速，相比 Block 约 **1.4x** 加速。Warp shuffle 的低延迟和多元素处理策略在数据量足够大时充分发挥优势。
- 小数据量（≤16K）时三者差距不大，kernel launch 开销占主导。
