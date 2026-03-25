# Scaled Dot-Product Attention CUDA 性能对比测试

对比手写 CUDA Attention kernel 与 PyTorch 标准库实现的计算性能差异，展示自定义算子与工业级库之间的差距。

## 算法

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_model)) @ V
```

## 实现方式

### 1. Custom CUDA — 5 个独立 kernel 组合

手写 CUDA kernel，将 attention 拆分为 5 步顺序执行：

1. **Transpose**：shared memory + bank conflict padding（`TRANS_BLOCK_SIZE=16`）
2. **GEMM (Q @ K^T)**：double-buffered shared memory + warp tiling + float4 向量化加载（`BM=128, BN=128, BK=16`）
3. **Scale**：逐元素除以 `sqrt(d_model)`
4. **Row Softmax**：warp shuffle 归约（max → exp → sum → normalize）
5. **GEMM (S @ V)**：同上 GEMM kernel

**特点：** 纯 FP32 标量运算，无 Tensor Core；每次调用 `cudaMalloc`/`cudaFree` 分配临时 buffer。

### 2. PyTorch SDPA — `F.scaled_dot_product_attention`

PyTorch 内置的融合 attention 算子，后端自动选择 FlashAttention / cuDNN / Math 实现。单次 kernel launch 完成全部计算，避免中间结果的显存读写。

### 3. PyTorch Manual — `torch.matmul` + `torch.softmax`

使用 PyTorch 标准操作手动拼接 attention 流程，底层调用 cuBLAS GEMM + cuDNN softmax。与 Custom CUDA 逻辑相同，但每个算子都是厂商深度调优版本。

## 文件结构

```
nvidia/
├── kernels/attention/
│   └── attention.cu            # 手写 CUDA attention 实现
└── benchmarks/attention/
    ├── benchmark_attention.py   # 性能对比测试脚本
    └── README.md
```

## 运行方式

**依赖：** CUDA 12.x、Python 3、CuPy、PyTorch、NumPy、SciPy

```bash
python benchmark_attention.py
```

脚本自动编译 `.cu` 为 `.so`，通过 ctypes 调用自定义 kernel，同时运行 PyTorch 两种实现进行对比。每组配置 warmup 10 次 + 正式运行 100 次取平均。

## Benchmark 结果

**测试环境：** Tesla V100-PCIE-32GB / CUDA 12.8 / PyTorch 2.8.0 / 100 次平均

### 计算时间 (ms)

| seq_len | d_model | Custom CUDA | PyTorch SDPA | PyTorch Manual |
|---------|---------|-------------|--------------|----------------|
| 128 | 128 | 0.1446 | 0.0279 | 0.0785 |
| 128 | 256 | 0.2001 | 0.0453 | 0.0681 |
| 256 | 128 | 0.2015 | 0.0504 | 0.0877 |
| 256 | 256 | 0.2915 | 0.0870 | 0.0704 |
| 512 | 128 | 0.2987 | 0.0878 | 0.0791 |
| 512 | 256 | 0.3412 | 0.1544 | 0.1840 |
| 1024 | 128 | 1.5031 | 0.1672 | 0.0891 |
| 1024 | 256 | 1.1111 | 0.2991 | 0.1413 |

### 吞吐量 (GFLOPS)

| seq_len | d_model | Custom CUDA | PyTorch SDPA | PyTorch Manual |
|---------|---------|-------------|--------------|----------------|
| 128 | 128 | 58.8 | 305.2 | 108.4 |
| 128 | 256 | 84.5 | 373.4 | 248.4 |
| 256 | 128 | 168.6 | 674.2 | 387.4 |
| 256 | 256 | 231.8 | 776.5 | 959.9 |
| 512 | 128 | 454.8 | 1,546.5 | 1,718.4 |
| 512 | 256 | 791.6 | 1,749.6 | 1,468.0 |
| 1024 | 128 | 361.4 | 3,250.2 | 6,095.7 |
| 1024 | 256 | 972.3 | 3,611.9 | 7,643.2 |

### Custom CUDA 相对加速比

| seq_len | d_model | vs PyTorch SDPA | vs PyTorch Manual |
|---------|---------|-----------------|-------------------|
| 128 | 128 | 0.19x (慢 5.2 倍) | 0.54x (慢 1.9 倍) |
| 128 | 256 | 0.23x (慢 4.4 倍) | 0.34x (慢 2.9 倍) |
| 256 | 128 | 0.25x (慢 4.0 倍) | 0.44x (慢 2.3 倍) |
| 256 | 256 | 0.30x (慢 3.3 倍) | 0.24x (慢 4.1 倍) |
| 512 | 128 | 0.29x (慢 3.4 倍) | 0.26x (慢 3.8 倍) |
| 512 | 256 | 0.45x (慢 2.2 倍) | 0.54x (慢 1.9 倍) |
| 1024 | 128 | 0.11x (慢 9.0 倍) | 0.06x (慢 16.9 倍) |
| 1024 | 256 | 0.27x (慢 3.7 倍) | 0.13x (慢 7.9 倍) |

所有配置下三种实现均通过正确性验证（与 CPU 参考结果误差 < 1e-6）。

## 结果分析

### 1. Custom CUDA 在所有配置下均慢于标准库

- 小尺寸（seq_len ≤ 256）：慢 2–5 倍，5 次 kernel launch 的固定开销在小矩阵上占比过大
- 大尺寸（seq_len = 1024）：差距急剧拉大到 **7–17 倍**，说明 GEMM kernel 效率远不及 cuBLAS

### 2. PyTorch Manual 在大矩阵下表现最优

在 `1024×256` 配置下达到 **7,643 GFLOPS**，得益于 cuBLAS 对 V100 的极致调优（Tensor Core 利用、warp scheduling 等）。GEMM 是计算主体，cuBLAS 的优势在大矩阵下被充分放大。

### 3. PyTorch SDPA 在中小尺寸有优势

FlashAttention 通过 kernel fusion 减少中间矩阵的显存读写，在 `seq_len ≤ 512` 时效率高于 Manual 拆分调用。但在单 head、无 batch 的场景下不是其最优工作点，大尺寸下反而不如 cuBLAS 直接计算。

### 4. Custom CUDA 性能瓶颈分析

| 瓶颈 | 说明 |
|------|------|
| **无 Tensor Core** | 纯 FP32 标量 FMA，未利用 V100 的 FP16 Tensor Core（理论 125 TFLOPS vs FP32 15.7 TFLOPS） |
| **5 次 kernel launch** | 每步独立 launch 产生 ~5μs 开销，小矩阵时占比显著；标准库可做 kernel fusion |
| **运行时内存分配** | 每次 `solve()` 调用 `cudaMalloc`/`cudaFree`，引入 GPU 同步和系统调用开销 |
| **GEMM tiling 策略** | 固定 `BM=BN=128` 对小矩阵（128×128）padding 浪费严重，无 auto-tuning 机制 |
| **无 double buffering 预取** | 虽有 shared memory double buffer，但缺少 global memory 的异步预取（`cp.async`） |

### 5. 可改进方向

- 引入 `wmma` API 或 PTX `mma` 指令利用 Tensor Core
- 将 transpose + scale + softmax 融合为单个 kernel 减少 launch 次数
- 预分配临时 buffer 避免运行时 `cudaMalloc`
- 针对不同矩阵尺寸做 auto-tuning 选择最优 tile size
