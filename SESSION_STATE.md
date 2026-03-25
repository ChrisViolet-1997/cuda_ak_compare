# CUDA Attention Kernel 优化项目 - 会话状态

**最后更新**: 2026-03-25
**当前阶段**: Softmax算子融合优化 (已完成)

---

## 项目概述

这是一个CUDA kernel学习项目，对比手写kernel与PyTorch标准库的性能差异。

**目录结构**:
```
nvidia/
├── kernels/
│   ├── attention/
│   │   └── attention.cu              # 主kernel文件（包含naive和online softmax）
│   └── softmax/                      # 其他softmax实现
└── benchmarks/
    ├── attention/
    │   ├── benchmark_attention.py    # 性能测试脚本
    │   ├── README.md                 # 基础文档
    │   └── OPTIMIZATION_LOG.md       # 优化记录
    └── softmax/                      # Softmax单独测试
```

---

## 当前实现状态

### Attention计算流程 (5个kernel)

1. **Transpose**: `transpose_kernel` - K矩阵转置
2. **GEMM**: `matmul_kernel` - Q @ K^T
3. **Scale**: `scale_kernel` - 除以sqrt(d_model)
4. **Softmax**: `row_softmax_kernel` (naive) 或 `row_softmax_online_kernel` (优化)
5. **GEMM**: `matmul_kernel` - S @ V

### 已实现的优化

#### Softmax算子融合 (2026-03-25完成)

**两个版本对比**:

| 版本 | 函数名 | 内存遍历次数 | 性能 |
|------|--------|--------------|------|
| Naive | `row_softmax_kernel` | 3次 | 基准 |
| Online | `row_softmax_online_kernel` | 2次 | 大矩阵提升10-17% |

**Host接口**:
- `solve()` - 使用naive softmax
- `solve_online()` - 使用优化后的online softmax

**性能数据** (V100, 100次平均):
```
1024×128:  1.1060ms → 1.0051ms  (1.10x加速)
1024×256:  1.2411ms → 1.0609ms  (1.17x加速)
```

**优化原理**:
```
Naive:   求max → 计算exp+写回+求sum → 读取+归一化  (3次遍历)
Online:  求max → 计算exp+求sum → 直接写归一化结果  (2次遍历)
```

---

## 代码关键位置

### attention.cu 中的重要函数

| 行号范围 | 函数/Kernel | 说明 |
|---------|-------------|------|
| 20-37 | `transpose_kernel` | 矩阵转置，使用shared memory + padding避免bank conflict |
| 235-246 | `matmul_kernel` | GEMM主kernel，调用block tiling函数 |
| 62-233 | `gemm_block_tile` | GEMM核心实现，double buffering + warp tiling |
| 251-256 | `scale_kernel` | 逐元素缩放 |
| 283-338 | `row_softmax_kernel` | **Naive softmax** (3次遍历) |
| 340-402 | `row_softmax_online_kernel` | **优化softmax** (2次遍历) |
| 447-481 | `solve()` | Host入口，使用naive softmax |
| 488-522 | `solve_online()` | Host入口，使用online softmax |

### benchmark_attention.py 关键函数

| 行号范围 | 函数 | 说明 |
|---------|------|------|
| 30-50 | `benchmark_custom_attention` | 自定义kernel性能测试，支持use_online参数 |
| 53-76 | `benchmark_pytorch_attention` | PyTorch SDPA测试 |
| 79-100 | `benchmark_pytorch_manual` | PyTorch手动拼接测试 |
| 125-242 | `main` | 主测试流程，对比4种实现 |

---

## 已知问题

1. **512×256配置性能异常**
   - Online版本在此配置下反而慢26% (0.74x)
   - 可能原因：exp重复计算、cache miss
   - 需要用Nsight Compute profiling分析

2. **小矩阵提升不明显**
   - ≤256×256配置下加速比接近1.0
   - Kernel launch固定开销占主导

---

## 下一步优化路线

### 优先级1: 解决exp重复计算
**问题**: Online版本中exp被计算2次
- Phase 2求sum时: `exp_val = __expf(row_data[i] - row_max)`
- 写回时: `__expf(row_data[i] - row_max) * inv_sum`

**方案**: 使用shared memory缓存exp值
```cuda
// Phase 2: 计算exp并缓存到shared memory
__shared__ float smem_exp[SOFTMAX_BLOCK_SIZE];
for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
    float exp_val = __expf(row_data[i] - row_max);
    smem_exp[tid] = exp_val;
    local_sum += exp_val;
}
// ... reduction ...
// 写回时从shared memory读取
for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
    row_data[i] = smem_exp[tid] * inv_sum;
}
```

**限制**: 需要考虑cols > SOFTMAX_BLOCK_SIZE的情况

### 优先级2: Scale + Softmax融合
**目标**: 将5个kernel减少到4个
- 当前: scale_kernel → row_softmax_kernel
- 优化后: fused_scale_softmax_kernel

**实现**: 在softmax的max查找阶段就除以scale
```cuda
// 读取时直接缩放
float x = row_data[i] * scale;
local_max = fmaxf(local_max, x);
```

### 优先级3: 真正的单次遍历Online Softmax
**难点**: Warp reduction需要正确合并(max, sum)对
```cuda
// 合并两个(m, s)对的公式：
m_new = max(m1, m2)
s_new = s1 * exp(m1 - m_new) + s2 * exp(m2 - m_new)
```

**之前的尝试**: 在warp reduction时产生NaN（数值不稳定）
**需要**: 仔细处理边界情况和初始化

### 优先级4: Profiling分析
使用Nsight Compute分析512×256性能下降：
```bash
ncu --set full -o profile python benchmark_attention.py
```

---

## 测试命令

**运行完整benchmark**:
```bash
cd /root/cuda_ak_compare/nvidia/benchmarks/attention
python benchmark_attention.py
```

**编译kernel**:
```bash
nvcc -shared -Xcompiler -fPIC \
     -o /tmp/attention.so \
     /root/cuda_ak_compare/nvidia/kernels/attention/attention.cu \
     -O3 --std=c++17 -lcudart
```

**查看GPU信息**:
```bash
nvidia-smi
```

---

## 参考资料

**已实现的优化技术**:
- Shared memory tiling (GEMM)
- Double buffering (GEMM)
- Warp-level shuffle reduction (Softmax)
- Operator fusion (Softmax)

**学习过的项目文件**:
- `nvidia/kernels/softmax/softmax_warp.cu` - Warp shuffle示例
- `nvidia/benchmarks/softmax/benchmark_softmax.py` - Softmax独立测试

**性能对比参考**:
- PyTorch SDPA: FlashAttention/cuDNN backend，单kernel fusion
- PyTorch Manual: cuBLAS + cuDNN，工业级优化
- Custom: 学习用，还有很大优化空间

---

## 环境信息

- **GPU**: Tesla V100-PCIE-32GB
- **CUDA**: 12.8
- **PyTorch**: 2.8.0+cu128
- **CuPy**: 14.0.1
- **工作目录**: /root/cuda_ak_compare

---

## 快速恢复checklist

下次继续时：
1. ✅ 检查git状态: `cd /root/cuda_ak_compare && git status`
2. ✅ 运行测试确认代码正常: `python nvidia/benchmarks/attention/benchmark_attention.py`
3. ✅ 查看优化日志: `cat nvidia/benchmarks/attention/OPTIMIZATION_LOG.md`
4. ✅ 选择优化方向，从"下一步优化路线"中选一个开始

**最后工作成果**:
- Softmax算子融合完成 ✅
- 大矩阵性能提升10-17% ✅
- 文档完整 ✅
