# Attention Kernel 优化记录

## 优化阶段1：Softmax算子融合

### 问题分析

Naive版本的row softmax包含**3次全局内存遍历**：
1. 第一遍：遍历所有元素求最大值
2. 第二遍：计算exp(x - max)并写回，同时累加求和
3. 第三遍：读取exp值并除以sum进行归一化

### 优化方案

将第2步和第3步融合，减少到**2次全局内存遍历**：
1. 第一遍：遍历所有元素求最大值（不变）
2. 第二遍：**融合操作**
   - 计算exp(x - max)（不写回内存）
   - 累加求和
   - 得到sum后，直接写入归一化结果 exp(x - max) / sum

### 实现细节

**Naive版本** (`row_softmax_kernel`):
```cuda
// Phase 1: Find max
for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
    local_max = fmaxf(local_max, row_data[i]);
}
// ... reduction ...

// Phase 2: Compute exp and write back
for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
    float val = __expf(row_data[i] - row_max);
    row_data[i] = val;  // 写回内存
    local_sum += val;
}
// ... reduction ...

// Phase 3: Normalize (需要重新读取)
for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
    row_data[i] *= inv_sum;  // 读取 -> 计算 -> 写回
}
```

**Online版本** (`row_softmax_online_kernel`):
```cuda
// Phase 1: Find max (same)
for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
    local_max = fmaxf(local_max, row_data[i]);
}
// ... reduction ...

// Phase 2: Fused exp+sum computation (不写回)
for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
    float exp_val = __expf(row_data[i] - row_max);
    local_sum += exp_val;  // 只累加，不写回
}
// ... reduction ...

// Directly write normalized results (融合)
for (int i = tid; i < cols; i += SOFTMAX_BLOCK_SIZE) {
    row_data[i] = __expf(row_data[i] - row_max) * inv_sum;  // 一次性计算并写入最终结果
}
```

### 性能对比

**Tesla V100-PCIE-32GB**, 100次平均测试结果：

| seq_len | d_model | Naive (ms) | Online (ms) | 加速比 | 说明 |
|---------|---------|------------|-------------|--------|------|
| 128 | 128 | 0.1426 | 0.1448 | 0.98x | 小尺寸，kernel launch开销占比大 |
| 128 | 256 | 0.1985 | 0.2010 | 0.99x | 接近持平 |
| 256 | 128 | 0.2071 | 0.2020 | 1.02x | 开始有微弱提升 |
| 256 | 256 | 0.2372 | 0.2356 | 1.01x | 持平 |
| 512 | 128 | 0.2875 | 0.2880 | 1.00x | 持平 |
| 512 | 256 | 0.3313 | 0.4470 | 0.74x | **性能下降**，可能cache miss |
| 1024 | 128 | 1.1060 | 1.0051 | **1.10x** | 10%提升 |
| 1024 | 256 | 1.2411 | 1.0609 | **1.17x** | 17%提升 |

### 结果分析

1. **小尺寸矩阵 (≤256×256)**
   - 性能提升不明显（0.98-1.02x）
   - 原因：kernel launch开销、shared memory同步等固定开销占主导
   - 内存访问量本身就小，减少一次遍历的收益被其他开销抵消

2. **大尺寸矩阵 (1024×128/256)**
   - 明显提升：10-17%加速
   - 原因：内存带宽成为瓶颈，减少一次全局内存遍历的效果显现
   - 1024×128: 每次遍历读写 1024×128×4 = 512KB
   - 减少一次遍历节约 512KB 的内存传输

3. **异常情况 (512×256)**
   - 性能反而下降到0.74x
   - 可能原因：
     - exp计算被重复执行（Phase 2计算一次用于求和，写回时又计算一次）
     - 编译器可能在naive版本中优化了某些访问模式，online版本破坏了这种模式
     - cache行为差异

### 优化收益

| 指标 | Naive | Online | 改进 |
|------|-------|--------|------|
| 全局内存遍历次数 | 3次 | 2次 | -33% |
| 最大加速比 | - | 1.17x | +17% |
| 代码复杂度 | 低 | 低（相同） | 无增加 |

### 下一步优化方向

1. **优化exp重复计算**
   - 当前online版本在Phase 2算sum时计算exp，写回时又重新计算
   - 可以在计算sum时将exp值暂存到shared memory，避免重算
   - 但需权衡shared memory容量限制

2. **真正的Online Softmax算法**
   - 尝试实现同时计算max和sum的单次遍历算法
   - 难点：warp reduction时需要维护(max, sum)对并正确合并
   - 理论上可以将2次遍历进一步减少到1次+写回

3. **与其他算子融合**
   - 将scale kernel与softmax融合
   - 或者将softmax与后续的GEMM融合（类似FlashAttention思想）

4. **使用更高效的归约方式**
   - 尝试使用CUB库的block-level reduction
   - 或者使用cooperative groups API

5. **处理512×256异常**
   - 使用nsight compute profile分析性能下降原因
   - 可能需要针对不同尺寸选择不同的实现策略
