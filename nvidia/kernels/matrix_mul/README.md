# CUDA矩阵乘法性能对比与参数调优

本项目对比了naive矩阵乘法实现与高度优化的CUDA kernel实现，并针对NVIDIA A100 GPU进行了参数调优。

## 项目结构

```
matrix_mul/
├── naive_kernel.cu                      # Naive实现
├── optimized_kernel.cu                  # 优化实现
├── test_comparison.py                   # 基础性能对比测试
├── tune_parameters.py                   # 参数调优工具
├── test_bk_large.py                     # BK参数大矩阵测试
├── visualize_bk_comprehensive.py        # BK参数可视化分析
├── tuning_results.csv                   # 参数调优结果
├── bk_tuning_large_matrices.csv         # BK参数测试结果
├── matrix_mul_comparison.png            # 基础对比图表
├── bk_comprehensive_analysis.png        # BK参数综合分析图
└── README.md                            # 本文档
```

## 实现方法

### 1. Naive Kernel (`naive_kernel.cu`)

**特点：**
- 每个线程计算输出矩阵的一个元素
- 直接从全局内存读取数据
- 无共享内存优化
- 无向量化加载

**实现：**
```cuda
__global__ void matrix_mul_naive_kernel(const float* A, const float* B, float* C,
                                         int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### 2. Optimized Kernel (`optimized_kernel.cu`)

**优化策略：**

#### 内存优化
- **共享内存分块**：使用shared memory进行数据重用
- **float4向量化**：使用float4进行向量化加载/存储
- **矩阵转置**：矩阵A转置存储避免bank conflict
- **双缓冲技术**：隐藏全局内存访问延迟

#### 计算优化
- **多层级分块**：Block → Warp → Sub-Warp → Thread
- **寄存器分块**：减少共享内存访问
- **循环展开**：提高指令级并行
- **合并内存访问**：优化内存访问模式

#### 最优配置参数（针对A100 GPU）

```
Block Tiling:    BM=128, BK=16, BN=128
Warp Tiling:     WM=64,  WN=64
Sub-Warp Tiling: WSUBM=64, WSUBN=16
Thread Tiling:   TM=8,   TN=4
```

**参数说明：**
- `BM/BN/BK`: Block级别的分块大小（M/N/K维度）
- `WM/WN`: Warp级别的分块大小
- `WSUBM/WSUBN`: Sub-Warp级别的分块大小
- `TM/TN`: 每个线程处理的元素数量

**约束条件：**
- `(WSUBM/TM) × (WSUBN/TN) = 32` (warp size)
- `BM % WM = 0`, `BN % WN = 0`
- `WM % WSUBM = 0`, `WN % WSUBN = 0`
- `TN % 4 = 0` (float4向量化)
- Shared memory < 48KB

## 性能测试结果

### 基础对比（Naive vs Optimized）

测试矩阵尺寸：512×512×512 到 4096×4096×4096

| 矩阵尺寸        | Naive (ms) | Optimized (ms) | 加速比 |
|----------------|------------|----------------|--------|
| 512×512×512    | 0.180      | 0.236          | 0.76x  |
| 1024×1024×1024 | 1.230      | 0.463          | 2.66x  |
| 2048×2048×2048 | 9.689      | 1.753          | 5.53x  |
| 4096×4096×4096 | 74.091     | 12.114         | 6.12x  |

**观察：**
- 小矩阵（512）：优化版本反而慢，因为启动开销
- 中等矩阵（1024-2048）：加速比2.66x - 5.53x
- 大矩阵（4096）：加速比达到6.12x

### BK参数调优结果

测试了BK=8, 16, 32在不同矩阵尺寸下的性能表现。

#### 测试结果汇总

| 矩阵尺寸          | BK=8 (GFLOPS) | BK=16 (GFLOPS) | BK=32 | 最优BK | 性能提升 |
|------------------|---------------|----------------|-------|--------|----------|
| 2048×2048×2048   | 8934.03       | 8947.86        | ✗     | 16     | +0.2%    |
| 4096×4096×4096   | 10335.65      | 11096.87       | ✗     | 16     | +7.4%    |
| 8192×8192×8192   | 11953.56      | 12091.44       | ✗     | 16     | +1.2%    |
| 16384×16384×16384| 12066.95      | 12260.50       | ✗     | 16     | +1.6%    |

**注：** BK=32因超出shared memory限制（66.6KB > 48KB）无法使用

#### 关键发现

1. **BK=16是最优选择**
   - 在所有测试尺寸上都表现最佳
   - 相比BK=8提升0.2% - 7.4%
   - Shared memory使用：32.5 KB（远低于48KB限制）

2. **矩阵尺寸影响**
   - 4096×4096×4096时BK影响最大（+7.4%）
   - 超大矩阵（8192+）时影响较小（1-2%）
   - 小矩阵（2048）时影响最小（0.2%）

3. **BK参数权衡**
   - **更大的BK**：更好的数据重用，但需要更多shared memory
   - **更小的BK**：更少的shared memory，但K维度循环次数更多
   - **BK=16**：在两者之间取得最佳平衡

4. **Shared Memory使用**
   - BK=8:  16.2 KB
   - BK=16: 32.5 KB
   - BK=32: 66.6 KB（超限）

## 使用方法

### 环境要求

```bash
# Python依赖
pip install numpy cupy-cuda12x matplotlib pandas

# CUDA环境
CUDA Toolkit 12.x
NVIDIA GPU (推荐A100)
```

### 运行基础对比测试

```bash
python test_comparison.py
```

输出：
- 控制台显示各尺寸性能对比
- 生成 `matrix_mul_comparison.png` 图表

### 运行参数调优

```bash
python tune_parameters.py
```

输出：
- 测试多种参数配置
- 生成 `tuning_results.csv` 结果文件
- 显示最优配置

### 运行BK参数测试

```bash
python test_bk_large.py
```

输出：
- 测试不同BK值在大矩阵上的表现
- 生成 `bk_tuning_large_matrices.csv`

### 生成可视化分析

```bash
python visualize_bk_comprehensive.py
```

输出：
- 生成 `bk_comprehensive_analysis.png` 综合分析图
- 包含6个子图的详细分析

## 性能分析

### 为什么优化版本更快？

1. **内存访问优化**
   - Naive: 每次计算都从全局内存读取（高延迟）
   - Optimized: 数据加载到shared memory后重复使用（低延迟）

2. **向量化加载**
   - Naive: 单个float加载
   - Optimized: float4向量化加载（4倍带宽）

3. **Bank Conflict避免**
   - 矩阵A转置存储，避免shared memory bank conflict

4. **双缓冲**
   - 计算和数据加载并行，隐藏内存延迟

5. **寄存器分块**
   - 减少shared memory访问次数
   - 提高数据局部性

### 为什么BK=16最优？

1. **Shared Memory平衡**
   - 32.5 KB使用量，远低于48KB限制
   - 留有足够余量给其他资源

2. **K维度循环次数**
   - BK=8: K/8次循环
   - BK=16: K/16次循环（减少一半）
   - 更少的循环开销

3. **数据重用效率**
   - 每次加载的数据块足够大，重用效率高
   - 不会因为块太大导致寄存器压力

4. **硬件适配**
   - 与A100的cache line大小匹配良好
   - 充分利用L1/L2 cache

## 最佳实践建议

### 针对A100 GPU

1. **使用推荐配置**
   ```
   BM=128, BK=16, BN=128
   WM=64, WN=64
   WSUBM=64, WSUBN=16
   TM=8, TN=4
   ```

2. **矩阵尺寸要求**
   - 最好是128的倍数（对齐block size）
   - 如果不是，使用fallback kernel处理边界

3. **性能预期**
   - 4096×4096×4096: ~12 TFLOPS
   - 8192×8192×8192: ~12 TFLOPS
   - 16384×16384×16384: ~12.3 TFLOPS

### 针对其他GPU

如果使用其他GPU，建议：

1. **运行参数调优**
   ```bash
   python tune_parameters.py
   ```

2. **调整搜索空间**
   - 根据GPU的shared memory大小调整BK范围
   - 根据warp size调整thread tiling

3. **验证正确性**
   - 所有配置都会自动验证正确性
   - 使用CuBLAS作为参考

## 技术细节

### 线程映射

每个block包含 `(BM/WM) × (BN/WN)` 个warp，每个warp 32个线程。

线程到输出元素的映射：
```
warp_id = threadIdx.x / 32
wy = warp_id / (BN/WN)
wx = warp_id % (BN/WN)

thread_id = threadIdx.x % 32
ty = thread_id / (WSUBN/TN)
tx = thread_id % (WSUBN/TN)

output[wy*WM + ty*TM : wy*WM + (ty+1)*TM,
       wx*WN + tx*TN : wx*WN + (tx+1)*TN]
```
### sub-warp实现形式

每个sub-warp也是有32个线程 `(WSUBM/TM) × (WSUBM/TN) = 32`，一个线程处理 `TM * TN` 个元素。
sub-warp之间是分时处理，从而实现shared memory -> register 和 实际运算的同时进行；
与输入矩阵的K维被拆分为若干个block分别计算BK维度分时计算是一个逻辑。

**BK 的循环**：
- 目标：减少 Global Memory 访问。
- 手段：把 K 维切开，把一小块 $A$ 和 $B$ 搬进 Shared Memory，在这个 BK 范围内，Shared Memory 里的数据被整个 Thread Block 反复压榨。

**Sub-Warp 的循环**：
- 目标：减少 Shared Memory 访问。
- 手段：在 Warp Tile 内部切开，把一小块数据搬进寄存器，在这个 Sub-Warp 范围内，寄存器里的数据被 Warp 内的 32 个线程 反复压榨。



### 双缓冲实现

```cuda
__shared__ float As[2][kAsSize];
__shared__ float Bs[2][kBsSize];

int buffer_idx = 0;

// 加载第一个tile
load_tile(As[0], Bs[0]);
__syncthreads();

for (int k = BK; k < K; k += BK) {
    // 异步加载下一个tile到buffer[1-buffer_idx]
    load_tile(As[1-buffer_idx], Bs[1-buffer_idx]);

    // 计算当前tile (buffer[buffer_idx])
    compute(As[buffer_idx], Bs[buffer_idx]);

    __syncthreads();
    buffer_idx = 1 - buffer_idx;
}

// 计算最后一个tile
compute(As[buffer_idx], Bs[buffer_idx]);
```

### 向量化加载示例

```cuda
// 加载4个连续的float
float4 data = CFLOAT4(A[offset]);

// 等价于
// float data.x = A[offset];
// float data.y = A[offset+1];
// float data.z = A[offset+2];
// float data.w = A[offset+3];
```

## 性能对比

### vs cuBLAS

在A100上，cuBLAS的SGEMM性能约为 **19.5 TFLOPS**（理论峰值的~80%）。

我们的优化kernel达到 **12.3 TFLOPS**，约为cuBLAS的 **63%**。

差距主要来自：
1. cuBLAS使用Tensor Cores（我们使用CUDA Cores）
2. cuBLAS有更复杂的分块策略
3. cuBLAS针对各种尺寸都有专门优化

### vs Naive实现

在4096×4096×4096上：
- Naive: 1.85 TFLOPS
- Optimized: 11.1 TFLOPS
- **加速比: 6.0x**

## 参考资料

1. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. [CUTLASS: Fast Linear Algebra in CUDA C++](https://github.com/NVIDIA/cutlass)
3. [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
4. [NVIDIA A100 Tensor Core GPU Architecture](https://www.nvidia.com/en-us/data-center/a100/)

## 许可证

本项目仅供学习和研究使用。

## 作者

CUDA矩阵乘法优化项目

## 更新日志

- 2024-03: 初始版本
  - 实现naive和optimized kernel
  - 参数调优工具
  - BK参数分析
  - 综合性能测试
