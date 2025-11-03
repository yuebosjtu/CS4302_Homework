# CS4302 Homework 2

## 1. 项目结构

```
hw2/
├── Makefile                    # 构建脚本
├── README.md
├── 1_MatrixMul/                # CUDA 矩阵乘法优化
│   └── src/
│       └── MatrixMul.cu
└── 2_ArraySum/                 # CUDA 并行数组求和
    └── src/
        └── ArraySum.cu
```

## 2. 环境要求

- **编译器**: NVIDIA CUDA Compiler (`nvcc`)
- **构建工具**: `make`
- **硬件**: 支持 CUDA 的 NVIDIA GPU

### 环境配置

- **CUDA Toolkit**: 必须安装 NVIDIA CUDA Toolkit，它包含了 `nvcc` 编译器和相关库。
- **构建工具**:
    - **Windows**: 需要安装 `make` 工具，例如通过 `MinGW-w64` 或 `MSYS2` 获取。
    - **Linux/macOS**: 通常自带 `make`。
- **环境变量**: 确保 `nvcc` 和 `make` 已经添加到系统的 `PATH` 环境变量中。

## 3. 编译说明

### 编译所有项目

在 `hw2` 根目录下执行：

```bash
make
```

或者显式指定：

```bash
make all
```

这将编译两个项目，生成的可执行文件位于 `build/` 目录：
- `build/MatrixMul.exe`
- `build/ArraySum.exe`

### 清理编译结果

```bash
make clean
```

此命令会删除 `build/` 目录及其所有内容。

### 编译选项

Makefile 使用的编译选项：
- `-O2`: 开启 O2 级别优化。
- `-arch=sm_50`: 指定目标 GPU 计算能力架构。**请根据您的 GPU 型号修改此参数**（例如 `sm_60`, `sm_75`, `sm_86` 等）。

## 4. 运行说明

### (1) CUDA 矩阵乘法优化

**功能**:
比较 CPU、基础版 GPU Tiled 矩阵乘法和优化版 GPU Tiled 矩阵乘法的性能。

**运行方式**:

```bash
./build/MatrixMul.exe
```

程序会使用预设的矩阵维度（`M=1024, K=512, N=768`）进行计算，并输出性能对比结果。

### (2) CUDA 并行数组求和

**功能**:
使用 CUDA 实现大规模数组的并行求和，并与 CPU 计算结果进行验证。

**运行方式**:

```bash
./build/ArraySum.exe
```

程序会处理一个包含 `1024 * 1024` 个元素的数组，并输出 GPU 计算时间和与 CPU 结果的对比。

## 5. 代码分析

### (1) MatrixMul - 矩阵乘法优化

`MatrixMul.cu` 实现了三种矩阵乘法：
1.  **CPU 版本 (`cpuMatrixMul`)**: 经典的三重循环串行实现，作为性能基准和正确性验证。
2.  **基础 Tiled GPU 版本 (`tiledMatrixMul`)**:
    - **分块思想**: 将大矩阵划分为小的 `TILE_WIDTH x TILE_WIDTH` 的子矩阵（Tile）。
    - **共享内存**: 每个线程块将计算所需的 A 和 B 矩阵的 Tiles 加载到共享内存（`__shared__`），以减少对全局内存的访问延迟。
    - **协同计算**: 块内所有线程从共享内存中读取数据，协同计算出一个 C 矩阵的 Tile。
3.  **优化 GPU 版本 (`optimizedMatrixMul`)**: 在基础版上应用了两种关键优化：
    - **避免 Bank Conflict**:
        - **问题**: 当多个线程同时访问同一个共享内存 Bank 时会产生冲突，导致访问串行化，降低性能。
        - **策略**: 对共享内存数组的宽度进行填充（Padding），例如 `__shared__ float tileA[TILE_WIDTH][TILE_WIDTH + 1]`。这使得连续的列元素分布在不同的 Bank 中，从而避免了访问冲突。
    - **循环展开 (`#pragma unroll`)**:
        - **问题**: 循环控制（判断、跳转）会带来额外的指令开销。
        - **策略**: 编译器将循环体内的代码复制多次，减少循环迭代次数和分支开销，从而提高指令级并行度。

**性能预期**:
- GPU 版本会远快于 CPU 版本。
- 优化后的 GPU 版本会比基础 GPU 版本有显著的性能提升，因为 Bank Conflict 和循环开销是影响 GPU Kernel 性能的常见瓶颈。

### (2) ArraySum - 并行数组求和

`ArraySum.cu` 实现了一个基于**归约（Reduction）**思想的并行求和算法。

**算法流程**:
1.  **数据加载**: 每个线程负责从全局内存加载一个元素到共享内存 `partialSum` 中。
2.  **块内归约 (Tree-based Reduction)**:
    - 在线程块内部，使用共享内存进行树状归约。
    - 在每次迭代中，将活动线程数减半，每对线程中的一个将两个数相加，存入其中一个的位置。
    - 例如，`stride` 从 `blockDim.x / 2` 开始，每次减半。`threadIdx.x` 为 `i` 的线程将其 `partialSum[i]` 与 `partialSum[i + stride]` 相加。
    - 这个过程持续进行，直到 `stride` 变为 0。此时，`partialSum[0]` 中存储了整个线程块所有元素的和。
3.  **块间归约**:
    - 每个线程块的 0 号线程，将块内归约的结果（`partialSum[0]`）写回到全局内存的一个输出数组中。
    - **最终求和**: 由于 GPU Kernel 执行后得到的 `h_output` 是各个块的部分和，因此在 `main` 函数中，由 CPU 对这些部分和进行最终的求和，得到全局总和。

**性能分析**:
- 该算法利用共享内存的高速特性，显著减少了对全局内存的原子操作或多次读写，是 CUDA 中实现并行归约的标准高效模式。
- 当数组非常大时，可以进一步优化，通过第二个 Kernel 对第一步产生的部分和在 GPU 上再次进行归约，而不是在 CPU 上完成，以避免数据从 GPU 拷回 CPU 的开销。
