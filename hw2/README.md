# CS4302 Homework 2

## 1. 项目结构

```
hw2/
├── Makefile                    # 构建脚本
├── README.md                   # 程序编译与性能分析文档
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
- `-arch=sm_50`: 指定目标 GPU 计算能力架构。

## 4. 运行说明

### (1) CUDA 矩阵乘法优化

**功能**:
比较 CPU、基础版 GPU Tiled 矩阵乘法和优化版 GPU Tiled 矩阵乘法的性能。

**运行方式**:

在根目录下运行

```bash
./build/MatrixMul.exe
```

程序会使用预设的矩阵维度（`M=1024, K=512, N=768`）进行计算，并输出性能对比结果。若想测试不同的矩阵维度，可直接修改`M, K, N`的值。

### (2) CUDA 并行数组求和

**功能**:
使用 CUDA 实现大规模数组的并行求和，并与 CPU 计算结果进行验证。

**运行方式**:

在根目录下运行

```bash
./build/ArraySum.exe
```

程序会处理一个包含 `8M` 个元素的数组，并输出 GPU 计算时间和与 CPU 结果的对比。

## 5. 性能优化分析

### (1) CUDA 矩阵乘法优化

运行结果如下：
```
Matrix dimensions: A(1024 x 512) * B(512 x 768) = C(1024 x 768)
=== CPU Matrix Multiplication ===
CPU execution time: 157.00 ms

=== GPU Matrix Multiplication (Basic Tiled) ===
GPU execution time (Basic): 11.30 ms
Speedup vs CPU: 13.89x
[Basic] Results match! GPU produces correct results.

=== GPU Matrix Multiplication (Optimized: Bank Conflict Avoidance + Loop Unrolling) ===
GPU execution time (Optimized): 0.23 ms
Speedup vs CPU: 684.18x
Speedup vs Basic: 49.25x (97.97% improvement)
[Optimized] Results match! GPU produces correct results.
```

1. **基础版 GPU 矩阵乘法**：
   - 使用 Tiled 矩阵乘法，将数据分块加载到共享内存中，减少了全局内存访问的次数。
   - 运行时间为 11.30 ms，相较于 CPU 的 157.00 ms 提升了约 13.89 倍。

2. **优化版 GPU 矩阵乘法**：
   - **避免 Bank Conflicts**：通过在共享内存中添加 padding，避免了线程访问共享内存时的冲突，提高了内存访问效率。
   - **循环展开 (Loop Unrolling)**：对计算循环进行了展开，减少了循环控制的开销。
   - 优化后运行时间为 0.23 ms，相较于基础版提升了 49.25 倍（97.97% 的性能提升），相较于 CPU 提升了 684.18 倍。

### (2) CUDA 并行数组求和

运行结果如下：
```
Array size: 8388608 elements (32.00 MB)

=== CPU Array Sum ===
CPU execution time: 4.0000 ms
CPU sum result: 4149918.000000

=== Test 1: Original Kernel (with atomicAdd) ===
Number of blocks: 32768
GPU execution time: 0.3849 ms
Speedup vs CPU: 10.39x
GPU sum result: 4150054.250000
Relative error: 3.283197e-05 - PASSED

=== Test 2: Optimized Kernel 1 (Bank Conflict Avoidance) ===
Number of blocks: 32768
GPU execution time: 0.1080 ms
Speedup vs CPU: 37.05x
Speedup vs Basic: 3.57x (71.95% improvement)
GPU sum result: 4150054.250000
Relative error: 3.283197e-05 - PASSED

=== Test 3: Optimized Kernel 2 (Warp Divergence Reduction) ===
Number of blocks: 32768
GPU execution time: 0.0812 ms
Speedup vs CPU: 49.29x
Speedup vs Basic: 4.74x (78.92% improvement)
GPU sum result: 4150054.250000
Relative error: 3.283197e-05 - PASSED

=== Test 4: Optimized Kernel 3 (2 Elements per Thread) ===
Number of blocks: 16384
GPU execution time: 0.0581 ms
Speedup vs CPU: 68.83x
Speedup vs Basic: 6.62x (84.90% improvement)
GPU sum result: 4150054.000000
Relative error: 3.277173e-05 - PASSED

=== Test 5: Optimized Kernel 4 (4 Elements per Thread + Fully Unrolled) ===
Number of blocks: 8192
GPU execution time: 0.0446 ms
Speedup vs CPU: 89.61x
Speedup vs Basic: 8.62x (88.40% improvement)
GPU sum result: 4150054.000000
Relative error: 3.277173e-05 - PASSED
```

1. **原始 Kernel**：
   - 使用 `atomicAdd` 进行归约操作，虽然保证了正确性，但由于线程竞争导致性能受限。
   - 运行时间为 0.3849 ms，相较于 CPU 提升了约 10.39 倍。

2. **优化 Kernel 1 (Bank Conflict Avoidance)**：
   - 通过在共享内存中添加 padding，避免了 Bank Conflicts。
   - 运行时间为 0.1080 ms，相较于原始 Kernel 提升了 3.57 倍（71.95% 的性能提升）。

3. **优化 Kernel 2 (Warp Divergence Reduction)**：
   - 使用顺序地址访问，减少了 Warp Divergence。
   - 运行时间为 0.0812 ms，相较于原始 Kernel 提升了 4.74 倍（78.92% 的性能提升）。

4. **优化 Kernel 3 (2 Elements per Thread)**：
   - 每个线程处理多个元素，减少了线程块的数量。
   - 运行时间为 0.0581 ms，相较于原始 Kernel 提升了 6.62 倍（84.90% 的性能提升）。

5. **优化 Kernel 4 (4 Elements per Thread + Fully Unrolled)**：
   - 每个线程处理更多元素，并对归约循环进行了完全展开。
   - 运行时间为 0.0446 ms，相较于原始 Kernel 提升了 8.62 倍（88.40% 的性能提升）。

6. **加速原因**：
   - **共享内存优化**：避免 Bank Conflicts 后，线程可以高效地并行访问共享内存。
   - **Warp 优化**：减少 Warp Divergence 后，线程执行更加一致。
   - **计算优化**：循环展开和多元素处理减少了线程间的同步和控制开销。
   - **线程块优化**：减少线程块数量后，GPU 资源利用率更高。
