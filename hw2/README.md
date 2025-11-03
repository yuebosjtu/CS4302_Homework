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
- `-arch=sm_50`: 指定目标 GPU 计算能力架构。

## 4. 运行说明

### (1) CUDA 矩阵乘法优化

**功能**:
比较 CPU、基础版 GPU Tiled 矩阵乘法和优化版 GPU Tiled 矩阵乘法的性能。

**运行方式**:

```bash
./build/MatrixMul.exe
```

程序会使用预设的矩阵维度（`M=1024, K=512, N=768`）进行计算，并输出性能对比结果。若想测试不同的矩阵维度，可直接修改`M, K, N`的值。

### (2) CUDA 并行数组求和

**功能**:
使用 CUDA 实现大规模数组的并行求和，并与 CPU 计算结果进行验证。

**运行方式**:

```bash
./build/ArraySum.exe
```

程序会处理一个包含 `8M` 个元素的数组，并输出 GPU 计算时间和与 CPU 结果的对比。

## 5. 性能优化分析

### (1) CUDA 矩阵乘法优化

### (2) CUDA 并行数组求和
