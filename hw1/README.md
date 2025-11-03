# CS4302 Homework 1

## 1. 项目结构

```
hw1/
├── Makefile                    # 构建脚本
├── README.md                   # 本文档
├── 1_Floyd/                    # Floyd-Warshall 最短路径算法
│   ├── main.cpp
│   ├── include/
│   │   ├── floyd.h
│   │   └── utils.h
│   └── src/
│       └── floyd.cpp
├── 2_MonteCarlo/              # Monte Carlo 方法估算圆周率
│   ├── main.cpp
│   ├── include/
│   │   └── MonteCarlo.h
│   └── src/
│       └── MonteCarlo.cpp
└── 3_Convolution/             # 2D 卷积运算
    ├── main.cpp
    ├── include/
    │   └── Convolution.h
    └── src/
        └── Convolution.cpp
```

## 2. 环境要求

- **编译器**: g++ (支持 C++11 及以上)
- **构建工具**: make
- **并行库**: OpenMP

### Windows 环境配置

如果在 Windows 上使用，需要安装：
- MinGW-w64 或 MSYS2 (提供 g++ 和 make)
- 确保 `g++` 和 `make` 已添加到系统 PATH
- 也可以使用MinGW-w32，编译时使用```mingw32-make```代替```make```即可

## 3. 编译说明

### 编译所有项目

在根目录下执行：

```bash
make
```

或者显式指定：

```bash
make all
```

这将编译三个项目，生成的可执行文件位于 `build/` 目录：
- `build/floyd.exe`
- `build/montecarlo.exe`
- `build/convolution.exe`

### 编译单个项目

```bash
make floyd          # 只编译 Floyd-Warshall 项目
make montecarlo     # 只编译 Monte Carlo 项目
make convolution    # 只编译 Convolution 项目
```

### 清理编译结果

```bash
make clean
```

### 编译选项

Makefile 使用的编译选项：
- `-std=c++11`: 使用 C++11 标准
- `-O2`: 开启 O2 级别优化
- `-Wall`: 显示所有警告
- `-fopenmp`: 启用 OpenMP 支持

## 4. 运行说明

### (1) Floyd-Warshall 最短路径算法

**功能**: 比较串行和并行版本的 Floyd-Warshall 算法性能。

**运行方式**:

```bash
./build/floyd.exe [n] [density] [num_threads]
```

**参数说明**:
- `n`: 图的节点数量（默认: 1000）
- `density`: 图的边密度，范围 0.0-1.0（默认: 0.3）
- `num_threads`: OpenMP 线程数（默认: 4）

**示例**:

```bash
# 使用默认参数
./build/floyd.exe

# 自定义参数：2000个节点，密度0.5，4线程
./build/floyd.exe 2000 0.5 4
```

### (2) Monte Carlo 方法估算圆周率

**功能**: 使用 Monte Carlo 方法估算圆周率$π$，比较串行和并行性能。

**运行方式**:

```bash
./build/montecarlo.exe [num_samples] [num_threads]
```

**参数说明**:
- `num_samples`: 采样点数量（默认: 10000000）
- `num_threads`: OpenMP 线程数（默认: 系统最大线程数）

**示例**:

```bash
# 使用默认参数
./build/montecarlo.exe

# 自定义参数：1000000个采样点，4线程
./build/montecarlo.exe 1000000 4
```

### (3) 2D 卷积运算

**功能**: 比较串行和并行版本的 2D 卷积运算性能。

**运行方式**:

```bash
./build/convolution.exe [M] [N] [K] [num_threads]
```

**参数说明**:
- `M`: 输入矩阵行数（默认: 256）
- `N`: 输入矩阵列数（默认: 256）
- `K`: 卷积核大小 (K×K)（默认: 3）
- `num_threads`: OpenMP 线程数（默认: 4）

**示例**:

```bash
# 使用默认参数
./build/convolution.exe

# 自定义参数：1000×1000矩阵，3×3卷积核，4线程
./build/convolution.exe 1000 1000 3 4
```

## 5. 性能分析

### (1) Floyd-Warshall 最短路径算法
默认参数下输出结果如下：
```
Floyd-Warshall Algorithm Performance Comparison
Matrix size: 1000x1000
Graph density: 0.3
Number of threads: 4
----------------------------------------

Running serial Floyd-Warshall...

Running parallel Floyd-Warshall with 4 threads...
Serial time: 0.415813 seconds
Parallel time: 0.171774 seconds
Results match! Serial and parallel versions produce identical results.

Performance Results:
Basic parallel speedup: 2.42x
```

可以看到串行与并行程序的结果一致，并行相较于串行加速了2.42倍。
当$n=200$（矩阵大小为$200\times200$）时，串行程序执行时间是并行的0.68倍。粗略估计当$n\geq300$时才会实现加速（$n=300$时加速了1.14倍）。
这说明只有当数据量达到一定大小的时候，openmp并行化才会实现加速，可能是因为并行程序分配任务的时间长，效率不如串行直接运行；另外，当数据量大于这个阈值时，数据量越大，加速比就越大。


### (2) Monte Carlo 方法估算圆周率
默认参数下输出结果如下：
```
Monte Carlo pi estimation
samples: 10000000, threads: 32

Serial: pi = 3.14232, time = 0.110876 s
Parallel: pi = 3.14141, time = 0.01093 s
Speedup (serial / parallel) = 10.1442
```
可以看到串行与并行程序的结果一致，并行程序的结果比串行程序的结果更接近真实值，且并行相较于串行加速了10.1442倍。
类似地，当采样点数为100000（比默认值少10倍）的时候，加速比为4.4708，仍然支持“数据量越大，加速比就越大”的结论。

### (3) 2D 卷积运算
默认参数下输出结果如下：
```
2D Convolution Performance Comparison
Input matrix size: 5000x5000
Filter size: 20x20
Output matrix size: 4981x4981
Number of threads: 32
----------------------------------------

Running serial convolution...
Running parallel convolution with 32 threads...

Performance Results:
Serial time: 3.200503 seconds
Parallel time: 0.969395 seconds
Results match! Serial and parallel versions produce identical results.
Speedup: 3.30x
```
可以看到串行与并行程序的结果一致，并行相较于串行加速了3.30倍。