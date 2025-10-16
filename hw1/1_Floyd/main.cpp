#include "floyd.h"
#include "utils.h"
#include <omp.h>
#include <iostream>
#include <iomanip>

int main(int argc, char *argv[]) {
    int n = 100;  // default graph size
    double density = 0.3;
    int num_threads = 4;
    
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        density = atof(argv[2]);
    }
    if (argc > 3) {
        num_threads = atoi(argv[3]);
    }
    
    std::cout << "Floyd-Warshall Algorithm Performance Comparison\n";
    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Graph density: " << density << std::endl;
    std::cout << "Number of threads: " << num_threads << std::endl;
    std::cout << "----------------------------------------\n";
    
    // 分配并初始化矩阵
    int **matrix_serial = allocate_matrix(n);
    int **matrix_parallel = allocate_matrix(n);
    
    // 生成随机图
    generate_random_graph(matrix_serial, n, density);
    
    // 复制矩阵用于不同版本
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_parallel[i][j] = matrix_serial[i][j];
        }
    }
    
    if (n <= 10) {
        std::cout << "\nOriginal matrix:\n";
        print_matrix(matrix_serial, n);
    }

    // 并行版本测试
    std::cout << "\nRunning parallel Floyd-Warshall with " << num_threads << " threads...\n";
    double start_time = get_time();
    floyd_parallel(matrix_parallel, n, num_threads);
    double parallel_time = get_time() - start_time;
    
    // test serial version
    std::cout << "\nRunning serial Floyd-Warshall...\n";
    start_time = get_time();
    floyd_serial(matrix_serial, n);
    double serial_time = get_time() - start_time;
    
    std::cout << "Serial time: " << std::fixed << std::setprecision(6) 
              << serial_time << " seconds\n";  

    std::cout << "Parallel time: " << std::fixed << std::setprecision(6) 
              << parallel_time << " seconds\n";
    
    // 计算加速比
    double speedup = serial_time / parallel_time;
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "Basic parallel speedup: " << std::fixed << std::setprecision(2) << speedup << "x";
    
    // 清理内存
    free_matrix(matrix_serial, n);
    free_matrix(matrix_parallel, n);
    
    return 0;
}