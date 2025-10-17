#include "floyd.h"
#include "utils.h"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>

int main(int argc, char *argv[]) {
    int n = 1000;  // default graph size
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
    
    int **matrix_serial = allocate_matrix(n);
    int **matrix_parallel = allocate_matrix(n);
    
    generate_random_graph(matrix_serial, n, density);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_parallel[i][j] = matrix_serial[i][j];
        }
    }
    
    if (n <= 10) {
        std::cout << "\nOriginal matrix:\n";
        print_matrix(matrix_serial, n);
    }

    // test serial version
    std::cout << "\nRunning serial Floyd-Warshall...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    floyd_serial(matrix_serial, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration<double>(t1 - t0).count();

    // test parallel version
    std::cout << "\nRunning parallel Floyd-Warshall with " << num_threads << " threads...\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    floyd_parallel(matrix_parallel, n, num_threads);
    auto t3 = std::chrono::high_resolution_clock::now();
    double parallel_time = std::chrono::duration<double>(t3 - t2).count();

    std::cout << "Serial time: " << std::fixed << std::setprecision(6) 
              << serial_time << " seconds\n";  

    std::cout << "Parallel time: " << std::fixed << std::setprecision(6) 
              << parallel_time << " seconds\n";
    
    // verify whether results match
    bool results_match = true;
    for (int i = 0; i < n && results_match; i++) {
        for (int j = 0; j < n && results_match; j++) {
            if (matrix_serial[i][j] != matrix_parallel[i][j]) {
                results_match = false;
                std::cout << "Mismatch found at position (" << i << ", " << j << "): "
                          << "Serial = " << matrix_serial[i][j] 
                          << ", Parallel = " << matrix_parallel[i][j] << std::endl;
            }
        }
    }
    
    if (results_match) {
        std::cout << "Results match! Serial and parallel versions produce identical results.\n";
    } else {
        std::cout << "Results do not match! There are differences between serial and parallel versions.\n";
    }
    
    if (n <= 10) {
        std::cout << "\nFinal result matrix (serial):\n";
        print_matrix(matrix_serial, n);
        std::cout << "\nFinal result matrix (parallel):\n";
        print_matrix(matrix_parallel, n);
    }
    
    // compute speedup
    double speedup = serial_time / parallel_time;
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "Basic parallel speedup: " << std::fixed << std::setprecision(2) << speedup << "x";
    std::cout << "\n";
    
    free_matrix(matrix_serial, n);
    free_matrix(matrix_parallel, n);
    
    return 0;
}