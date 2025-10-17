#include "Convolution.h"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <ctime>

int main(int argc, char *argv[]) {
    int M = 256;  // default input matrix rows
    int N = 256;  // default input matrix cols
    int K = 3;    // default filter size (K x K)
    int num_threads = 4;  // default number of threads
    
    if (argc > 1) {
        M = atoi(argv[1]);
    }
    if (argc > 2) {
        N = atoi(argv[2]);
    }
    if (argc > 3) {
        K = atoi(argv[3]);
    }
    if (argc > 4) {
        num_threads = atoi(argv[4]);
    }
    
    std::cout << "2D Convolution Performance Comparison\n";
    std::cout << "Input matrix size: " << M << "x" << N << std::endl;
    std::cout << "Filter size: " << K << "x" << K << std::endl;
    std::cout << "Output matrix size: " << (M-K+1) << "x" << (N-K+1) << std::endl;
    std::cout << "Number of threads: " << num_threads << std::endl;
    std::cout << "----------------------------------------\n";
    
    // Seed random number generator
    srand(static_cast<unsigned>(time(0)));
    
    // Allocate matrices
    double** input = allocate_matrix(M, N);
    double** filter = allocate_matrix(K, K);
    double** output_serial = allocate_matrix(M - K + 1, N - K + 1);
    double** output_parallel = allocate_matrix(M - K + 1, N - K + 1);
    
    // Initialize input matrix with random values
    initialize_random_matrix(input, M, N, 0.0, 10.0);
    
    // Initialize filter with random values
    initialize_random_matrix(filter, K, K, -1.0, 1.0);
    
    // Print small matrices for verification
    if (M <= 8 && N <= 8) {
        print_matrix(input, M, N, "Input matrix");
        print_matrix(filter, K, K, "Filter matrix");
    }
    
    // Run serial version
    std::cout << "\nRunning serial convolution...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    convolution_serial(input, M, N, filter, K, output_serial);
    auto t1 = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration<double>(t1 - t0).count();
    
    // Run parallel version
    std::cout << "Running parallel convolution with " << num_threads << " threads...\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    convolution_parallel(input, M, N, filter, K, output_parallel, num_threads);
    auto t3 = std::chrono::high_resolution_clock::now();
    double parallel_time = std::chrono::duration<double>(t3 - t2).count();
    
    // Display timing results
    std::cout << "\nPerformance Results:\n";
    std::cout << "Serial time: " << std::fixed << std::setprecision(6) 
              << serial_time << " seconds\n";
    std::cout << "Parallel time: " << std::fixed << std::setprecision(6) 
              << parallel_time << " seconds\n";
    
    // Verify results
    bool results_match = matrices_equal(output_serial, output_parallel, 
                                      M - K + 1, N - K + 1, 1e-10);
    
    if (results_match) {
        std::cout << "Results match! Serial and parallel versions produce identical results.\n";
    } else {
        std::cout << "Results do not match! There are differences between serial and parallel versions.\n";
    }
    
    // Print small output matrices for verification
    if (M <= 8 && N <= 8) {
        print_matrix(output_serial, M - K + 1, N - K + 1, "Output matrix (serial)");
        print_matrix(output_parallel, M - K + 1, N - K + 1, "Output matrix (parallel)");
    }
    
    // Calculate and display speedup
    double speedup = serial_time / parallel_time;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    
    // Calculate throughput
    long long operations = static_cast<long long>(M - K + 1) * (N - K + 1) * K * K;
    std::cout << "Total operations: " << operations << std::endl;
    std::cout << "Serial throughput: " << std::scientific << std::setprecision(2) 
              << operations / serial_time << " ops/sec\n";
    std::cout << "Parallel throughput: " << std::scientific << std::setprecision(2) 
              << operations / parallel_time << " ops/sec\n";
    
    // Free memory
    free_matrix(input, M);
    free_matrix(filter, K);
    free_matrix(output_serial, M - K + 1);
    free_matrix(output_parallel, M - K + 1);
    
    return 0;
}