#include "Convolution.h"
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <cmath>

// Allocate 2D matrix
double** allocate_matrix(int rows, int cols) {
    double** matrix = new double*[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new double[cols];
    }
    return matrix;
}

// Free 2D matrix
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Initialize matrix with random values
void initialize_random_matrix(double** matrix, int rows, int cols, double min_val, double max_val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = min_val + (max_val - min_val) * ((double)rand() / RAND_MAX);
        }
    }
}

// Print matrix
void print_matrix(double** matrix, int rows, int cols, const char* name) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << std::setprecision(2) << std::fixed << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Check if two matrices are equal within tolerance
bool matrices_equal(double** a, double** b, int rows, int cols, double tolerance) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (std::abs(a[i][j] - b[i][j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// Serial 2D convolution implementation
void convolution_serial(double** input, int M, int N, double** filter, int K, double** output) {
    int output_rows = M - K + 1;
    int output_cols = N - K + 1;
    
    // For each position in the output matrix
    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            double sum = 0.0;
            
            // Apply filter (dot product)
            for (int fi = 0; fi < K; fi++) {
                for (int fj = 0; fj < K; fj++) {
                    sum += input[i + fi][j + fj] * filter[fi][fj];
                }
            }
            
            output[i][j] = sum;
        }
    }
}

// Parallel 2D convolution implementation using OpenMP with reduction
void convolution_parallel(double** input, int M, int N, double** filter, int K, double** output, int num_threads) {
    int output_rows = M - K + 1;
    int output_cols = N - K + 1;
    
    omp_set_num_threads(num_threads);
    
    // Parallelize the outer loops
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            double sum = 0.0;
            
            // Use reduction for the dot product calculation
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K * K; k++) {
                int fi = k / K;
                int fj = k % K;
                sum += input[i + fi][j + fj] * filter[fi][fj];
            }
            
            output[i][j] = sum;
        }
    }
}