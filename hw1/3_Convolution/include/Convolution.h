#ifndef CONVOLUTION_H
#define CONVOLUTION_H

// 2D convolution - serial version
void convolution_serial(double** input, int M, int N, double** filter, int K, double** output);

// 2D convolution - parallel version using OpenMP with reduction
void convolution_parallel(double** input, int M, int N, double** filter, int K, double** output, int num_threads);

// Utility functions
double** allocate_matrix(int rows, int cols);

void free_matrix(double** matrix, int rows);

void initialize_random_matrix(double** matrix, int rows, int cols, double min_val, double max_val);

void print_matrix(double** matrix, int rows, int cols, const char* name);

bool matrices_equal(double** a, double** b, int rows, int cols, double tolerance);

#endif // CONVOLUTION_H