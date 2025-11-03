#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 16
// Add padding to avoid bank conflicts (assuming 32 banks)
#define TILE_WIDTH_PADDED (TILE_WIDTH + 1)

// Original tiled matrix multiplication
__global__ void tiledMatrixMul(const float *A, const float *B, float *C, int M, int K, int N) {
    // A: M x K matrix
    // B: K x N matrix
    // C: M x N matrix
    
    // Allocate shared memory for tiles of A and B
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Calculate row and column indices of the element of C computed by this thread
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0.0f;

    // Loop over all tiles needed to compute C[row][col]
    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        // Load tile from matrix A into shared memory if within bounds
        if (row < M && (m * TILE_WIDTH + threadIdx.x) < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + m * TILE_WIDTH + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from matrix B into shared memory if within bounds
        if (col < N && (m * TILE_WIDTH + threadIdx.y) < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform multiplication for the tile and accumulate the result
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to matrix C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = Pvalue;
    }
}

// Optimized version with bank conflict avoidance and loop unrolling
__global__ void optimizedMatrixMul(const float *A, const float *B, float *C, int M, int K, int N) {
    // A: M x K matrix
    // B: K x N matrix
    // C: M x N matrix
    
    // Allocate padded shared memory to avoid bank conflicts
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH_PADDED];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH_PADDED];

    // Calculate row and column indices of the element of C computed by this thread
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0.0f;

    // Loop over all tiles needed to compute C[row][col]
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int m = 0; m < numTiles; m++) {
        // Load tile from matrix A into shared memory with padding
        if (row < M && (m * TILE_WIDTH + threadIdx.x) < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + m * TILE_WIDTH + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from matrix B into shared memory with padding
        if (col < N && (m * TILE_WIDTH + threadIdx.y) < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform multiplication for the tile with loop unrolling
        // Unroll the loop by factor of 4 with safe boundary handling
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH / 4 * 4; k += 4) {
            Pvalue += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            Pvalue += tileA[threadIdx.y][k + 1] * tileB[k + 1][threadIdx.x];
            Pvalue += tileA[threadIdx.y][k + 2] * tileB[k + 2][threadIdx.x];
            Pvalue += tileA[threadIdx.y][k + 3] * tileB[k + 3][threadIdx.x];
        }
        // Handle remaining elements if TILE_WIDTH is not divisible by 4
        #pragma unroll
        for (int k = TILE_WIDTH / 4 * 4; k < TILE_WIDTH; k++) {
            Pvalue += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to matrix C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = Pvalue;
    }
}

void initializeMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

// CPU matrix multiplication for validation
void cpuMatrixMul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool validateResults(const float *gpu_result, const float *cpu_result, int M, int N, const char *kernel_name) {
    bool results_match = true;
    int mismatch_count = 0;
    int max_mismatches_to_show = 5;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(gpu_result[i * N + j] - cpu_result[i * N + j]) > 1e-3) {
                if (mismatch_count < max_mismatches_to_show) {
                    fprintf(stderr, "[%s] Mismatch at element (%d, %d): CPU = %f, GPU = %f\n", 
                            kernel_name, i, j, cpu_result[i * N + j], gpu_result[i * N + j]);
                }
                mismatch_count++;
                results_match = false;
            }
        }
    }
    
    if (results_match) {
        printf("[%s] Results match! GPU produces correct results.\n", kernel_name);
    } else {
        fprintf(stderr, "[%s] Results mismatch! Total mismatches: %d\n", kernel_name, mismatch_count);
    }
    
    return results_match;
}

int main() {
    // Matrix dimensions: A(M x K) * B(K x N) = C(M x N)
    int M = 1024;  // Rows of A and C
    int K = 512;   // Cols of A and Rows of B
    int N = 768;   // Cols of B and C
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C_cpu = (float *)malloc(size_C);
    float *h_C_basic = (float *)malloc(size_C);
    float *h_C_opt1 = (float *)malloc(size_C);

    srand(time(NULL));
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    printf("Matrix dimensions: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);

    // CPU Matrix Multiplication
    printf("=== CPU Matrix Multiplication ===\n");
    clock_t cpu_start = clock();
    cpuMatrixMul(h_A, h_B, h_C_cpu, M, K, N);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU execution time: %.2f ms\n\n", cpu_time);

    // Setup grid and block dimensions
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_time = 0;

    // Test 1: Basic tiled matrix multiplication
    printf("=== GPU Matrix Multiplication (Basic Tiled) ===\n");
    cudaMemset(d_C, 0, size_C);
    cudaEventRecord(start);
    tiledMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU execution time (Basic): %.2f ms\n", gpu_time);
    printf("Speedup vs CPU: %.2fx\n", cpu_time / gpu_time);
    cudaMemcpy(h_C_basic, d_C, size_C, cudaMemcpyDeviceToHost);
    validateResults(h_C_basic, h_C_cpu, M, N, "Basic");
    float basic_time = gpu_time;
    printf("\n");

    // Test 2: Optimized with bank conflict avoidance and loop unrolling
    printf("=== GPU Matrix Multiplication (Optimized: Bank Conflict Avoidance + Loop Unrolling) ===\n");
    cudaMemset(d_C, 0, size_C);
    cudaEventRecord(start);
    optimizedMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU execution time (Optimized): %.2f ms\n", gpu_time);
    printf("Speedup vs CPU: %.2fx\n", cpu_time / gpu_time);
    printf("Speedup vs Basic: %.2fx (%.2f%% improvement)\n", 
           basic_time / gpu_time, (1 - gpu_time / basic_time) * 100);
    cudaMemcpy(h_C_opt1, d_C, size_C, cudaMemcpyDeviceToHost);
    validateResults(h_C_opt1, h_C_cpu, M, N, "Optimized");
    printf("\n");

    // Performance Summary
    printf("==========================================================\n");
    printf("=== Performance Summary ===\n");
    printf("==========================================================\n");
    printf("CPU time:         %.2f ms\n", cpu_time);
    printf("GPU Basic time:   %.2f ms (%.2fx speedup vs CPU)\n", basic_time, cpu_time / basic_time);
    printf("GPU Optimized: %.2f ms (%.2fx speedup vs CPU, %.2fx vs Basic)\n", 
           gpu_time, cpu_time / gpu_time, basic_time / gpu_time);
    printf("==========================================================\n");
    printf("\nOptimizations applied:\n");
    printf("1. Bank Conflict Avoidance: Added padding to shared memory arrays\n");
    printf("2. Loop Unrolling: Unrolled inner loop by factor of 4\n");
    printf("==========================================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_basic);
    free(h_C_opt1);
    
    return 0;
}
