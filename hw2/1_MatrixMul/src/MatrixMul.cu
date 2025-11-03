#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 16

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

void initializeMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

// CPU matrix multiplication
void cpuMatrixMul(const float *A, const float *B, float *C, int M, int K, int N) {
    // A: M x K matrix
    // B: K x N matrix
    // C: M x N matrix
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
    float *h_C = (float *)malloc(size_C);
    float *h_C_cpu = (float *)malloc(size_C);  // For CPU result

    srand(time(NULL));
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // ============ CPU Matrix Multiplication ============
    printf("Matrix dimensions: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);
    printf("\n=== CPU Matrix Multiplication ===\n");
    
    clock_t cpu_start = clock();
    cpuMatrixMul(h_A, h_B, h_C_cpu, M, K, N);
    clock_t cpu_end = clock();
    
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU execution time: %.2f ms\n", cpu_time);

    // ============ GPU Matrix Multiplication ============
    printf("\n=== GPU Matrix Multiplication ===\n");
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tiledMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU execution time: %.2f ms\n", gpu_time);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // ============ Performance Comparison ============
    printf("\n=== Performance Comparison ===\n");
    printf("Speedup (CPU/GPU): %.2fx\n", cpu_time / gpu_time);
    printf("GPU is %.2f%% faster than CPU\n", (1 - gpu_time / cpu_time) * 100);

    // ============ Validate Results ============
    printf("\n=== Validating Results ===\n");
    bool results_match = true;
    int mismatch_count = 0;
    int max_mismatches_to_show = 5;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(h_C[i * N + j] - h_C_cpu[i * N + j]) > 1e-3) {
                if (mismatch_count < max_mismatches_to_show) {
                    fprintf(stderr, "Mismatch at element (%d, %d): CPU = %f, GPU = %f\n", 
                            i, j, h_C_cpu[i * N + j], h_C[i * N + j]);
                }
                mismatch_count++;
                results_match = false;
            }
        }
    }
    
    if (results_match) {
        printf("Results match! CPU and GPU produce identical results.\n");
    } else {
        fprintf(stderr, "Results mismatch! Total mismatches: %d\n", mismatch_count);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);

    printf("\nTile-based matrix multiplication completed successfully!\n");
    return 0;
}