#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256

// GPU kernel: Reduction-based array sum
__global__ void arraySumReduction(const float *input, float *output, int n) {
    // Shared memory for partial sums within a block
    __shared__ float partialSum[BLOCK_SIZE];
    
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localTid = threadIdx.x;
    
    // Load data from global memory to shared memory
    // Each thread loads one element (or 0 if out of bounds)
    if (tid < n) {
        partialSum[localTid] = input[tid];
    } else {
        partialSum[localTid] = 0.0f;
    }
    __syncthreads();
    
    // Perform tree-based reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localTid < stride) {
            partialSum[localTid] += partialSum[localTid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the block's result to global memory
    if (localTid == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

// Initialize array with random values
void initializeArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 100) / 100.0f;
    }
}

// CPU-based array sum for verification
float cpuArraySum(const float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    // Array size
    int N = 1024 * 1024;  // 1M elements
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float *)malloc(size);
    float *h_output = NULL;
    if (!h_input) {
        fprintf(stderr, "Host memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize input array with random values
    srand(time(NULL));
    initializeArray(h_input, N);
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaError_t err;
    
    err = cudaMalloc((void **)&d_input, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_input error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Calculate grid size
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t outputSize = blocksPerGrid * sizeof(float);
    
    err = cudaMalloc((void **)&d_output, outputSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_output error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // First kernel launch: reduce array to block-level partial sums
    cudaEventRecord(start);
    arraySumReduction<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Allocate host memory for partial sums
    h_output = (float *)malloc(outputSize);
    if (!h_output) {
        fprintf(stderr, "Host output memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    // Copy partial sums back to host
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    // Final reduction on CPU (sum all partial sums)
    float gpu_sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        gpu_sum += h_output[i];
    }
    
    printf("Array size: %d elements\n", N);
    printf("GPU array sum time: %f ms\n", milliseconds);
    printf("GPU sum result: %f\n", gpu_sum);
    
    // Verify result with CPU computation
    printf("Verifying result...\n");
    float cpu_sum = cpuArraySum(h_input, N);
    printf("CPU sum result: %f\n", cpu_sum);
    
    float diff = fabs(gpu_sum - cpu_sum);
    float relative_error = diff / cpu_sum;
    if (relative_error < 1e-4) {
        printf("Result verification passed! (Relative error: %e)\n", relative_error);
    } else {
        fprintf(stderr, "Result verification failed!\n");
        fprintf(stderr, "Absolute difference: %f, Relative error: %e\n", diff, relative_error);
        exit(EXIT_FAILURE);
    }
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Free host memory
    free(h_input);
    free(h_output);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Array sum completed successfully!\n");
    return 0;
}
