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
        if (localTid < stride && (localTid + stride) < BLOCK_SIZE) {
            // partialSum[localTid] += partialSum[localTid + stride];
            atomicAdd(&partialSum[localTid], partialSum[localTid + stride]);
        }
        __syncthreads();
    }
    
    // Thread 0 writes the block's result to global memory
    if (localTid == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

// GPU kernel: Final reduction for small array (output from first reduction)
// This kernel is designed to handle the output array from the first reduction
// It uses a single block to sum all partial results
__global__ void finalReduction(float *data, float *result, int n) {
    // Shared memory for partial sums within the block
    __shared__ float partialSum[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int localTid = threadIdx.x;
    
    // Each thread may need to handle multiple elements if n > BLOCK_SIZE
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += data[i];
    }
    partialSum[localTid] = sum;
    __syncthreads();
    
    // Perform tree-based reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localTid < stride) {
            partialSum[localTid] += partialSum[localTid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the final result to global memory
    if (localTid == 0) {
        result[0] = partialSum[0];
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
    int N = 1 << 23;  // 1 billion elements
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float *)malloc(size);
    float gpu_sum = 0.0f;
    
    // Initialize input array with random values
    srand(time(NULL));
    initializeArray(h_input, N);
    
    // Allocate device memory
    float *d_input, *d_output;
    
    cudaMalloc((void **)&d_input, size);
    
    // Calculate grid size
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t outputSize = blocksPerGrid * sizeof(float);
    
    cudaMalloc((void **)&d_output, outputSize);
    
    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate device memory for final result
    float *d_result;
    cudaMalloc((void **)&d_result, sizeof(float));
    
    // Start timing
    cudaEventRecord(start);
    
    // First kernel launch: reduce array to block-level partial sums
    arraySumReduction<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    
    // Second kernel launch: final reduction on GPU
    // Use a single block to sum all partial results
    finalReduction<<<1, threadsPerBlock>>>(d_output, d_result, blocksPerGrid);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy final result back to host
    cudaMemcpy(&gpu_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Array size: %d elements\n", N);
    printf("Number of blocks in first reduction: %d\n", blocksPerGrid);
    printf("GPU array sum time (including both reductions): %f ms\n", milliseconds);
    printf("GPU sum result: %f\n", gpu_sum);
    
    // CPU computation with timing
    printf("\nVerifying result with CPU computation...\n");
    
    // Start CPU timing
    clock_t cpu_start = clock();
    float cpu_sum = cpuArraySum(h_input, N);
    clock_t cpu_end = clock();
    
    double cpu_time_ms = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    
    printf("CPU array sum time: %f ms\n", cpu_time_ms);
    printf("CPU sum result: %f\n", cpu_sum);
    
    // Performance comparison
    printf("\n========== Performance Comparison ==========\n");
    printf("GPU Time: %.4f ms\n", milliseconds);
    printf("CPU Time: %.4f ms\n", cpu_time_ms);
    printf("Speedup: %.2fx (GPU is %.2fx faster than CPU)\n", 
           cpu_time_ms / milliseconds, cpu_time_ms / milliseconds);
    printf("===========================================\n");
    
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
    cudaFree(d_result);
    
    // Free host memory
    free(h_input);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
