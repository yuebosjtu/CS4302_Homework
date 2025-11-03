#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256
// Add padding to avoid bank conflicts (assuming 32 banks)
#define BLOCK_SIZE_PADDED (BLOCK_SIZE + 1)

// Original GPU kernel: Reduction-based array sum
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

// Optimized kernel 1: Avoiding bank conflicts with padded shared memory
__global__ void arraySumOptimized1(const float *input, float *output, int n) {
    // Using a 1D padded array to avoid bank conflicts
    __shared__ float partialSum[BLOCK_SIZE + 32];  // Add padding for bank conflict avoidance
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localTid = threadIdx.x;
    
    // Load data from global memory to shared memory
    if (tid < n) {
        partialSum[localTid] = input[tid];
    } else {
        partialSum[localTid] = 0.0f;
    }
    __syncthreads();
    
    // Perform tree-based reduction with better memory access pattern
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localTid < stride) {
            partialSum[localTid] += partialSum[localTid + stride];
        }
        __syncthreads();
    }
    
    if (localTid == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

// Optimized kernel 2: Resolving warp divergence + Sequential addressing
__global__ void arraySumOptimized2(const float *input, float *output, int n) {
    __shared__ float partialSum[BLOCK_SIZE + 32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localTid = threadIdx.x;
    
    // Load data
    if (tid < n) {
        partialSum[localTid] = input[tid];
    } else {
        partialSum[localTid] = 0.0f;
    }
    __syncthreads();
    
    // Sequential addressing to avoid warp divergence
    // All threads in first half of block are active
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (localTid < stride) {
            partialSum[localTid] += partialSum[localTid + stride];
        }
        __syncthreads();
    }
    
    // Unroll last warp (no __syncthreads needed within a warp)
    if (localTid < 32) {
        volatile float* vPartialSum = partialSum;
        vPartialSum[localTid] += vPartialSum[localTid + 32];
        vPartialSum[localTid] += vPartialSum[localTid + 16];
        vPartialSum[localTid] += vPartialSum[localTid + 8];
        vPartialSum[localTid] += vPartialSum[localTid + 4];
        vPartialSum[localTid] += vPartialSum[localTid + 2];
        vPartialSum[localTid] += vPartialSum[localTid + 1];
    }
    
    if (localTid == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

// Optimized kernel 3: Multiple elements per thread + Loop unrolling
__global__ void arraySumOptimized3(const float *input, float *output, int n) {
    __shared__ float partialSum[BLOCK_SIZE + 32];
    
    int tid = blockIdx.x * blockDim.x * 2 + threadIdx.x;  // Each block handles 2x elements
    int localTid = threadIdx.x;
    
    // Each thread loads and sums multiple elements (reduces number of blocks)
    float sum = 0.0f;
    if (tid < n) sum += input[tid];
    if (tid + blockDim.x < n) sum += input[tid + blockDim.x];
    
    partialSum[localTid] = sum;
    __syncthreads();
    
    // Sequential addressing reduction with unrolled last warp
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (localTid < stride) {
            partialSum[localTid] += partialSum[localTid + stride];
        }
        __syncthreads();
    }
    
    // Unroll last warp
    if (localTid < 32) {
        volatile float* vPartialSum = partialSum;
        vPartialSum[localTid] += vPartialSum[localTid + 32];
        vPartialSum[localTid] += vPartialSum[localTid + 16];
        vPartialSum[localTid] += vPartialSum[localTid + 8];
        vPartialSum[localTid] += vPartialSum[localTid + 4];
        vPartialSum[localTid] += vPartialSum[localTid + 2];
        vPartialSum[localTid] += vPartialSum[localTid + 1];
    }
    
    if (localTid == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

// Optimized kernel 4: Fully unrolled with 4 elements per thread
__global__ void arraySumOptimized4(const float *input, float *output, int n) {
    __shared__ float partialSum[BLOCK_SIZE + 32];
    
    int tid = blockIdx.x * blockDim.x * 4 + threadIdx.x;  // Each block handles 4x elements
    int localTid = threadIdx.x;
    int gridStride = blockDim.x;
    
    // Each thread loads and sums 4 elements with loop unrolling
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * gridStride;
        if (idx < n) {
            sum += input[idx];
        }
    }
    
    partialSum[localTid] = sum;
    __syncthreads();
    
    // Fully unrolled reduction for powers of 2
    if (BLOCK_SIZE >= 512) {
        if (localTid < 256) { partialSum[localTid] += partialSum[localTid + 256]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (localTid < 128) { partialSum[localTid] += partialSum[localTid + 128]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (localTid < 64) { partialSum[localTid] += partialSum[localTid + 64]; }
        __syncthreads();
    }
    
    // Unroll last warp
    if (localTid < 32) {
        volatile float* vPartialSum = partialSum;
        if (BLOCK_SIZE >= 64) vPartialSum[localTid] += vPartialSum[localTid + 32];
        vPartialSum[localTid] += vPartialSum[localTid + 16];
        vPartialSum[localTid] += vPartialSum[localTid + 8];
        vPartialSum[localTid] += vPartialSum[localTid + 4];
        vPartialSum[localTid] += vPartialSum[localTid + 2];
        vPartialSum[localTid] += vPartialSum[localTid + 1];
    }
    
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
    int N = 1 << 23;  // 8M elements
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float *)malloc(size);
    
    // Initialize input array with random values
    srand(time(NULL));
    initializeArray(h_input, N);
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    
    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // CPU computation with timing
    printf("Array size: %d elements (%.2f MB)\n", N, size / (1024.0 * 1024.0));
    printf("\n=== CPU Array Sum ===\n");
    clock_t cpu_start = clock();
    float cpu_sum = cpuArraySum(h_input, N);
    clock_t cpu_end = clock();
    double cpu_time_ms = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU execution time: %.4f ms\n", cpu_time_ms);
    printf("CPU sum result: %f\n\n", cpu_sum);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float gpu_sum = 0.0f;
    float milliseconds = 0;
    float basic_time = 0;
    
    // Test 1: Original kernel
    {
        printf("=== Test 1: Original Kernel (with atomicAdd) ===\n");
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        size_t outputSize = blocksPerGrid * sizeof(float);
        cudaMalloc((void **)&d_output, outputSize);
        
        float *d_result;
        cudaMalloc((void **)&d_result, sizeof(float));
        
        cudaEventRecord(start);
        arraySumReduction<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
        finalReduction<<<1, threadsPerBlock>>>(d_output, d_result, blocksPerGrid);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        basic_time = milliseconds;
        
        cudaMemcpy(&gpu_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("Number of blocks: %d\n", blocksPerGrid);
        printf("GPU execution time: %.4f ms\n", milliseconds);
        printf("Speedup vs CPU: %.2fx\n", cpu_time_ms / milliseconds);
        printf("GPU sum result: %f\n", gpu_sum);
        
        float diff = fabs(gpu_sum - cpu_sum);
        float relative_error = diff / cpu_sum;
        printf("Relative error: %e - %s\n\n", relative_error, 
               relative_error < 1e-4 ? "PASSED" : "FAILED");
        
        cudaFree(d_output);
        cudaFree(d_result);
    }
    
    // Test 2: Optimized kernel 1 (Avoiding bank conflicts)
    {
        printf("=== Test 2: Optimized Kernel 1 (Bank Conflict Avoidance) ===\n");
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        size_t outputSize = blocksPerGrid * sizeof(float);
        cudaMalloc((void **)&d_output, outputSize);
        
        float *d_result;
        cudaMalloc((void **)&d_result, sizeof(float));
        
        cudaEventRecord(start);
        arraySumOptimized1<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
        finalReduction<<<1, threadsPerBlock>>>(d_output, d_result, blocksPerGrid);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaMemcpy(&gpu_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("Number of blocks: %d\n", blocksPerGrid);
        printf("GPU execution time: %.4f ms\n", milliseconds);
        printf("Speedup vs CPU: %.2fx\n", cpu_time_ms / milliseconds);
        printf("Speedup vs Basic: %.2fx (%.2f%% improvement)\n", 
               basic_time / milliseconds, (1 - milliseconds / basic_time) * 100);
        printf("GPU sum result: %f\n", gpu_sum);
        
        float diff = fabs(gpu_sum - cpu_sum);
        float relative_error = diff / cpu_sum;
        printf("Relative error: %e - %s\n\n", relative_error, 
               relative_error < 1e-4 ? "PASSED" : "FAILED");
        
        cudaFree(d_output);
        cudaFree(d_result);
    }
    
    // Test 3: Optimized kernel 2 (Warp divergence reduction + Sequential addressing)
    {
        printf("=== Test 3: Optimized Kernel 2 (Warp Divergence Reduction) ===\n");
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        size_t outputSize = blocksPerGrid * sizeof(float);
        cudaMalloc((void **)&d_output, outputSize);
        
        float *d_result;
        cudaMalloc((void **)&d_result, sizeof(float));
        
        cudaEventRecord(start);
        arraySumOptimized2<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
        finalReduction<<<1, threadsPerBlock>>>(d_output, d_result, blocksPerGrid);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaMemcpy(&gpu_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("Number of blocks: %d\n", blocksPerGrid);
        printf("GPU execution time: %.4f ms\n", milliseconds);
        printf("Speedup vs CPU: %.2fx\n", cpu_time_ms / milliseconds);
        printf("Speedup vs Basic: %.2fx (%.2f%% improvement)\n", 
               basic_time / milliseconds, (1 - milliseconds / basic_time) * 100);
        printf("GPU sum result: %f\n", gpu_sum);
        
        float diff = fabs(gpu_sum - cpu_sum);
        float relative_error = diff / cpu_sum;
        printf("Relative error: %e - %s\n\n", relative_error, 
               relative_error < 1e-4 ? "PASSED" : "FAILED");
        
        cudaFree(d_output);
        cudaFree(d_result);
    }
    
    // Test 4: Optimized kernel 3 (2 elements per thread)
    {
        printf("=== Test 4: Optimized Kernel 3 (2 Elements per Thread) ===\n");
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
        size_t outputSize = blocksPerGrid * sizeof(float);
        cudaMalloc((void **)&d_output, outputSize);
        
        float *d_result;
        cudaMalloc((void **)&d_result, sizeof(float));
        
        cudaEventRecord(start);
        arraySumOptimized3<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
        finalReduction<<<1, threadsPerBlock>>>(d_output, d_result, blocksPerGrid);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaMemcpy(&gpu_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("Number of blocks: %d\n", blocksPerGrid);
        printf("GPU execution time: %.4f ms\n", milliseconds);
        printf("Speedup vs CPU: %.2fx\n", cpu_time_ms / milliseconds);
        printf("Speedup vs Basic: %.2fx (%.2f%% improvement)\n", 
               basic_time / milliseconds, (1 - milliseconds / basic_time) * 100);
        printf("GPU sum result: %f\n", gpu_sum);
        
        float diff = fabs(gpu_sum - cpu_sum);
        float relative_error = diff / cpu_sum;
        printf("Relative error: %e - %s\n\n", relative_error, 
               relative_error < 1e-4 ? "PASSED" : "FAILED");
        
        cudaFree(d_output);
        cudaFree(d_result);
    }
    
    // Test 5: Optimized kernel 4 (4 elements per thread + Fully unrolled)
    {
        printf("=== Test 5: Optimized Kernel 4 (4 Elements per Thread + Fully Unrolled) ===\n");
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (N + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
        size_t outputSize = blocksPerGrid * sizeof(float);
        cudaMalloc((void **)&d_output, outputSize);
        
        float *d_result;
        cudaMalloc((void **)&d_result, sizeof(float));
        
        cudaEventRecord(start);
        arraySumOptimized4<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
        finalReduction<<<1, threadsPerBlock>>>(d_output, d_result, blocksPerGrid);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaMemcpy(&gpu_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("Number of blocks: %d\n", blocksPerGrid);
        printf("GPU execution time: %.4f ms\n", milliseconds);
        printf("Speedup vs CPU: %.2fx\n", cpu_time_ms / milliseconds);
        printf("Speedup vs Basic: %.2fx (%.2f%% improvement)\n", 
               basic_time / milliseconds, (1 - milliseconds / basic_time) * 100);
        printf("GPU sum result: %f\n", gpu_sum);
        
        float diff = fabs(gpu_sum - cpu_sum);
        float relative_error = diff / cpu_sum;
        printf("Relative error: %e - %s\n\n", relative_error, 
               relative_error < 1e-4 ? "PASSED" : "FAILED");
        
        cudaFree(d_output);
        cudaFree(d_result);
    }
    
    // Summary
    printf("========== Summary ==========\n");
    printf("All optimizations applied:\n");
    printf("1. Bank conflict avoidance (padded shared memory)\n");
    printf("2. Warp divergence reduction (sequential addressing)\n");
    printf("3. Loop unrolling (last warp unrolling)\n");
    printf("4. Multiple elements per thread (reduced blocks)\n");
    printf("5. Fully unrolled reduction loops\n");
    printf("==============================\n");
    
    // Free device memory
    cudaFree(d_input);
    
    // Free host memory
    free(h_input);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
