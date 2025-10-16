#include "floyd.h"
#include <omp.h>

void floyd_serial(int **A, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (A[i][k] != INF && A[k][j] != INF) {
                    int new_path = A[i][k] + A[k][j];
                    if (new_path < A[i][j]) {
                        A[i][j] = new_path;
                    }
                }
            }
        }
    }
}
void floyd_parallel(int **A, int n, int num_threads) {
    // 改进的并行版本
    omp_set_num_threads(num_threads);
    
    for (int k = 0; k < n; k++) {
        // 并行化i循环，每个线程处理不同的行
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (A[i][k] != INF && A[k][j] != INF) {
                    int new_path = A[i][k] + A[k][j];
                    if (new_path < A[i][j]) {
                        A[i][j] = new_path;
                    }
                }
            }
        }
    }
}

void floyd_parallel_optimized(int **A, int n, int num_threads) {
    // 更优化的并行版本：减少条件判断，预取数据
    omp_set_num_threads(num_threads);
    
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for schedule(static) 
        for (int i = 0; i < n; i++) {
            int aik = A[i][k];  // 预取A[i][k]值，减少重复访问
            if (aik != INF) {   // 只有当A[i][k]不是INF时才进入内层循环
                for (int j = 0; j < n; j++) {
                    int akj = A[k][j];
                    if (akj != INF) {
                        int new_path = aik + akj;
                        if (new_path < A[i][j]) {
                            A[i][j] = new_path;
                        }
                    }
                }
            }
        }
    }
}

void floyd_parallel_blocked(int **A, int n, int num_threads, int block_size) {
    // 分块并行版本：改善缓存局部性
    omp_set_num_threads(num_threads);
    
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int ii = 0; ii < n; ii += block_size) {
            for (int jj = 0; jj < n; jj += block_size) {
                // 处理块内的元素
                int i_end = (ii + block_size < n) ? ii + block_size : n;
                int j_end = (jj + block_size < n) ? jj + block_size : n;
                
                for (int i = ii; i < i_end; i++) {
                    int aik = A[i][k];
                    if (aik != INF) {
                        for (int j = jj; j < j_end; j++) {
                            int akj = A[k][j];
                            if (akj != INF) {
                                int new_path = aik + akj;
                                if (new_path < A[i][j]) {
                                    A[i][j] = new_path;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}