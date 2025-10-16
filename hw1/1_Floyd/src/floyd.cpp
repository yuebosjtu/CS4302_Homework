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
    omp_set_num_threads(num_threads);
    
    for (int k = 0; k < n; k++) {
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