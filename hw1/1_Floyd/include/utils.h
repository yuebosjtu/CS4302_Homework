#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>

#define INF 9999999

// get current time
inline double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

inline void print_matrix(int **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] == INF)
                printf("INF ");
            else
                printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
}

inline int** allocate_matrix(int n) {
    int **matrix = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (int*)malloc(n * sizeof(int));
    }
    return matrix;
}

inline void free_matrix(int **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// generate adjacency matrix of a random graph
inline void generate_random_graph(int **matrix, int n, double density) {
    srand(time(NULL));

    // initialize matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                matrix[i][j] = 0;
            }
            else {
                matrix[i][j] = INF;
            }
        }
    }

    // generate edges based on density
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && (double)rand() / RAND_MAX < density) {
                matrix[i][j] = rand() % 100 + 1;
            }
        }
    }
}

#endif // UTILS_H