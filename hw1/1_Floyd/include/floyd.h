#ifndef FLOYD_H
#define FLOYD_H

#define INF 9999999

void floyd_serial(int **A, int n);

void floyd_parallel(int **A, int n, int num_threads);

void floyd_parallel_optimized(int **A, int n, int num_threads);

void floyd_parallel_blocked(int **A, int n, int num_threads, int block_size);

#endif // FLOYD_H