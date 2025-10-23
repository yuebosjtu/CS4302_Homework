#ifndef FLOYD_H
#define FLOYD_H

#define INF 9999999

void floyd_serial(int **A, int n);

void floyd_parallel(int **A, int n, int num_threads);

#endif // FLOYD_H