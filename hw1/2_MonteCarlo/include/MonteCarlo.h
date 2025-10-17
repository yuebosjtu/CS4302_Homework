#ifndef MONTECARLO_H
#define MONTECARLO_H

double monte_carlo_serial(int num_samples);

double monte_carlo_parallel(int num_samples, int num_threads);

#endif // MONTECARLO_H