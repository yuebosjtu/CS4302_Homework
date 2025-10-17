#ifndef MONTECARLO_H
#define MONTECARLO_H

// 主接口函数
double monte_carlo_pi(int num_samples);

// 串行版本
double monte_carlo_pi_serial(int num_samples);

// 并行版本
double monte_carlo_pi_parallel(int num_samples, int num_threads);

// 改进的并行版本
double monte_carlo_pi_parallel_improved(int num_samples, int num_threads);

// 测试函数
void test_monte_carlo();

#endif // MONTECARLO_H