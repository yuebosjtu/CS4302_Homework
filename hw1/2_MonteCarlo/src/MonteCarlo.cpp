#include "MonteCarlo.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <random>
#include <thread>

// 生成随机数的函数，基于种子
double erand48(unsigned short xi[3]) {
    // 使用传入的种子初始化随机数生成器
    static thread_local std::mt19937 gen(xi[0] + (xi[1] << 8) + (xi[2] << 16));
    static thread_local std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen);
}

// 串行版本的蒙特卡洛方法计算π
double monte_carlo_pi_serial(int num_samples) {
    int count = 0;
    unsigned short xi[3] = {1, 2, 3}; // 随机数种子
    
    for (int i = 0; i < num_samples; ++i) {
        double x = erand48(xi);
        double y = erand48(xi);
        if (x * x + y * y <= 1.0) {
            count++;
        }
    }
    
    return 4.0 * count / num_samples;
}

// 并行版本的蒙特卡洛方法计算π
double monte_carlo_pi_parallel(int num_samples, int num_threads) {
    int total_count = 0;
    
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int local_count = 0;
        
        // 为每个线程创建不同的随机数种子，确保不同的随机数流
        unsigned short xi[3];
        xi[0] = (unsigned short)(time(NULL) + tid * 12345);
        xi[1] = (unsigned short)(tid * 54321 + 1);
        xi[2] = (unsigned short)(tid * 98765 + 2);
        
        // 计算每个线程处理的样本数
        int samples_per_thread = num_samples / num_threads;
        int start = tid * samples_per_thread;
        int end = (tid == num_threads - 1) ? num_samples : start + samples_per_thread;
        
        // 蒙特卡洛采样
        for (int i = start; i < end; ++i) {
            double x = erand48(xi);
            double y = erand48(xi);
            if (x * x + y * y <= 1.0) {
                local_count++;
            }
        }
        
        // 原子操作累加结果
        #pragma omp atomic
        total_count += local_count;
    }
    
    return 4.0 * total_count / num_samples;
}

// 改进的并行版本，使用线程本地随机数生成器
double monte_carlo_pi_parallel_improved(int num_samples, int num_threads) {
    int total_count = 0;
    
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int local_count = 0;
        
        // 使用C++11的随机数生成器，为每个线程创建独立的生成器
        std::random_device rd;
        std::mt19937 gen(rd() + tid * 12345); // 不同线程使用不同种子
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        
        // 计算每个线程处理的样本数
        int samples_per_thread = num_samples / num_threads;
        int start = tid * samples_per_thread;
        int end = (tid == num_threads - 1) ? num_samples : start + samples_per_thread;
        
        // 蒙特卡洛采样
        for (int i = start; i < end; ++i) {
            double x = dis(gen);
            double y = dis(gen);
            if (x * x + y * y <= 1.0) {
                local_count++;
            }
        }
        
        // 原子操作累加结果
        #pragma omp atomic
        total_count += local_count;
    }
    
    return 4.0 * total_count / num_samples;
}

// 主接口函数
double monte_carlo_pi(int num_samples) {
    return monte_carlo_pi_serial(num_samples);
}

// 测试函数，比较串行和并行性能
void test_monte_carlo() {
    int samples = 1000000;
    int threads = 4;
    
    std::cout << "蒙特卡洛方法计算π测试" << std::endl;
    std::cout << "样本数: " << samples << std::endl;
    std::cout << "线程数: " << threads << std::endl;
    std::cout << "真实π值: " << 3.141592653589793 << std::endl;
    std::cout << std::endl;
    
    // 串行版本
    double start_time = omp_get_wtime();
    double pi_serial = monte_carlo_pi_serial(samples);
    double serial_time = omp_get_wtime() - start_time;
    
    std::cout << "串行结果: " << pi_serial << std::endl;
    std::cout << "串行时间: " << serial_time << " 秒" << std::endl;
    std::cout << "串行误差: " << abs(pi_serial - 3.141592653589793) << std::endl;
    std::cout << std::endl;
    
    // 并行版本
    start_time = omp_get_wtime();
    double pi_parallel = monte_carlo_pi_parallel(samples, threads);
    double parallel_time = omp_get_wtime() - start_time;
    
    std::cout << "并行结果: " << pi_parallel << std::endl;
    std::cout << "并行时间: " << parallel_time << " 秒" << std::endl;
    std::cout << "并行误差: " << abs(pi_parallel - 3.141592653589793) << std::endl;
    std::cout << "加速比: " << serial_time / parallel_time << std::endl;
    std::cout << std::endl;
    
    // 改进的并行版本
    start_time = omp_get_wtime();
    double pi_improved = monte_carlo_pi_parallel_improved(samples, threads);
    double improved_time = omp_get_wtime() - start_time;
    
    std::cout << "改进并行结果: " << pi_improved << std::endl;
    std::cout << "改进并行时间: " << improved_time << " 秒" << std::endl;
    std::cout << "改进并行误差: " << abs(pi_improved - 3.141592653589793) << std::endl;
    std::cout << "改进加速比: " << serial_time / improved_time << std::endl;
}
