#include "MonteCarlo.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    int samples = 1000000;  // 默认样本数
    int threads = 4;        // 默认线程数
    
    // 解析命令行参数
    if (argc >= 2) {
        samples = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    
    std::cout << "=== 蒙特卡洛方法计算π ===" << std::endl;
    std::cout << "参数设置:" << std::endl;
    std::cout << "  样本数: " << samples << std::endl;
    std::cout << "  线程数: " << threads << std::endl;
    std::cout << "  真实π值: 3.141592653589793" << std::endl;
    std::cout << std::endl;
    
    // 运行测试
    if (argc >= 4 && std::string(argv[3]) == "test") {
        test_monte_carlo();
    } else {
        // 运行用户指定的配置
        double pi_serial = monte_carlo_pi_serial(samples);
        double pi_parallel = monte_carlo_pi_parallel(samples, threads);
        double pi_improved = monte_carlo_pi_parallel_improved(samples, threads);
        
        std::cout << "计算结果:" << std::endl;
        std::cout << "  串行版本:     " << pi_serial << std::endl;
        std::cout << "  并行版本:     " << pi_parallel << std::endl;
        std::cout << "  改进并行版本: " << pi_improved << std::endl;
        std::cout << std::endl;
        
        std::cout << "误差分析:" << std::endl;
        std::cout << "  串行误差:     " << abs(pi_serial - 3.141592653589793) << std::endl;
        std::cout << "  并行误差:     " << abs(pi_parallel - 3.141592653589793) << std::endl;
        std::cout << "  改进并行误差: " << abs(pi_improved - 3.141592653589793) << std::endl;
    }
    
    return 0;
}