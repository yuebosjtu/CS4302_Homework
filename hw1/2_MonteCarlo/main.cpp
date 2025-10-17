#include "MonteCarlo.h"

#include <iostream>
#include <iomanip>
#include <omp.h>
#include <chrono>

int main(int argc, char** argv) {
	int num_samples = 10000000; // default sample count
	int num_threads = omp_get_max_threads(); // default thread count

	if (argc > 1) {
		num_samples = std::atoi(argv[1]);
	}
	if (argc > 2) {
		num_threads = std::atoi(argv[2]);
	}

	std::cout << "Monte Carlo pi estimation\n";
	std::cout << "samples: " << num_samples << ", threads: " << num_threads << "\n\n";

	// serial
	auto t0 = std::chrono::high_resolution_clock::now();
	double pi_serial = monte_carlo_serial(num_samples);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> dur_serial = t1 - t0;

	std::cout << "Serial: pi = " << pi_serial
			  << ", time = " << dur_serial.count() << " s\n";

	// parallel
	omp_set_num_threads(num_threads);
	auto t2 = std::chrono::high_resolution_clock::now();
	double pi_parallel = monte_carlo_parallel(num_samples, num_threads);
	auto t3 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> dur_parallel = t3 - t2;

	std::cout << "Parallel: pi = " << pi_parallel
			  << ", time = " << dur_parallel.count() << " s\n";

	std::cout << "Speedup (serial / parallel) = ";
	if (dur_parallel.count() > 0.0)
		std::cout << (dur_serial.count() / dur_parallel.count()) << "\n";
	else
		std::cout << "inf\n";

	return 0;
}