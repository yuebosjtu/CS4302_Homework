#include "MonteCarlo.h"

#include <random>
#include <chrono>
#include <omp.h>

// serial Monte Carlo pi computation
double monte_carlo_serial(int num_samples) {
	if (num_samples <= 0) return 0.0;

	// use C++11 (mt19937_64) to obtain random numbers
	uint64_t seed = static_cast<uint64_t>(
		std::chrono::high_resolution_clock::now().time_since_epoch().count());
	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	long long inside = 0;
	for (int i = 0; i < num_samples; ++i) {
		double x = dist(rng);
		double y = dist(rng);
		if (x*x + y*y <= 1.0) ++inside;
	}
	return 4.0 * static_cast<double>(inside) / static_cast<double>(num_samples);
}

// parallel Monte Carlo pi computation
double monte_carlo_parallel(int num_samples, int num_threads) {
	if (num_samples <= 0) return 0.0;
	if (num_threads <= 0) num_threads = 1;

	long long inside = 0;

	#pragma omp parallel num_threads(num_threads)
	{
		int tid = omp_get_thread_num();

		// generate a unique seed for each thread
		uint64_t base_seed = static_cast<uint64_t>(
			std::chrono::high_resolution_clock::now().time_since_epoch().count());
		// mix in constants to reduce correlation between threads/runs
		uint64_t thread_seed = base_seed ^ (static_cast<uint64_t>(tid) + 0x9e3779b97f4a7c15ULL + (base_seed<<6) + (base_seed>>2));

		std::mt19937_64 rng(thread_seed);
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		// use reduction to accumulate results from all threads
		#pragma omp for reduction(+:inside) schedule(static)
		for (int i = 0; i < num_samples; ++i) {
			double x = dist(rng);
			double y = dist(rng);
			if (x*x + y*y <= 1.0) ++inside;
		}
	}

	return 4.0 * static_cast<double>(inside) / static_cast<double>(num_samples);
}

