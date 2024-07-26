#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cmath>

using namespace std;

double monte_carlo_worker(int num_points, unsigned int seed) 
{
    int count = 0;

    for (int i = 0; i < num_points; ++i) {
        double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
        double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;
        if (x * x + y * y <= 1.0) 
        {
            ++count;
        }
    }
    return count;
}

double monte_carlo_parallel(int num_points, int num_threads) 
{
    int total_inside_circle = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num();
        int count = monte_carlo_worker(num_points / num_threads, seed);
        
        #pragma omp atomic
        total_inside_circle += count;
    }

    return (4.0 * total_inside_circle) / num_points;
}

int main() 
{
    srand(time(0));
    int num_points;
    int num_threads;

    cout << "Enter Number of Iterations: ";
    cin >> num_points;
    cout << "Enter Number of Threads for execution: ";
    cin >> num_threads;

    clock_t start = clock();
    unsigned int seed = time(NULL);
    double pi_sequential = (4.0 * monte_carlo_worker(num_points, seed)) / num_points;
    clock_t end = clock();
    double sequential_time = double(end - start) / CLOCKS_PER_SEC;
    cout << "Sequential Time = " << sequential_time << " seconds" << endl;

    start = clock();
    double pi_parallel = monte_carlo_parallel(num_points, num_threads);
    end = clock();
    double parallel_time = double(end - start) / CLOCKS_PER_SEC;
    cout << "Parallel Time = " << parallel_time << " seconds" << endl;
    cout << "Speedup = " << sequential_time / parallel_time << "x" << endl;
    cout << "Approximate PI value (Parallel): " << pi_parallel << endl;

    return 0;
}
