#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        #pragma omp critical
        {
            std::cout << "Hello from thread " << tid
                      << " / " << nthreads << std::endl;
        }
    }
    return 0;
}
