


#include <omp.h>

void Histogram(const float* age, int* const hist, const int n, const float group_width,
    const int m) {
#ifdef __MIC__
    const int vecLen = 16; // Length of vectorized loop (lower is better,
                           // but a multiple of 64/sizeof(int))
#else
    const int vecLen = 32; // Length of vectorized loop (lower is better,
                           // but a multiple of 64/sizeof(int))
#endif
    const float recGroupWidth = 1.0f/group_width; // Pre-compute the reciprocal
    const int nPrime = n - n%vecLen; // nPrime is a multiple of vecLen
    const int nThreads = omp_get_max_threads();
    // Shared histogram with a private section for each thread
    int mPadded = m + (64 - m%64);
    int hist_global[nThreads*mPadded];

    // Strip-mining the loop in order to vectorize the inner short loop
#pragma omp parallel 
    {
        // Private variable to hold a copy of histogram in each thread
        int* hist_priv = &hist_global[omp_get_thread_num()*mPadded];
        hist_priv[0:m] = 0;

        // Temporary storage for vecLen indices. Necessary for vectorization
        int index[vecLen] __attribute__((aligned(64))); 

        // Distribute work across threads
        // Strip-mining the loop in order to vectorize the inner short loop
#pragma omp for
        for (int ii = 0; ii < nPrime; ii+=vecLen) { 
            // Vectorize the multiplication and rounding
#pragma vector aligned
            for (int i = ii; i < ii+vecLen; i++) 
                index[i-ii] = (int) ( age[i] * recGroupWidth );

            // Scattered memory access, does not get vectorized
            // There is no synchronization in this for-loop,
            // however, false sharing occurs here and ruins the performance
            for (int c = 0; c < vecLen; c++) 
                hist_priv[index[c]]++;
        }

        // Finish with the tail of the data (if n is not a multiple of vecLen)
#pragma omp single
        for (int i = nPrime; i < n; i++)
            hist_priv[(int) ( age[i] * recGroupWidth )]++;

        // Reduce private copies into global histogram
        for (int c = 0; c < m; c++) {
            // Protect the += operation with the lightweight atomic mutex
#pragma omp atomic 
            hist[c] += hist_priv[c];
        }
    }
}
