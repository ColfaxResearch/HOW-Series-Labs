/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.04-threading-misc-histogram/solutions/instruction-02/worker.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



void Histogram(const float* age, int* const hist, const int n, const float group_width,
    const int m) {

    const int vecLen = 16; // Length of vectorized loop (lower is better,
                           // but a multiple of 64/sizeof(int))
    const float recGroupWidth = 1.0f/group_width; // Pre-compute the reciprocal
    const int nPrime = n - n%vecLen; // nPrime is a multiple of vecLen

    // Strip-mining the loop in order to vectorize the inner short loop
    for (int ii = 0; ii < nPrime; ii+=vecLen) { 
        // Temporary storage for vecLen indices. Necessary for vectorization
        int index[vecLen] __attribute__((aligned(64))); 

        // Vectorize the multiplication and rounding
        for (int i = ii; i < ii+vecLen; i++) 
            index[i-ii] = (int) ( age[i] * recGroupWidth );

        // Scattered memory access, does not get vectorized
        for (int c = 0; c < vecLen; c++) 
            hist[index[c]]++;
    }

    // Finish with the tail of the data (if n is not a multiple of vecLen)
    for (int i = nPrime; i < n; i++)
        hist[(int) ( age[i] * recGroupWidth )]++;
}
