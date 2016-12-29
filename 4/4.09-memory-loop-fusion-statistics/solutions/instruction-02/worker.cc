/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.09-memory-loop-fusion-statistics/solutions/instruction-02/worker.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <omp.h>
#include <mkl_vsl.h>
#include <cmath>

void GenerateRandomNumbers(const int n, float* const data, const float mean, const float stdev, VSLStreamStatePtr &rng) {
  
  // Filling one array with normally distributed n random numbers
  int status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, 
			     rng, n, data, mean, stdev);

}

void ComputeMeanAndStdev(const int n, const float* data, 
			 float* const resultMean, float* const resultStdev) {

  // Processing one array to compute the mean and standard deviation
  float sumx=0.0f, sumx2=0.0f;
#pragma vector aligned
  for (int j = 0; j < n; j++) {
    sumx  += data[j];
    sumx2 += data[j]*data[j];
  }
  *resultMean = sumx/(float)n;
  *resultStdev = sqrtf(sumx2/(float)n-(*resultMean)*(*resultMean));

}

void RunStatistics(const int m, const int n, 
		   float* const resultMean, float* const resultStdev) {

#pragma omp parallel
  {
    // We no longer need a buffer for m*n elements, only a buffer for n elements in each thread.
    // Saving on buffer space will increase performance.
    float data[n] __attribute__((aligned(64)));

    VSLStreamStatePtr rng;
    const int seed = omp_get_thread_num();
    int status = vslNewStream(&rng, VSL_BRNG_MT19937, omp_get_thread_num());

#pragma omp for schedule(guided)
    for (int i = 0; i < m; i++) {
      const float mean = (float)i;
      const float stdev = 1.0f;

      // Generating data for one array
      GenerateRandomNumbers(n, data, mean, stdev, rng);

      // Processing it immediately
      ComputeMeanAndStdev(n, data, &resultMean[i], &resultStdev[i]);
    }

    vslDeleteStream(&rng);
  }

}

