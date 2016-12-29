/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.09-memory-loop-fusion-statistics/solutions/instruction-01/main.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cmath>
#include <omp.h>

void RunStatistics(const int m, const int n, float* const mean, float* const stdev);

int main(){

  const int m = 10000; // Number of data sets
  const int n = 40000; // Number of elements in each data set

  // Allocating data for results
  float resultMean[m];
  float resultStdev[m];

  printf("Running statistics on %d arrays of %d random numbers on %s...\n\n", 
	 m, n, 
#ifndef __MIC__
         "CPU"
#else
         "MIC"
#endif
	   );

  const int nTrials = 10;
  double rate = 0, dRate = 0; // Benchmarking data
  const int skipTrials = 2; // First few trials are warm-up on Xeon Phi coprocessor
  printf("\033[1m%5s %10s %8s\033[0m\n", "Trial", "Time, s", "GVal/s");
  for (int trial = 1; trial <= nTrials; trial++) {

    const double tStart = omp_get_wtime();
    RunStatistics(m, n, resultMean, resultStdev);
    const double tEnd = omp_get_wtime();

    const float HztoGFLOPs = float(n)*float(m)*1e-9;
    
    if (trial > skipTrials) { // Collect statistics
      rate  += HztoGFLOPs/(tEnd - tStart); 
      dRate += HztoGFLOPs*HztoGFLOPs/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e %8.2f %s\n", 
	   trial, (tEnd-tStart), HztoGFLOPs/(tEnd-tStart), (trial<=skipTrials?"*":""));
    fflush(stdout);
  }

  rate/=(double)(nTrials-skipTrials); 
  dRate=sqrt(dRate/(double)(nTrials-skipTrials)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.2f +- %.2f billion values/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");

  printf("Some of the results for verification:\n    ...\n");
  for (int i = 10; i < 14; i++)
    printf("    i=%d:   x = %8.2f+-%.2f (expected = %8.2f+-%.2f)\n", i, resultMean[i], resultStdev[i], (float)i, 1.0f);
  printf("    ...\n"); fflush(stdout);

}
