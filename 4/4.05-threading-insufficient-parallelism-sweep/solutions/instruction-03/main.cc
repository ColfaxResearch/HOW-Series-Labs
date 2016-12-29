/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.05-threading-insufficient-parallelism-sweep/solutions/instruction-03/main.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <malloc.h>
#include <cmath>
#include <omp.h>
#include <cstdio>

void SumColumns(const int m, const int n, long* M, long* s);

int main(){
  const int n = 100000000, m = 4; // n is the number of columns (inner dimension), 
  // m is the number of rows (outer dimension)
  long* matrix = (long*)_mm_malloc(sizeof(long)*m*n, 64);
  long* sums   = (long*)_mm_malloc(sizeof(long)*m, 64); // will contain sum of matrix rows
  const double HztoPerf = 1e-9*double(m*n)*sizeof(long);

  const int nTrials=10;
  double rate=0, dRate=0;

  printf("\n\033[1mComputing the sums of elements in each row of a wide, short matrix.\033[0m\n");
  printf("Problem size: %.3f GB, outer dimension: %d, threads: %d (%s)\n\n", 
	 (double)(sizeof(long))*(double)(n)*(double)m/(double)(1<<30),
	 m, omp_get_max_threads(),
#ifndef __MIC__
	 "CPU"
#else
	 "MIC"
#endif
	 );

  // Initializing data
#pragma omp parallel for
  for (int i = 0; i < m; i++) 
    for (int j = 0; j < n; j++)
      matrix[i*n + j] = (long)i;

  printf("\033[1m%5s %10s  %10s\033[0m\n", "Trial", "Time, s", "Perf(GB/s)");
  
  const int skipTrials=2;
  // Benchmarking SumColumns(...)
  for (int trial = 1; trial <= nTrials; trial++) {
    const double tStart=omp_get_wtime();
    SumColumns(m, n, matrix, sums);
    const double tEnd=omp_get_wtime();

    if ( trial > skipTrials) { // First two iterations are slow on Xeon Phi; exclude them
      rate  += HztoPerf/(tEnd - tStart); 
      dRate += HztoPerf*HztoPerf/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e  %10.2f %s\n", 
	   trial, (tEnd-tStart), HztoPerf/(tEnd-tStart), (trial<=skipTrials?"*":""));
    fflush(stdout);

    // Verifying that the result is correct
    for (int i = 0; i < m; i++) 
      if (sums[i] != i*n) 
	printf("Results are incorrect!");

  }
  rate/=(double)(nTrials-skipTrials); 
  dRate=sqrt(dRate/(double)(nTrials-skipTrials)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.2f +- %.2f GB/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");

  _mm_free(sums); _mm_free(matrix);
}
