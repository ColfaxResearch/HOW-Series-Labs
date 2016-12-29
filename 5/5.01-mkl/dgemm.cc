/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/5/5.01-mkl/dgemm.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <omp.h>

int main(int argc, char* argv[]) {

  long n;
  if (argc>1)
    n = atoi(argv[1]);
  else
    n = 8000;

  double* A=(double*)_mm_malloc(n*n*sizeof(double), 64);
  double* B=(double*)_mm_malloc(n*n*sizeof(double), 64);
  double* C=(double*)_mm_malloc(n*n*sizeof(double), 64);

  const double HztoPerf = 1e-9*double(2*n*n*n);

  const int nTrials=10;
  const int skipTrials=2;
  double rate=0, dRate=0;

  printf("\n\033[1mBenchmarking DGEMM.\033[0m\n");
  printf("Problem size: %dx%d (%.3f GB)\n",
	 n, n, double(3L*n*n*sizeof(double))*1e-9);
  printf("    Platform: %s\n",
#ifndef __MIC__
	 "CPU"
#else
	 "MIC"
#endif
	 );
  printf("     Threads: %d\n", omp_get_max_threads());
  printf("    Affinity: %s\n\n", getenv("KMP_AFFINITY"));

  // Initializing data
#pragma omp parallel for
  for (int i = 0; i < n*n; i++) {
    A[i] =  (double)i;
    B[i] = -(double)i;
    C[i] = 0.0;
  }

  printf("\033[1m%5s %10s %15s\033[0m\n", "Trial", "Time, s", "Perf, GFLOP/s");

  for (int trial = 1; trial <= nTrials; trial++) {
    const double tStart = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, n, n, 1.0, A, n, B, n, 0.0, C, n);
    const double tEnd = omp_get_wtime();

    if ( trial > skipTrials) { // First two iterations are slow on Xeon Phi; exclude them
      rate  += HztoPerf/(tEnd - tStart); 
      dRate += HztoPerf*HztoPerf/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e %15.2f %s\n", 
	   trial, (tEnd-tStart), HztoPerf/(tEnd-tStart), (trial<=skipTrials?"*":""));
    fflush(stdout);
  }

  rate/=(double)(nTrials-skipTrials); 
  dRate=sqrt(dRate/(double)(nTrials-skipTrials)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.2f +- %.2f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");


  _mm_free(A);
  _mm_free(B);
  _mm_free(C);

}
