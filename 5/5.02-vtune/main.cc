/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/5/5.02-vtune/main.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <cmath>
#include <mkl_vsl.h>

int IterativeSolver(const int n, const double* M, const double* b, double* x, const double minAccuracy);

void InitializeMatrix(const int n, double* M) {
  // "Good" (diagonally-dominated) matrix for Jacobi method
  for (int i = 0; i < n; i++) {
    double sum = 0;
    for (int j = 0; j < n; j++) {
      M[i*n+j] = (double)(i*n+j);
      sum += M[i*n+j];
    }
    sum -= M[i*n+i];
    M[i*n+i] = 2.0*sum;
  }
}

int main(int argv, char* argc[]){
  const int n=256;
  const int nBVectors = 20000; // The number of b-vectors
  printf("\n\033[1mOpenMP Scheduling Modes Benchmark\033[0m\n");
  printf("\nSolving %d systems of size %dx%d with the Jacobi method on %s...\n", 
	 nBVectors, n, n,
#ifndef __MIC__
	 "CPU"
#else
	 "MIC"
#endif
	 );
  double* M = (double*) _mm_malloc(sizeof(double)*n*n, 64);
  double* x = (double*) _mm_malloc(sizeof(double)*n*nBVectors, 64);
  double* b = (double*) _mm_malloc(sizeof(double)*n*nBVectors, 64);
  double* accuracy = (double*) _mm_malloc(sizeof(double)*nBVectors, 64);
  InitializeMatrix(n, M);
  VSLStreamStatePtr rnStream; 
  vslNewStream( &rnStream, VSL_BRNG_MT19937, 1234 );
  vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, n*nBVectors, b, 0.0, 1.0);
  for (int i = 0; i < nBVectors; i++)
    accuracy[i] = 0.5*(1.0+sin(15.0*double(i)/double(nBVectors)));
  accuracy[0:nBVectors]=exp(-28.0+26.0*accuracy[0:nBVectors]);
  printf("Initialized %d vectors and a [%d x %d] matrix\n\n",
	 nBVectors, n, n); fflush(stdout);

  const int nTrials=10;
  const int nMethods = 14;
  const int itSkip = 1;
  printf("\033[1m%10s %10s\033[0m\n", "Trial", "Time, s");
  double tAvg = 0.0;
  double dtAvg = 0.0;
  for (int t=0; t<nTrials; t++){
    const double t0 = omp_get_wtime();
#pragma omp parallel for
    for (int c = 0; c < nBVectors; c++)
      IterativeSolver(n, M, &b[c*n], &x[c*n], accuracy[c]);
    const double t1 = omp_get_wtime();
    printf(" %10d %10.4f %s\n", t+1, t1-t0, (t+1<=itSkip?"*":""));
    fflush(stdout);
    if (t >= itSkip) {
      tAvg += (t1-t0);
      dtAvg += (t1-t0)*(t1-t0);
    }
    fflush(stdout);
  }
  tAvg /= (double)(nTrials-itSkip);
  dtAvg /= (double)(nTrials-itSkip);
  dtAvg = sqrt(dtAvg - tAvg*tAvg);
  printf("\033[1m%s %12s \033[42m%10.4f +- %.4f s\033[0m\n",
	 "Average time:", "", tAvg, dtAvg);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");

  _mm_free(M);
  _mm_free(x);
  _mm_free(b);
  _mm_free(accuracy);
}
