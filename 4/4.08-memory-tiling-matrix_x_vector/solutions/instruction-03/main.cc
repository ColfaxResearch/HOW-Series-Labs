#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>


void recursiveMultiply(const double* const A, const double* const b,  double* c, const long n, const long m, const long lda){

  const long jThreshold = 8192L;
  const long iThreshold = 64L;

  assert(m%iThreshold == 0);
  assert(n%jThreshold == 0);

  if ((m <= iThreshold) && (n <= jThreshold)) {

    for (long i = 0; i < m; i++)
#pragma vector aligned
      for (long j = 0; j < n; j++)
	c[i] = A[i*lda+j] * b[j];

  } else {
    
    if (m*jThreshold >= n*iThreshold) {

      double c1[m/2] __attribute__((aligned(64)));
#pragma omp task
      {
      recursiveMultiply(&A[0*lda + 0], &b[0], c1, n, m/2, lda);
      }

      double c2[m/2] __attribute__((aligned(64)));
      recursiveMultiply(&A[(m/2)*lda + 0], &b[m/2], c2, n, m/2, lda);

#pragma omp taskwait

      c[0:m/2]   += c1[0:m/2];
      c[m/2:m/2] += c2[0:m/2];

    } else {

      double c1[m] __attribute__((aligned(64)));
#pragma omp task
      {
      recursiveMultiply(&A[0*lda + 0], &b[0], c1, n/2, m, lda);
      }

      double c2[m] __attribute__((aligned(64)));
      recursiveMultiply(&A[0*lda + n/2], &b[0], c2, n/2, m, lda);

#pragma omp taskwait

      c[0:m] += c1[0:m];
      c[0:m] += c2[0:m];

    }

  }

}


void multiply(const double* const A, const double* const b,  double* c, const long n, const long m){

#pragma omp parallel
  {
    #pragma omp single
    {
      recursiveMultiply(A, b, c, n, m, n);
    }
  }

} 


int main(){
  const long m=1<<10;
  const long n=1<<19;

  double * A = (double*) _mm_malloc(sizeof(double)*n*m, 64);
  double * b = (double*) _mm_malloc(sizeof(double)*n, 64);
  double * c = (double*) _mm_malloc(sizeof(double)*m, 64);

#pragma omp parallel for  
  for (long i = 0; i < m; i++) {
    for (long j = 0; j < n; j++) {
      A[i*n+j]= i;
    }
  }

#pragma omp parallel for  
  for (long i = 0; i < n; i++) {
    b[i]=(double)i/(double)n;
  }
  for (long i = 0; i < m; i++) {
    c[i] = 0;
  }


  printf("Matrix vector multiplication with %d x %d Matrix (%.2f GB) on %s...\n\n", 
	 n, m, (double)(n*m)*8.0e-9,
#ifndef __MIC__
         "CPU"
#else
         "MIC"
#endif
	   );
  const int nSteps = 10;
  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration is warm-up on Xeon Phi coprocessor
  printf("\033[1m%5s %10s %8s\033[0m\n", "Step", "Time, s", "GFLOP/s");
  for (int step = 1; step <= nSteps; step++) {
    
    const double tStart = omp_get_wtime(); // Start timing
    multiply(A,b,c,n, m);
    const double tEnd = omp_get_wtime(); // End timing
    
    const float HztoGFLOPs = 2*1e-9*n*m;
    
    if (step > skipSteps) { // Collect statistics
      rate  += HztoGFLOPs/(tEnd - tStart); 
      dRate += HztoGFLOPs*HztoGFLOPs/((tEnd - tStart)*(tEnd-tStart)); 
    }

    if(step == 1) {
      double norm = 0.0;
      for ( int i = 0 ; i < m; i++)
	norm += (c[i]-i*((double)n-1.0)/2.0)*(c[i]-i*((double)n-1.0)/2.0);
      
      if (norm > 1e-10)
	printf("Something is wrong... Norm is equal to %f\n", norm);

      for (int i = 0; i < m; i++) 
	c[i] = 0.0;
    }
    
    printf("%5d %10.3e %8.1f %s\n", 
	   step, (tEnd-tStart), HztoGFLOPs/(tEnd-tStart), (step<=skipSteps?"*":""));
    fflush(stdout);
  }
  rate/=(double)(nSteps-skipSteps); 
  dRate=sqrt(dRate/(double)(nSteps-skipSteps)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");
  
  printf("Checking the results...\n");

  _mm_free(A);
  _mm_free(b);
  _mm_free(c);

}
