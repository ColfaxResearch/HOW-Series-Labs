


#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>

void multiply(const double* const A, const double* const b,  double* c, const long n, const long m){

  const long jTile = 4096L;

  assert(n%jTile == 0);

#pragma omp parallel 
  {
    double temp_c[m] __attribute__((aligned(64)));
    temp_c[:] =0;

#pragma omp for
    for (long jj =0; jj < n; jj+=jTile)
      for (long i = 0; i < m; i++)
#pragma vector aligned
	for (long j =jj; j < jj+jTile; j++)
	  temp_c[i] += A[i*n+j] * b[j];
  
    for(long i = 0; i < m; i++) {
#pragma omp atomic
      c[i]+= temp_c[i];
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
  
  _mm_free(A);
  _mm_free(b);
  _mm_free(c);

}

