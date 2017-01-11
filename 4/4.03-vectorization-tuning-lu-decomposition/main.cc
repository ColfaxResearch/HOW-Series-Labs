#include <cmath>
#include <cstdio>
#include <omp.h>

void LU_Doolittle(const int n, float* A, float* L) {
  // This is the performance-critical function
  // On input: 
  //         square matrix A of size [n x n]
  // On output: 
  //         argument A is the upper triangular matrix U,
  //         argument L is the unit lower triangular matrix L
  L[0:n*n]=0.0f;
  for (int b = 0; b < n; b++) {
    L[b*n + b] = 1.0f;
    for (int i = b+1; i < n; i++) {
      L[i*n + b] = A[i*n + b]/A[b*n + b];
      for (int j = b; j < n; j++) 
	A[i*n + j] -= L[i*n + b]*A[b*n + j];
    }
  }
  L[(n-1)*n + (n-1)] = 1.0f;
}

void VerifyResult(const int n, float* U, float* L, float* refA) {

  // First, verifying that A=LU
  float A[n*n];
  A[:] = 0.0f;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
	A[i*n + j] += L[i*n + k]*U[k*n + j];

  double deviation1 = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      deviation1 += (refA[i*n+j] - A[i*n+j])*(refA[i*n+j] - A[i*n+j]);
    }
  }
  deviation1 /= (double)(n*n);
  if (isnan(deviation1) || (deviation1 > 1.0e-2)) {
    printf("ERROR: LU is not equal to A (deviation1=%e)!\n", deviation1);
    exit(1);
  }

  // Second, verifying that L is lower triangular
  double deviation2 = 0.0;
  for (int i = 0; i < n; i++)
    for (int j = i+1; j < n; j++)
      deviation2 += L[i*n+j]*L[i*n+j];
  if (isnan(deviation2) || (deviation2 > 1.0e-6)) {
    printf("ERROR: L is not lower triangular (deviation2=%e)!\n", deviation2);
    exit(1);
  }

  // Second, verifying that L is unit lower triangular
  double deviation3 = 0.0;
  for (int i = 0; i < n; i++)
    deviation3 += (1.0-L[i*n+i])*(1.0-L[i*n+i]);
  if (isnan(deviation3) || (deviation3 > 1.0e-6)) {
    printf("ERROR: L does not have 1s on main diagonal (deviation3=%e)!\n", deviation3);
    exit(1);
  }

  // Fourth, verifying that U is upper triangular
  double deviation4 = 0.0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < i; j++)
      deviation4 += U[i*n+j]*U[i*n+j];
  if (isnan(deviation4) || (deviation4 > 1.0e-2)) {
    printf("ERROR: U is not upper triangular (deviation4=%e)!\n", deviation4);
    exit(1);
  }

#ifdef VERBOSE
  printf("\nL:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", L[i*n+j]);
    printf("\n");
  }

  printf("\nU:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", U[i*n+j]);
    printf("\n");
  }

  printf("\nLU:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", A[i*n+j]);
    printf("\n");
  }

  printf("\nA:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", refA[i*n+j]);
    printf("\n");
  }

  printf("deviation1=%e\n", deviation1);
  printf("deviation2=%e\n", deviation2);
  printf("deviation3=%e\n", deviation3);
  printf("deviation4=%e\n", deviation4);
#endif

}

int main(const int argc, const char** argv) {

  // Problem size and other parameters
  const int n=128;
  const int nMatrices=10000;
  const double HztoPerf = 1e-9*2.0/3.0*double(n*n*n)*nMatrices;

  const size_t containerSize = sizeof(float)*n*n+64;
  char* dataA = (char*) _mm_malloc(containerSize*nMatrices, 64);
  char* dataL = (char*) _mm_malloc(containerSize*nMatrices, 64);
  float referenceMatrix[n*n];

  // Initialize matrices
#pragma omp parallel for
  for (int m = 0; m < nMatrices; m++) {
    float* matrix = (float*)(&dataA[m*containerSize]);
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            matrix[i*n+j] = (float)(i*n+j);
            sum += matrix[i*n+j];
        }
        sum -= matrix[i*n+i];
        matrix[i*n+i] = 2.0f*sum;
    }
    matrix[n*n] = 0.0f; // Touch just in case
  }
  referenceMatrix[0:n*n] = ((float*)dataA)[0:n*n];
  
  // Perform benchmark
  printf("LU decomposition of %d matrices of size %dx%d on %s...\n\n", 
	 nMatrices, n, n,
#ifndef __MIC__
	 "CPU"
#else
	 "MIC"
#endif
	 );

  double rate = 0, dRate = 0; // Benchmarking data
  const int nTrials = 10;
  const int skipTrials = 3; // First step is warm-up on Xeon Phi coprocessor
  printf("\033[1m%5s %10s %8s\033[0m\n", "Trial", "Time, s", "GFLOP/s");
  for (int trial = 1; trial <= nTrials; trial++) {

    const double tStart = omp_get_wtime(); // Start timing
#pragma omp parallel for
    for (int m = 0; m < nMatrices; m++) {
      float* matrixA = (float*)(&dataA[m*containerSize]);
      float* matrixL = (float*)(&dataL[m*containerSize]);
      LU_Doolittle(n, matrixA, matrixL);
    }
    const double tEnd = omp_get_wtime(); // End timing

    if (trial == 1) VerifyResult(n, (float*)(&dataA[0]), (float*)(&dataL[0]), referenceMatrix);

    if (trial > skipTrials) { // Collect statistics
      rate  += HztoPerf/(tEnd - tStart); 
      dRate += HztoPerf*HztoPerf/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e %8.2f %s\n", 
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

  _mm_free(dataA);
  _mm_free(dataL);

}
