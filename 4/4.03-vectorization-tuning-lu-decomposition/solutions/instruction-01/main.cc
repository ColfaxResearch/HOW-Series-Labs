#include <cmath>
#include <cstdio>
#include <omp.h>
#include <cassert>

void LU_decomp(const int n, float* const A) {
  // LU decomposition (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal
  for (int b = 0; b < n; b++) {
    // Strength reduction:
    const float recAbb = 1.0f/A[b*n + b];
    for (int i = b+1; i < n; i++) {
      A[i*n + b] = A[i*n + b]*recAbb;
      for (int j = b+1; j < n; j++) 
	A[i*n + j] -= A[i*n + b]*A[b*n + j];
    }
  }
}

void VerifyResult(const int n, float* LU, float* refA) {

  // Verifying that A=LU
  float A[n*n];
  float L[n*n];
  float U[n*n];
  A[:] = 0.0f;
  L[:] = 0.0f;
  U[:] = 0.0f;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++)
      L[i*n + j] = LU[i*n + j];
    L[i*n+i] = 1.0f;
    for (int j = i; j < n; j++)
      U[i*n + j] = LU[i*n + j];
  }
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
    //    exit(1);
  }

#ifdef VERBOSE
  printf("\n(L-D)+U:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", LU[i*n+j]);
    printf("\n");
  }

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
#endif

}

int main(const int argc, const char** argv) {

  // Problem size and other parameters
  const int n=128;
  const int nMatrices=10000;
  const double HztoPerf = 1e-9*2.0/3.0*double(n*n*n)*nMatrices;

  const size_t containerSize = sizeof(float)*n*n+64;
  char* dataA = (char*) _mm_malloc(containerSize*nMatrices, 64);
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
    matrix[(n-1)*n+n] = 0.0f; // Touch just in case
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
      LU_decomp(n, matrixA);
    }
    const double tEnd = omp_get_wtime(); // End timing

    if (trial == 1) VerifyResult(n, (float*)(&dataA[0]), referenceMatrix);

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

}
