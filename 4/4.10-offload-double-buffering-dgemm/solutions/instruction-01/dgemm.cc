


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <omp.h>

// Function to be modified for offloading
void Calculate_dgemm(const int n, const int nMatrices, double* A_arr, double* B_arr, double* C_arr, double &computation_time, double &offload_time) {
  double t_comp = 0.0, t_off = 0.0;

  for(int i = 0; i < nMatrices; i++) {
    double* A = &A_arr[i*n*n];
    double* B = &B_arr[i*n*n];
    double* C = &C_arr[i*n*n];

    const double t_0 = omp_get_wtime();
#pragma offload_transfer target(mic:0) \
  in(A: length(n*n) alloc_if(0) free_if(0))	\
  in(B: length(n*n) alloc_if(0) free_if(0)) 
    

#pragma offload target(mic:0) \
  in(A: length(0) alloc_if(0) free_if(0)) \
  in(B: length(0) alloc_if(0) free_if(0)) \
  in(C: length(0) alloc_if(0) free_if(0))
    {
      const double t_mic_0 = omp_get_wtime();
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      		  n, n, n, 1.0, A, n, B, n, 0.0, C, n);
      t_comp += omp_get_wtime() - t_mic_0;
    }


#pragma offload_transfer target(mic:0) \
  out(C: length(n*n) alloc_if(0) free_if(0))
    t_off += omp_get_wtime() - t_0;
  }
  t_off-= t_comp;

  computation_time = t_comp;
  offload_time = t_off;
}

// Function for verification. No change needed.
void create_reference(const int n, const int nMatrices, double* A_arr, double* B_arr, double* C_ref) {
  for(int i = 0; i < nMatrices; i++) {
    double* A = &A_arr[i*n*n];
    double* B = &B_arr[i*n*n];
    double* C = &C_ref[i*n*n];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, n, n, 1.0, A, n, B, n, 0.0, C, n);
  }
}

int main(int argc, char* argv[]) {

  const long n = 1<<10;
  const long nMatrices = 1<<3;
  
  double* A_arr=(double*)_mm_malloc(n*n*nMatrices*sizeof(double), 4096);
  double* B_arr=(double*)_mm_malloc(n*n*nMatrices*sizeof(double), 4096);
  double* C_arr=(double*)_mm_malloc(n*n*nMatrices*sizeof(double), 4096);
  double* C_ref=(double*)_mm_malloc(n*n*nMatrices*sizeof(double), 4096);

  const double HztoPerf = 1e-9*double(2*n*n*n)*double(nMatrices);
  const double HztoGBs = 3.0*double(nMatrices)*double(n*n)*8.0*1e-9;

  const int nTrials=10;
  const int skipTrials=2;
  double rate=0, dRate=0;

  printf("\n\033[1mBenchmarking DGEMM in the offload mode.\033[0m\n");
  printf("Problem size: %d matrices of size [%d x %d] (%.3f GB total)\n\n",
	 nMatrices, n, n, double(3L*n*n*sizeof(double))*1e-9);

  // Initializing data
#pragma omp parallel for
  for(int j = 0; j < nMatrices; j++) {
    double* A = &A_arr[j*n*n];
    double* B = &B_arr[j*n*n];
    double* C = &C_arr[j*n*n];    
    double* C_c = &C_ref[j*n*n];    
    for (int i = 0; i < n*n; i++) {
      A[i] =  (double)i*(double)(j+1)/(double)(n*n);
      B[i] = -(double)i*(double)(j+1)/(double)(n*n);
      C[i] = 0.0;
      C_c[i] = 0.0;
    }
  }
  create_reference(n, nMatrices, A_arr, B_arr, C_ref);
    
  //Memory retention
  for(int j = 0; j < nMatrices; j++) {
    double* A = &A_arr[j*n*n];
    double* B = &B_arr[j*n*n];
    double* C = &C_arr[j*n*n];    
#pragma offload_transfer target(mic:0) \
  in(A: length(n*n) alloc_if(1) free_if(0) align(4096)) \ 
  in(B: length(n*n) alloc_if(1) free_if(0) align(4096)) \
  in(C: length(n*n) alloc_if(1) free_if(0) align(4096)) 
  }
  
  printf("\033[1m%5s %12s %12s %12s %12s %12s %12s\033[0m\n", "Trial", "MIC time", "Comm. time", "Total time", "MIC perf.", "Bandwidth", "Total perf");
  printf("\033[1m%5s %12s %12s %12s %12s %12s %12s\033[0m\n", "", "(s)", "(s)", "(s)", "(GFLOP/s)", "(GB/s)", "(GFLOP/s)");


  for (int trial = 1; trial <= nTrials; trial++) {
    double computation_time, offload_time;

    const double tStart = omp_get_wtime();
    Calculate_dgemm(n, nMatrices, A_arr, B_arr, C_arr, computation_time, offload_time);
    const double tEnd = omp_get_wtime();

    // Verify result

    if(trial == 1) {
      for(int k = 0; k < nMatrices; k++) {
	double norm = 0.0;
	for(int i = 0; i <n*n; i++)
	  norm+= (C_arr[k*n*n+i]-C_ref[k*n*n+i])*(C_arr[k*n*n+i]-C_ref[k*n*n+i]);
	
	if(norm > 1e-9) {
	  printf("  ERROR on Matrix %d: norm = %.9f\n", k, norm);
	}
      }
    }

    if ( trial > skipTrials) { // First two iterations are slow on Xeon Phi; exclude them
      rate  += HztoPerf/(tEnd - tStart); 
      dRate += HztoPerf*HztoPerf/((tEnd - tStart)*(tEnd-tStart)); 
    }
    fflush(stdout);
    printf("%5d %12.4f %12.4f %12.4f %12.1f %12.3f %12.1f %s\n", 
	   trial, computation_time, offload_time, (tEnd-tStart), HztoPerf/computation_time, HztoGBs/offload_time, HztoPerf/(tEnd-tStart), (trial<=skipTrials?"*":""));

  }

  rate/=(double)(nTrials-skipTrials); 
  dRate=sqrt(dRate/(double)(nTrials-skipTrials)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.2f +- %.2f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");


  _mm_free(A_arr);
  _mm_free(B_arr);
  _mm_free(C_arr);

}
