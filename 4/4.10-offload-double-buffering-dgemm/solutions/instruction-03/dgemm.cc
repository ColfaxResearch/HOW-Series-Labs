


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <omp.h>

void Calculate_dgemm(const int n, const int nMatrices, double* A_arr, double* B_arr, double* C_arr, double &offload_time, double& computation_time) {

  double t_offl = 0.0, t_comp = 0.0, t0, t1;
  
  double* A_buff1 = &A_arr[0];    double* B_buff1 = &B_arr[0];    double* C_buff1 = &C_arr[0];
  double* A_buff2 = &A_arr[n*n];  double* B_buff2 = &B_arr[n*n];  double* C_buff2 = &C_arr[n*n];

  //First set: They are exceptions ///////

  t0 = omp_get_wtime();

#pragma offload_transfer target(mic:0) in(A_arr[0:n*n]: into (A_buff1[0:n*n])) in(B_arr[0:n*n] : into (B_buff1[0:n*n]))

  t1=omp_get_wtime();
  t_offl += t1-t0;

#pragma offload target(mic:0) in(A_buff1: length(0) alloc_if(0) free_if(0)) in(B_buff1: length(0) alloc_if(0) free_if(0)) in(C_buff1: length(0) alloc_if(0) free_if(0))  signal(A_buff1)
  {
    t0 = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, n, n, 1.0, A_buff1, n, B_buff1, n, 0.0, C_buff1, n);
    t1 = omp_get_wtime();
    t_comp += t1 - t0;
  }

  t0 = omp_get_wtime();
#pragma offload_transfer target(mic:0) in(A_arr[n*n:n*n]: into (A_buff2[0:n*n])) in(B_arr[n*n:n*n] : into (B_buff2[0:n*n]))
  t1 = omp_get_wtime();
  t_offl += t1-t0;

#pragma offload_wait target(mic:0) wait(A_buff1)


  // The body of the calculations/////////
  double* A_buff_trans = A_buff1; double* B_buff_trans = B_buff1; double* C_buff_trans = C_buff1;
  double* A_buff_calc  = A_buff2; double* B_buff_calc  = B_buff2; double* C_buff_calc  = C_buff2;
  for(int i = 1; i < nMatrices-1; i++) {
    double* A_trans = &A_arr[(i+1)*n*n]; // We send the next data set
    double* B_trans = &B_arr[(i+1)*n*n]; // We send the next data set
    double* C_trans = &C_arr[(i-1)*n*n]; // We recieve the previous result

#pragma offload target(mic:0) in(A_buff_calc: length(0) alloc_if(0) free_if(0)) in(B_buff_calc: length(0) alloc_if(0) free_if(0)) in(C_buff_calc: length(0) alloc_if(0) free_if(0)) signal(A_buff_calc)
    {
      t0 = omp_get_wtime();
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		  n, n, n, 1.0, A_buff_calc, n, B_buff_calc, n, 0.0, C_buff_calc, n);
      t1 = omp_get_wtime();
      t_comp += t1 - t0;
    }

    t0 = omp_get_wtime();
#pragma offload_transfer target(mic:0) in(A_trans[0:n*n]: into (A_buff_trans[0:n*n])) in(B_trans[0:n*n] : into (B_buff_trans[0:n*n]))
#pragma offload_transfer target(mic:0) out(C_buff_trans[0:n*n]: into (C_trans[0:n*n]))  
    t1=omp_get_wtime();
    t_offl += t1-t0;

#pragma offload_wait target(mic:0) wait(A_buff_calc)

    // Swapping Buffers
    if(i%2==1) {
      A_buff_trans = A_buff2; B_buff_trans = B_buff2; C_buff_trans = C_buff2;
      A_buff_calc  = A_buff1; B_buff_calc  = B_buff1; C_buff_calc  = C_buff1;
    } else {
      A_buff_trans = A_buff1; B_buff_trans = B_buff1; C_buff_trans = C_buff1;
      A_buff_calc  = A_buff2; B_buff_calc  = B_buff2; C_buff_calc  = C_buff2;
    }    
  }
  ////////////////////////////////////////

  //Last two sets: Again they are exceptions /////////
  double* C_trans = &C_arr[(nMatrices-2)*n*n]; // Second to last result


#pragma offload target(mic:0) in(A_buff_calc: length(0) alloc_if(0) free_if(0)) in(B_buff_calc: length(0) alloc_if(0) free_if(0)) in(C_buff_calc: length(0) alloc_if(0) free_if(0))  signal(A_buff_calc)
  {
    t0 = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, n, n, 1.0, A_buff_calc, n, B_buff_calc, n, 0.0, C_buff_calc, n);
    t1 = omp_get_wtime();
    t_comp += t1 - t0;
  }

  t0 = omp_get_wtime();
#pragma offload_transfer target(mic:0) out(C_buff_trans[0:n*n]: into (C_trans[0:n*n]))  
  t1=omp_get_wtime();
  t_offl += t1-t0;

#pragma offload_wait target(mic:0) wait(A_buff_calc)

  C_trans = &C_arr[(nMatrices-1)*n*n]; // Last result
  C_buff_trans = (nMatrices%2==1) ? C_buff1: C_buff2;

  t0 = omp_get_wtime();
#pragma offload_transfer target(mic:0) out(C_buff_trans[0:n*n]: into (C_trans[0:n*n]))  
  t1=omp_get_wtime();
  t_offl += t1 - t0;


  offload_time = t_offl;
  computation_time = t_comp;
  /////////////////////////// 
}

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
    for (int i = 0; i < n*n; i++) {
      A[i] =  (double)i*(double)j/(double)(n*n);
      B[i] = -(double)i*(double)j/(double)(n*n);
      C[i] = 0.0;
    }
  }
  create_reference(n, nMatrices, A_arr, B_arr, C_ref);


  // Using the pointers to the first two blocks as the pointers matched on the MIC for the buffers 
  double* A_buff1 = &A_arr[0];    double* B_buff1 = &B_arr[0];    double* C_buff1 = &C_arr[0];
  double* A_buff2 = &A_arr[n*n];  double* B_buff2 = &B_arr[n*n];  double* C_buff2 = &C_arr[n*n];
  
#pragma offload_transfer target(mic:0) \
  in(A_buff1: length(n*n) alloc_if(1) free_if(0) align(4096)) \
  in(B_buff1: length(n*n) alloc_if(1) free_if(0) align(4096)) \
  in(C_buff1: length(n*n) alloc_if(1) free_if(0) align(4096)) \
  in(A_buff2: length(n*n) alloc_if(1) free_if(0) align(4096)) \
  in(B_buff2: length(n*n) alloc_if(1) free_if(0) align(4096)) \
  in(C_buff2: length(n*n) alloc_if(1) free_if(0) align(4096))  

  printf("\033[1m%5s %12s %12s %12s %12s %12s %12s %12s\033[0m\n", "Trial", "Comm. time", "Comp. time", "Total time", "MIC perf.", "Bandwidth", "Masked", "Total perf");
  printf("\033[1m%5s %12s %12s %12s %12s %12s %12s %12s\033[0m\n", "", "(s)", "(s)", "(s)", "(GFLOP/s)", "(GB/s)", "frac.(%)", "(GFLOP/s)");

  for (int trial = 1; trial <= nTrials; trial++) {

    double offload_time, computation_time;

    const double tStart = omp_get_wtime();
    Calculate_dgemm(n, nMatrices, A_arr, B_arr, C_arr, offload_time, computation_time);
    const double tEnd = omp_get_wtime();

    if ( trial > skipTrials) { // First two iterations are slow on Xeon Phi; exclude them
      rate  += HztoPerf/(tEnd - tStart); 
      dRate += HztoPerf*HztoPerf/((tEnd - tStart)*(tEnd-tStart)); 
    }

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

    printf("%5d %12.4f %12.4f %12.4f %12.1f %12.3f %12.1f %12.1f %s\n", 
	   trial, offload_time, computation_time, (tEnd-tStart), HztoPerf/computation_time, HztoGBs/offload_time, (1.0 - fabs(offload_time-computation_time)/(tEnd-tStart))*100.0, HztoPerf/(tEnd-tStart), (trial<=skipTrials?"*":""));
    fflush(stdout);
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
