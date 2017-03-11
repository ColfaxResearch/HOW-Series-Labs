#include <cmath>
#pragma offload_attribute(push, target(mic))
#include <cstdio>
#pragma offload_attribute(pop)
#include <cstdlib>
#include <mkl.h>
#include <omp.h>

int main(int argc, char* argv[]) {

  long n;
  if (argc>1)
    n = atoi(argv[1]);
  else
    n = 320000000;

  const double HztoPerf = 1e-9*double(2*n*sizeof(double));

  // Run once on the host and once on the coprocessor
  for (int offloadFlag=0; offloadFlag<=1; offloadFlag++) {

#ifndef DNO_OFFLOAD
#pragma offload target(mic) if(offloadFlag) optional
#endif
    {
      double* A=(double*)_mm_malloc(n*sizeof(double), 64);
      double* B=(double*)_mm_malloc(n*sizeof(double), 64);

      const int nTrials=10;
      const int skipTrials=2;
      double rate=0, dRate=0;

      printf("\n\033[1mBenchmarking array copy.\033[0m\n");
      printf("Size of each array: %.3f GB\n",
	     double(n*sizeof(int))*1e-9);
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
      for (int i = 0; i < n; i++) {
	A[i] = 1.0;
	B[i] = 0.0;
      }

      printf("\033[1m%5s %10s %15s\033[0m\n", "Trial", "Time, s", "Perf, GB/s");

      for (int trial = 1; trial <= nTrials; trial++) {
	const double tStart = omp_get_wtime();
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	  B[i] = A[i];
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
      printf("\033[1m%s %4s \033[42m%10.2f +- %.2f GB/s\033[0m\n",
	     "Average performance:", "", rate, dRate);
      printf("-----------------------------------------------------\n");
      printf("* - warm-up, not included in average\n\n");


      _mm_free(A);
      _mm_free(B);

    }
  }

}
