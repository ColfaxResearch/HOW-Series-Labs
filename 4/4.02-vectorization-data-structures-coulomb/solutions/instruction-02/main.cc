/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.02-vectorization-data-structures-coulomb/solutions/instruction-02/main.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <omp.h>
#include <cmath>
#include "rngutil.h"

struct Charge_Distribution { 
    // This data layout permits effective vectorization of Coulomb's law application
    const int m; // Number of charges
    float * x; // Array of x-coordinates of charges
    float * y; // ...y-coordinates...
    float * z; // ...etc.
    float * q; // Charges
};

// This version vectorizes better thanks to unit-stride data access
void CalculateElectricPotential(
        const int m,    // Number of charges
        const Charge_Distribution & chg, // Charge distribution (structure of arrays)
        const float Rx, const float Ry, const float Rz, // Observation point
        float & phi  // Output: electric potential
        ) {
    phi=0.0f;
    for (int i=0; i<chg.m; i++)  {
        // Unit stride: (&chg.x[i+1] - &chg.x[i]) == sizeof(float)
        const float dx=chg.x[i] - Rx;
        const float dy=chg.y[i] - Ry;
        const float dz=chg.z[i] - Rz;
        phi -= chg.q[i] / sqrtf(dx*dx+dy*dy+dz*dz);
    }
}

int main(int argv, char* argc[]){
    const size_t n=1<<11;
    const size_t m=1<<11;
    const int nTrials=10;
    const int skipTrials=2;

    Charge_Distribution chg  = { .m = m };
    chg.x = (float*)malloc(sizeof(float)*m);
    chg.y = (float*)malloc(sizeof(float)*m);
    chg.z = (float*)malloc(sizeof(float)*m);
    chg.q = (float*)malloc(sizeof(float)*m);
    float* potential = (float*) malloc(sizeof(float)*n*n);

    // Initializing array of charges
    RNGutil rng[omp_get_max_threads()];
    printf("Initialization...");
#pragma omp parallel for schedule(guided)
    for (size_t i=0; i<n; i++) {
      const int iThread = omp_get_thread_num();
      chg.x[i] = rng[iThread].Next();
      chg.y[i] = rng[iThread].Next();
      chg.z[i] = rng[iThread].Next();
      chg.q[i] = rng[iThread].Next();
    }
    printf(" complete.\n");

    printf("\033[1m%5s %10s %8s\033[0m\n", "Trial", "Time, s", "GFLOP/s");
    double perf=0.0, dperf=0.0;
    for (int t=1; t<=nTrials; t++){
        potential[0:n*n]=0.0f;
        const double t0 = omp_get_wtime();
#pragma omp parallel for schedule(guided)
        for (int j = 0; j < n*n; j++) {
            const float Rx = (float)(j % n);
            const float Ry = (float)(j / n);
            const float Rz = 0.0f;
            CalculateElectricPotential(m, chg, Rx, Ry, Rz, potential[j]);
        }
        const double t1 = omp_get_wtime();

	const double HztoPerf = 10.0*1e-9*double(n*n)*double(m);
	if (t > skipTrials) {
	  perf += HztoPerf/(t1-t0);
	  dperf += HztoPerf*HztoPerf/((t1-t0)*(t1-t0));
	}

	printf("%5d %10.3e %8.1f %s\n", 
	       t, (t1-t0), HztoPerf/(t1-t0), (t<=skipTrials?"*":""));
	fflush(stdout);
    }
    perf/=(double)(nTrials-skipTrials); 
    dperf=sqrt(dperf/(double)(nTrials-skipTrials)-perf*perf);
    printf("-----------------------------------------------------\n");
    printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	   "Average performance:", "", perf, dperf);
    printf("-----------------------------------------------------\n");
    printf("* - warm-up, not included in average\n\n");
    free(chg.x);
    free(chg.y);
    free(chg.z);
    free(potential);
}
