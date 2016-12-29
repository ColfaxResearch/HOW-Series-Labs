/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.04-threading-misc-histogram/main.cc,
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

void HistogramReference(const float* age, int* const group, const int n,
    const float group_width){
    // Plain (scalar, sequentual) algorithm for computing the reference histogram
    for (long i = 0; i < n; i++){
        const int j = (int) floorf( age[i] / group_width );
        group[j]++;
    }
}

void Histogram(const float* age, int* const group, const int n, const float group_width,
    const int m);

int main(int argv, char* argc[]){
    const size_t n=1L<<27L;
    const float max_age=99.999f;
    const float group_width=20.0f;
    const size_t m = (size_t) floorf(max_age/group_width + 0.1f);
    const int nTrials=10;
    const int skipTrials=2;

    float* age = (float*) _mm_malloc(sizeof(int)*n, 64);
    int group[m];
    int ref_group[m];

    // Initializing array of ages
    printf("Initialization..."); fflush(stdout);
    RNGutil rng;
    const int size = rng.Size();
    for (int i = 0; i < n; i += size) {
      rng.SetBlock(&age[i]);
      age[i:size] *= age[i:size]*max_age;
    }
    
    // Computing the "correct" answer
    ref_group[:]=0;
    HistogramReference(age, ref_group, n, group_width);
    printf(" done.\n"); fflush(stdout);

    printf("\033[1m%5s %10s %10s\033[0m\n", "Trial", "Time, s", "Values/s");
    double perf=0.0, dperf=0.0;
    for (int t=1; t<=nTrials; t++){
        group[:] = 0;

        const double t0 = omp_get_wtime();
        Histogram(age, group, n, group_width, m);
        const double t1 = omp_get_wtime();

	const double HztoPerf = double(n);
	if (t > skipTrials) {
	  perf += HztoPerf/(t1-t0);
	  dperf += HztoPerf*HztoPerf/((t1-t0)*(t1-t0));
	}

	printf("%5d %10.3e %10.2e %s\n", 
	       t, (t1-t0), HztoPerf/(t1-t0), (t<=skipTrials?"*":""));
	fflush(stdout);

        for (int i=0; i<m; i++) {
            if (fabs((double)(ref_group[i]-group[i])) > 1e-4*fabs((double)(ref_group[i]
                +group[i]))) {
                printf("Result is incorrect!\n");
                for (int i=0; i<m; i++) printf(" (%d vs %d)", group[i], ref_group[i]);
            }
        }
        fflush(stdout);
    }
    perf/=(double)(nTrials-skipTrials); 
    dperf=sqrt(dperf/(double)(nTrials-skipTrials)-perf*perf);
    printf("---------------------------------------------------------\n");
    printf("\033[1m%s %4s \033[42m%10.2e +- %.2e values/s\033[0m\n",
	   "Average performance:", "", perf, dperf);
    printf("---------------------------------------------------------\n");
    printf("* - warm-up, not included in average\n\n");

    _mm_free(age);
}
