


#include <cmath>
#include <cstdio>
#include <omp.h>

int main(int argv, char** argc){

  //The size range to be tested. Must be a power of 2 
  //for the current set-up
  const size_t minimum_size = 1L<<10;
  const size_t maximum_size = 1L<<33;

  //Number of mics to use (currently does nothing)
  int mics = 1;
  if (argv>1) mics = atoi(argc[1]);

  //Number of Trials
  int reps = 5;
  if (argv>2) reps = atoi(argc[2]);

  //Data alignment
  int align = 64;
  if (argv>3) align = atoi(argc[3]);

  omp_set_num_threads(mics);

  printf("\n\033[1mBenchmarking offoad bandwidth.\033[0m\n\n");

  const int skipTrials=1;

  for (size_t size = minimum_size; size <= maximum_size; size *= 2L){

    printf("\033[1m#%8s %6s %10s %10s %10s %10s %10s\033[0m\n", "Devices", "Trial", "Size", "Time in", "Time out", "Bandw in", "Bandw out");
    printf("\033[1m#%8s %6s %10s %10s %10s %10s %10s\033[0m\n", "", "", "(MiB)", "(s)", "(s)", "(GB/s)", "(GB/s)");

    double avgBWin = 0.0, dBWin = 0.0;
    double avgTin = 0.0, dTin = 0.0;
    double avgBWout = 0.0, dBWout = 0.0;
    double avgTout = 0.0, dTout = 0.0;

    char * p =(char*) _mm_malloc(size, align);
    for(int trial = 1; trial <= reps; trial++) {

      const double t_offload_start = omp_get_wtime();
#pragma offload_transfer target(mic) in(p: length(size))
      const double timein = omp_get_wtime() - t_offload_start;

      const double t_offload2_start = omp_get_wtime();
#pragma offload_transfer target(mic) out(p: length(size))
      const double timeout = omp_get_wtime() - t_offload2_start;

      if (trial > skipTrials) {
	const double bwin = size/timein*1e-9;;
	avgBWin += bwin;
	dBWin += bwin*bwin;
	avgTin += timein;
	dTin += timein*timein;

	const double bwout = size/timeout*1e-9;;
	avgBWout += bwout;
	dBWout += bwout*bwout;
	avgTout += timeout;
	dTout += timeout*timeout;
      }
      printf(" %8d %6d %10.3f %10.6f %10.6f %10.3f %10.3f %s\n",
	     mics, trial, (double)size/(double)(1<<20), timein, timeout, size/timein*1e-9, size/timeout*1e-9, (trial<=skipTrials?"*":""));
      fflush(stdout);
    }
    _mm_free(p);

    avgBWin /= (reps-skipTrials);
    dBWin = sqrt(dBWin/(reps-skipTrials) - avgBWin*avgBWin);
    avgTin /= (reps-skipTrials);
    dTin = sqrt(dTin/(reps-skipTrials) - avgTin*avgTin);

    avgBWout /= (reps-skipTrials);
    dBWout = sqrt(dBWout/(reps-skipTrials) - avgBWout*avgBWout);
    avgTout /= (reps-skipTrials);
    dTout = sqrt(dTout/(reps-skipTrials) - avgTout*avgTout);

    printf("-----------------------------------------------------------------------\n");
    printf("\033[1m%s %10.3f \033[42m%10.6f %10.6f\033[0m\n",
	   "Average latency:", (double)size/(double)(1<<20), avgTin, avgTout);
    printf("\033[1m%s %10.3f \033[42m ±%8.6f  ±%8.6f\033[0m\n",
	   "                ", (double)size/(double)(1<<20), dTin, dTout);
    printf("\033[1m%s %8.3f   %19s \033[42m%10.3f %10.3f\033[0m\n",
	   "Average bandwidth:", (double)size/(double)(1<<20), "", avgBWin, avgBWout);
    printf("\033[1m%s %8.3f   %19s \033[42m    ±%4.3f     ±%4.3f\033[0m\n",
	   "                  ", (double)size/(double)(1<<20), "", dBWin, dBWout);
    printf("-----------------------------------------------------------------------\n");
    printf("* - warm-up, not included in average\n\n");
    
    printf("\n");
  }
}
