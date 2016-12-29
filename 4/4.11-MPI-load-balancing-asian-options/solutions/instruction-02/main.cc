/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.11-MPI-load-balancing-asian-options/solutions/instruction-02/main.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <mpi.h>
#include <mkl.h>
#include <mkl_vsl.h>
#include <cmath>
#include <omp.h>
#include <cstdio>
#include <unistd.h>

#include "options.h"

int main(int argc, char **argv) {

  FILE* result_file;
  //Initialize MPI
  MPI_Status mpiStatus;
  int myRank, mpiWorldSize;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiWorldSize);

  const int nOptions = 100;
  OptionType option[nOptions];
  PayoffType payoff[nOptions];

  if (myRank == reportingRank)
    result_file = fopen("pricing_result.txt", "w");
  

  const int nCalculations = 10;
  const int vMin = 0.05;
  const int vMax = 0.50;
  
  // Creating information on the types of nodes in the world
  // 0 = "CPU", 1 = "MIC"
  int rankTypes[mpiWorldSize];
  rankTypes[:] = 0;
  MPI_Barrier(MPI_COMM_WORLD);
#ifdef __MIC__
  rankTypes[myRank] = 1;
#endif
  MPI_Allreduce(MPI_IN_PLACE, &rankTypes, mpiWorldSize, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  double t0, t1; 
  double rate=0.0, dRate=0.0;
  double HztoPerf;
  const int skipTrials=2;
  for (int iCalc = 0; iCalc < nCalculations; iCalc++) {

    // Generate parameters to process
    const float strikeMin = 10.0f;
    const float strikeMax = 20.0f;
    for (int i = 0; i < nOptions; i++) {
      option[i].S = 15.3;
      option[i].K = strikeMin + (strikeMax - strikeMin)*(float)i/(float)(nOptions - 1);
      option[i].r = 0.08;
      option[i].v = vMin + (vMax - vMin)*(float)(iCalc)/(float)(nCalculations - 1);
      option[i].T = 1.0f;
      option[i].numIntervals = 30;
      option[i].numPaths = 1<<20;

      payoff[i].payoff_geom_put = 0.0f;
      payoff[i].payoff_arithm_put = 0.0f;
      payoff[i].payoff_geom_call = 0.0f;
      payoff[i].payoff_arithm_call = 0.0f;
    }

    // Send parameters to process to all workers

    MPI_Barrier(MPI_COMM_WORLD);

    if (myRank == reportingRank) {
      t0 = omp_get_wtime();
    } 

    HztoPerf=(double)option[0].numPaths*(double)nOptions*(double)option[0].numIntervals;

    double comp_time = 0.0; // For Reporting: the computation time
    int optioncount = 0; // For Reporting: Number of options processed  

    ComputeOnAllNodes(nOptions, // Number of option parameters to price
		      option, // Array of option parameters
		      payoff, // Array of computed payoffs
		      mpiWorldSize, // Size of MPI world for load distribution
		      rankTypes, // Number of MICs in the cluster
		      myRank,        // My ID
		      comp_time, // For Reporting: the computation time
		      optioncount // For Reporting: Number of options processed  
		      );

    if (myRank == reportingRank) {
      t1 = omp_get_wtime();
      // Reporting individual results
      for (int i = 0; i < mpiWorldSize; i++) {
	int rank_buff, optioncount_buff;
	double comp_t_buff;
	if (i != reportingRank) {
	  MPI_Recv(&rank_buff,1,MPI_INT,i,1,MPI_COMM_WORLD,&mpiStatus);
	  MPI_Recv(&comp_t_buff,1,MPI_DOUBLE,i,1,MPI_COMM_WORLD,&mpiStatus);
	  MPI_Recv(&optioncount_buff,1,MPI_INT,i,1,MPI_COMM_WORLD,&mpiStatus);
	  printf("Rank %d (%s): Computation %fs, Options Processed %d\n", 
		 rank_buff, (rankTypes[i]==1) ? "MIC" : "CPU", comp_t_buff, optioncount_buff);
	} else {
	  printf("Rank %d (%s): Computation %fs, Options Processed %d\n",
		 myRank, (rankTypes[i]==1) ? "MIC" : "CPU", comp_time, optioncount);
	}
      }

      printf("# Calculation %3d of %3d took %.3f seconds\n", iCalc+1, nCalculations, t1-t0);
      printf("# Net performance: %.2e random values/second\n\n", HztoPerf/(t1-t0));

      if (iCalc>=skipTrials) {
	rate += HztoPerf/(t1-t0);
	dRate += HztoPerf/(t1-t0)*HztoPerf/(t1-t0);
      }

      if(iCalc==0) {
	// Print calculation results:
	fprintf(result_file, "%8s %10s %10s %10s %10s\n", "Strike", "Gput", "Gcall", "Aput", "Acall");
	for (int i = 0; i < nOptions; i++) {
	  fprintf(result_file, "%8d %10.6f %10.6f %10.6f %10.6f\n", option[i].K,
		  payoff[i].payoff_geom_put,
		  payoff[i].payoff_geom_call,
		  payoff[i].payoff_arithm_put,
		  payoff[i].payoff_arithm_call);
	}
	fclose(result_file);
      }      
    } else {
      MPI_Send(&myRank,1,MPI_INT,0,1,MPI_COMM_WORLD);
      MPI_Send(&comp_time,1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
      MPI_Send(&optioncount,1,MPI_INT,0,1,MPI_COMM_WORLD);
    }
  }

  rate /= (nCalculations-skipTrials);
  dRate = sqrt(dRate/(nCalculations-skipTrials) - rate*rate);

  if (myRank == reportingRank) {
    printf("-----------------------------------------------------\n");
    printf("\033[1m%s %4s \033[42m%10.2e +- %.2e values/s\033[0m\n",
	   "Average performance:", "", rate, dRate);
    printf("-----------------------------------------------------\n");
  }

  MPI_Finalize();

  return 0;
}
