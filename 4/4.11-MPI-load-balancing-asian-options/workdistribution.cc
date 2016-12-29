/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.11-MPI-load-balancing-asian-options/workdistribution.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <omp.h>
#include <mpi.h>
#include "options.h"

void ComputeOnAllNodes(const int nOptions, // Number of option parameters to price
		       const OptionType* const option, // Array of option parameters
		       PayoffType* payoff, // Array of option parameters
		       const int mpiWorldSize, // Size of MPI world for load distribution
		       int* rankTypes, // Types of each node (0=CPU, 1=MIC)
		       const int myRank,        // My ID
		       double & comp_time, // For Reporting: the computation time
		       int & optioncount // For Reporting: Number of options processed  
		       ) {



  //Calculating workload share based on the rank 
  const double optionsPerProcess = double(nOptions)/double(mpiWorldSize);
  const int myFirstOption = int(optionsPerProcess*(myRank));
  const int myLastOption  = int(optionsPerProcess*(myRank+1));
  /////////////////////////////////

  const double comp_start=omp_get_wtime();
  // Static, even load distribution: assign options to ranks
  for (int i = myFirstOption; i < myLastOption; i++) {
    ComputeOptionPayoffs(option[i], payoff[i]);
    optioncount++;
  }
  const double comp_end=omp_get_wtime();

  // Collect results from all processes to reportingRank
  if (myRank == reportingRank) {
    MPI_Reduce(MPI_IN_PLACE, (float*)payoff, 4*nOptions, MPI_FLOAT, MPI_SUM, reportingRank, MPI_COMM_WORLD);
  } else {
    MPI_Reduce((float*)payoff, (float*)payoff, 4*nOptions, MPI_FLOAT, MPI_SUM, reportingRank, MPI_COMM_WORLD);
  }

  comp_time = comp_end - comp_start;
}
