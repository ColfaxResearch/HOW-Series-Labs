


#include <omp.h>
#include <mpi.h>
#include "options.h"

void DistributeWork(
		    const int nOptions, // Number of option parameters to price
		    const OptionType* const option, // Array of option parameters
		    const int mpiWorldSize // Size of MPI world for load distribution
		    ) {
  
  int option_index = 0; 
  MPI_Status mpiStatus;

  // Distribute option parameters to work on, one by one
  int terminates_sent = 0;
  while(terminates_sent < mpiWorldSize) { //waiting to send terminate value to all
    int target_source;
    // Wait for a request for work
    MPI_Recv(&target_source,1,MPI_INT,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&mpiStatus);
    if(option_index < nOptions) {
      // Assign the next option to work on
      MPI_Send(&option_index,1,MPI_INT,target_source,1,MPI_COMM_WORLD);
      option_index++;
    } else {
      // All work completed; send termination signal
      MPI_Send(&terminate_val,1,MPI_INT,target_source,1,MPI_COMM_WORLD);
      terminates_sent++;
    }
  }

}

void ReceiveWork(
		 const OptionType* const option, // Array of option parameters
		 PayoffType* payoff, // Array of option parameters
		 const int myRank,        // My ID
		 int & optioncount // For Reporting: Number of options processed  
		 ) {

  int option_index = 0; 
  MPI_Status mpiStatus;

  bool terminate = false;
  // Request work
  MPI_Send(&myRank,1,MPI_INT,0,1,MPI_COMM_WORLD);
  while(!terminate) {
    // Get the next option to process
    MPI_Recv(&option_index,1,MPI_INT,0,1,MPI_COMM_WORLD,&mpiStatus);
    if(option_index == terminate_val) {
      terminate = true;
    } else {
      // Process the assigned option
      ComputeOptionPayoffs(option[option_index], payoff[option_index]);
      optioncount++;
      MPI_Send(&myRank,1,MPI_INT,0,1,MPI_COMM_WORLD);
    }
  }

}

void ComputeOnAllNodes(const int nOptions, // Number of option parameters to price
		       const OptionType* const option, // Array of option parameters
		       PayoffType* payoff, // Array of option parameters
		       const int mpiWorldSize, // Size of MPI world for load distribution
		       int* rankTypes, // Types of each node (0=CPU, 1=MIC)
		       const int myRank,        // My ID
		       double & comp_time, // For Reporting: the computation time
		       int & optioncount // For Reporting: Number of options processed  
		       ) {

  MPI_Status mpiStatus;

  if(myRank == reportingRank) {
    const int nThreads = omp_get_max_threads();
    omp_set_nested(1);
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
      {
	DistributeWork(nOptions, option, mpiWorldSize);
	comp_time = 0.0;
      }
#pragma omp section
      {
	omp_set_num_threads(nThreads-1);
	const double comp_start = omp_get_wtime();
	ReceiveWork(option, payoff, myRank, optioncount);
	const double comp_end = omp_get_wtime();
	comp_time = comp_end - comp_start;
      }
    }
  } else {
    const double comp_start = omp_get_wtime();
    ReceiveWork(option, payoff, myRank, optioncount);
    const double comp_end = omp_get_wtime();
    comp_time = comp_end - comp_start;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (myRank == reportingRank) {
    MPI_Reduce(MPI_IN_PLACE, (float*)payoff, 4*nOptions, MPI_FLOAT, MPI_SUM, reportingRank, MPI_COMM_WORLD);
  } else {
    MPI_Reduce((float*)payoff, (float*)payoff, 4*nOptions, MPI_FLOAT, MPI_SUM, reportingRank, MPI_COMM_WORLD);
  }

}


