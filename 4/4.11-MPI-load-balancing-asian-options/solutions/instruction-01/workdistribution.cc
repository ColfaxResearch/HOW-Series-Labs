#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <vector>
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

  // Create two new groups: all CPUs and all MICs:
  // 1. Create a list of ranks that are CPUs and MICs
  std::vector<int> cpuRanks, micRanks;
  for (int i = 0; i < mpiWorldSize; i++) {
    if (rankTypes[i] == 0) cpuRanks.push_back(i);
    if (rankTypes[i] == 1) micRanks.push_back(i);
  }
  // 2. Create MPI groups, one of CPUs and another of MICs
  MPI_Group newGroup, origGroup;
  MPI_Comm_group(MPI_COMM_WORLD, &origGroup);
  if (rankTypes[myRank] == 0) {
    MPI_Group_incl(origGroup, cpuRanks.size(), &cpuRanks[0], &newGroup);
  } else {
    MPI_Group_incl(origGroup, micRanks.size(), &micRanks[0], &newGroup);
  }
  // 3. Query my place in the new group
  int myGroupRank;
  MPI_Group_rank(newGroup, &myGroupRank);

  //Calculating workload share based on the rank 
  // Using workload balancing factor controlled by the environment variable OPTIONS_ALPHA
  double alpha = 1.0;
  if (getenv("OPTIONS_ALPHA") != NULL)
    alpha = atof(getenv("OPTIONS_ALPHA"));

  const int lastOptionForCPUs = int(alpha*nOptions*double(cpuRanks.size())/mpiWorldSize);
  int myFirstOption, myLastOption;
  if (rankTypes[myRank] == 0) {
    const double optionsPerProcess = double(lastOptionForCPUs)/double(cpuRanks.size());
    myFirstOption = int(optionsPerProcess*(myGroupRank));
    myLastOption  = int(optionsPerProcess*(myGroupRank+1));
  } else {
    const double optionsPerProcess = double(nOptions-lastOptionForCPUs)/double(micRanks.size());
    myFirstOption = lastOptionForCPUs + int(optionsPerProcess*(myGroupRank));
    myLastOption  = lastOptionForCPUs + int(optionsPerProcess*(myGroupRank+1));
  }
  // printf("Rank #%2d (#%2d in group) will process option %4d through %d\n", myRank, myGroupRank, myFirstOption, myLastOption);
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
