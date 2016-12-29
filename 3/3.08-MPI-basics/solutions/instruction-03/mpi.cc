/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/3/3.08-MPI-basics/solutions/instruction-03/mpi.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <mpi.h>
#include <cstdio>

// Remember that you need to use mpiicpc to compile, and that
// you need an executable for each architectire (see lab 2.1-native)

// To run 6 processes on the CPU;
// mpirun -host localhost -np 6 {path_to_executable_on host}

// To run 2 processes on the CPU (including Boss) and 1 on the coprocessor;
// mpirun -host localhost -np 2 {path_to_executable_on_host} : \
//        -host mic0 -np 1 {path_to_executable_on_MIC}

int main(int argc, char** argv) {

    // Set up MPI environment
    int ret = MPI_Init(&argc,&argv);
    if (ret != MPI_SUCCESS) {
        printf("error: could not initialize MPI\n");
        MPI_Abort(MPI_COMM_WORLD, ret);
    }

    // determine name, rank and total number of processes.
    int worldSize, rank, namelen;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status stat;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &namelen);

    if (rank == 0) {
      printf("Hello World from Boss process, rank %d of %d running on %s\n",
	     rank, worldSize, name);

      // Iterating through the MPI_World and waiting for the Sends. Note that
      // we skip 0 because that is the boss process.
      for(int i = 1; i < worldSize; i++) {
	// Recv functions. Note that "i" is used for the source so that the
	// "Hello"s are received in order of rank. 
	MPI_Recv(&rank, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &stat);
	MPI_Recv(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, i, 1, MPI_COMM_WORLD, &stat);
	printf ("Received Hello from rank %d running on %s!\n", rank, name);
      }
    } else {
      // Send functions. Note that we specify source to be rank 0, so that the
      // message is sent to the Boss worker.
      MPI_Send(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
      MPI_Send(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
    
    // To achieve the same effect of ordered "Hello"s using tags
    // On boss process;
    // MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_TAG, i, MPI_COMM_WORLD, &stat);
    // On worker processes;
    // MPI_Send(&rank, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);

    MPI_Finalize();
}

