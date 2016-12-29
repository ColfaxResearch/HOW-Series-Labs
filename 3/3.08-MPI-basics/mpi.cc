/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/3/3.08-MPI-basics/mpi.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <mpi.h>
#include <cstdio>

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
    } else {
      printf ("Hello World from rank %d running on %s!\n", rank, name);
    }
    // Terminate MPI environment
    MPI_Finalize();
}


