/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/3/3.09-MPI-reduce/solutions/instruction-02/integral.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <mpi.h>

int main(int argc, char* argv[]){

  // Set up MPI environment
  int ret = MPI_Init(&argc,&argv);
  if (ret != MPI_SUCCESS) {
    printf("error: could not initialize MPI\n");
    MPI_Abort(MPI_COMM_WORLD, ret);
  }
  
  // determine name, rank and total number of processes.
  int worldSize, rank;
  MPI_Status stat;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Compute the integral
  const double x_lower_bound = 0.0;
  const double x_upper_bound = 1.0;
  const long nSteps = 100000000;
  const double dx = (x_upper_bound - x_lower_bound)/nSteps;

  double integral = 0.0;
  // Perform only a fraction of the work
  const long iMin = long((rank  ) * double(nSteps)/double(worldSize));
  const long iMax = long((rank+1) * double(nSteps)/double(worldSize));

  for(long i = iMin; i < iMax; i++) {
    //integral using the rectangle method 
    const double x = x_lower_bound + dx*(double(i) + 0.5);
    integral += 1.0/sqrt(x) * dx;
  }

  double total_integral = 0.0;
  MPI_Reduce(&integral, &total_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  const double analytical_result = (sqrt(x_upper_bound) - sqrt(x_lower_bound))*2;
  const double numerical_result = total_integral;

  // Report results
  if (rank == 0) {
    printf("Result = %.8f (Should be %.8f; err = %.8f)\n", numerical_result, analytical_result, analytical_result-numerical_result);
  }

  MPI_Finalize(); 
}


