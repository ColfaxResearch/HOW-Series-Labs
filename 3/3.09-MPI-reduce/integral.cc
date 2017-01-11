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
  for(long i = 0; i < nSteps; i++) {
    //integral using the rectangle method 
    const double x = x_lower_bound + dx*(double(i) + 0.5);
    integral += 1.0/sqrt(x) * dx;
  }

  const double analytical_result = (sqrt(x_upper_bound) - sqrt(x_lower_bound))*2;
  const double numerical_result = integral;

  // Report results
  printf("Result = %.8f (Should be %.8f; err = %.8f)\n", numerical_result, analytical_result, analytical_result-numerical_result);

  MPI_Finalize(); 
}


