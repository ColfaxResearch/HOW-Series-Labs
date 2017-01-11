#include <cstdio>
#include <cstdlib>

int main(){

  const int iterMax = 3;
  const int m=10, n=100000;
  double b[n], c_target[m], c_host[m];
  double * A = (double*) malloc(sizeof(double)*n*m);

  // Cilk Plus array notation
  A[0:n*m]=1.0/(double)n;

#pragma offload_transfer target(mic:1) in (A:length(n*m) free_if(0))

  for ( int iter = 0; iter < iterMax ; iter++) {
    b[:] = (double) iter;
    c_target[:]=0; // results calculated on the coprocessor
    c_host[:]=0; // results calculated on the host

#pragma offload target(mic:1) nocopy(A : free_if(iter==iterMax-1))	\
    signal(A)
    {
      // running the calculation on the coprocessor asynchronously
      for ( int i = 0 ; i < m ; i++)
        for ( int j = 0 ; j < n ; j++)
          c_target [i] += A[i*n+j] * b[j];
    }

    // the following code is running on the host asynchronously 
    for ( int i = 0 ; i < m ; i++)
      for ( int j = 0 ; j < n ; j++)
        c_host[i] += A[i*n+j] * b[j];

    // sync before proceeding
#pragma offload_wait target(mic:1) wait(A)

    double norm = 0.0;
    for ( int i = 0 ; i < m ; i++)
      norm += (c_target[i] - c_host[i])*(c_target[i] - c_host[i]);
    if (norm > 1e-10)
      printf("ERROR! In iter %i, the norm is %e:\n", iter, norm);
    else
      printf("Yep, iter %i is good!\n", iter);
  }
}
