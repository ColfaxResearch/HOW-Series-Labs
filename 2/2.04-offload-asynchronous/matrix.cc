/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/2/2.04-offload-asynchronous/matrix.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cstdlib>

int main(){

  const int iterMax = 3;
  const int m=10, n=100000;
  double b[n], c_target[m], c_host[m];
  double * A = (double*) malloc(sizeof(double)*n*m);

  // Cilk Plus array notation
  A[0:n*m]=1.0/(double)n;

  for ( int iter = 0; iter < iterMax ; iter++) {
    b[:] = (double) iter;
    c_target[:]=0; // results calculated on the coprocessor
    c_host[:]=0; // results calculated on the host

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

    double norm = 0.0;
    for ( int i = 0 ; i < m ; i++)
      norm += (c_target[i] - c_host[i])*(c_target[i] - c_host[i]);
    if (norm > 1e-10)
      printf("ERROR! In iter %i, the norm is %e:\n", iter, norm);
    else
      printf("Yep, iter %i is good!\n", iter);
  }
}
