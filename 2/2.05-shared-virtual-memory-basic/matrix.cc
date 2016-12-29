/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/2/2.05-shared-virtual-memory-basic/matrix.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cstdlib>

#define m 10
#define n 100000

double sum;
double b[n];
double* A;

// Multiplies a matrix by a vector, and sums the result 
void multiply_then_add() {
  for ( int i = 0 ; i < m ; i++)
    for ( int j = 0 ; j < n ; j++)
      sum += A[i*n+j] * b[j];
}

int main(){
  A = (double*) malloc(sizeof(double)*n*m);
  // Cilk Plus array notation
  A[0:n*m]=1.0/(double)n;
  b[:]=1.0;
  sum =0.0;
  printf("Running the matrix-vector multiplication\n");
  multiply_then_add();
  printf("sum = %f (should be %f)\n", sum, (double) m);
}
