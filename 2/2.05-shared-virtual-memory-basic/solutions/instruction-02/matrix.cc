/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/2/2.05-shared-virtual-memory-basic/solutions/instruction-02/matrix.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cstdlib>
#include <offload.h>

#define m 10
#define n 100000

_Cilk_shared double sum;
_Cilk_shared double b[n];
_Cilk_shared double* _Cilk_shared A;

// Multiplies a matrix by a vector, and sums the result 
_Cilk_shared void multiply_then_add() {
#ifdef __MIC__  
  for ( int i = 0 ; i < m ; i++)
    for ( int j = 0 ; j < n ; j++)
      sum += A[i*n+j] * b[j];
  printf("Coprocessor %d finished the calculations!\n", _Offload_get_device_number());
#else
  printf("Offload failed!\n");
#endif
}

int main(){
  A = (_Cilk_shared double*) _Offload_shared_malloc(sizeof(double)*n*m);
  // Cilk Plus array notation
  A[0:n*m]=1.0/(double)n;
  b[:]=1.0;
  sum =0.0;
  printf("Running the matrix-vector multiplication\n");
  const int numDevices = _Offload_number_of_devices();
  for(int i=0; i<numDevices; i++)
    _Cilk_offload_to(i) multiply_then_add();
  printf("sum = %f (should be %f)\n", sum, (double) m*numDevices);
}
