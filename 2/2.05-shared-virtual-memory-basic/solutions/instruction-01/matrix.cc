


#include <cstdio>
#include <cstdlib>

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
  _Cilk_offload multiply_then_add();
  printf("sum = %f (should be %f)\n", sum, (double) m);
}
