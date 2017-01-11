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
