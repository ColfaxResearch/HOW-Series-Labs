#include <cstdio>
#include <cstdlib>

int main(){
  const int n=5000; // This problem scales as n^3. 
                    // This value may need to be adjusted

  double * A = (double*) malloc(sizeof(double)*n*n);
  double * B = (double*) malloc(sizeof(double)*n*n);
  double * C = (double*) malloc(sizeof(double)*n*n);
  
  printf("Carrying out matrix-matrix multiplication\n");

  // Cilk Plus array notation
  for (int i = 0 ; i < n; i++) {
    for(int j = 0; j < n; j++) {
      A[i*n+j]=(double)i/(double)n;
      B[i*n+j]=(double)j/(double)n;
    }
  }


  C[0:n*n]=0.0;
  
  // C = A x B
  for ( int i = 0 ; i < n ; i++){
    for ( int j = 0 ; j < n ; j++) {
      for ( int k = 0 ; k < n ; k++) {
	C[i*n+j] += A[i*n+k]*B[k*n+j];
      }
    }
  }
  
  printf("Checking the results...\n");
  double norm = 0.0;
  for ( int i = 0 ; i < n ; i++)
    for ( int j = 0 ; j < n ; j++)
      norm += (C[i*n+j]-(double)(i*j)/(double)n)*(C[i*n+j]-(double)(i*j)/(double)n);
  
  if (norm > 1e-10)
    printf("Something is wrong... Norm is equal to %f\n", norm);
  else
    printf("Yup, we're good!\n");  
}
