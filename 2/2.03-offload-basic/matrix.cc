#include <cstdio>
#include <cstdlib>

int main(){

    const int m=10, n=100000;
    const int maxIter = 5;
    double c[m], b[n];
    double * A = (double*) malloc(sizeof(double)*n*m);

    // Cilk Plus array notation
    A[0:n*m]=1.0/(double)n;
    b[:]=0.0;
    c[:]=0;

    printf("Running the matrix-vector multiplication\n");
    for ( int iter = 0; iter < maxIter ; iter++) {
        b[:] = b[:] + 1.0;
        for ( int i = 0 ; i < m ; i++)
                for ( int j = 0 ; j < n ; j++)
                        c[i] += A[i*n+j] * b[j];

    }

    printf("Checking the results...\n");
    double norm = 0.0;
    for ( int i = 0 ; i < m ; i++)
        norm += (c[i]-(maxIter+1)*maxIter/2.0)*(c[i]-(maxIter+1)*maxIter/2.0);

    if (norm > 1e-10)
        printf("Something is wrong... Norm is equal to %f\n", norm);
    else
        printf("Yup, we're good!\n");

}
