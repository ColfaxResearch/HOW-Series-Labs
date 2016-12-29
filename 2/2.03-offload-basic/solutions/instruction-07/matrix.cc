/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/2/2.03-offload-basic/solutions/instruction-07/matrix.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cstdlib>

__attribute__((target(mic))) void multiply(int n, int m, double* A, double* b, double* c){
        for ( int i = 0 ; i < m ; i++)
                for ( int j = 0 ; j < n ; j++)
                        c[i] += A[i*n+j] * b[j];
#ifdef __MIC__
        printf("Code is running on the MIC architecture.\n");
#else
        printf("Code is running on the host.\n");
#endif
}


int main(){

    const int m=10, n=100000;
    const int maxIter = 5;
    double b[n], c[m];
    double * A = (double*) malloc(sizeof(double)*n*m);

    // Cilk Plus array notation
    A[0:n*m]=1.0/(double)n;
    b[:]=0.0;
    c[:]=0;

    printf("Running the matrix-vector multiplication\n");
    for ( int iter = 0; iter < maxIter ; iter++) {
        b[:] = b[:] + 1.0;
#pragma offload target(mic) in(A[0:n*m]) optional
        multiply(n, m, A, b, c);
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
