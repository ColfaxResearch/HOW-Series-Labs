/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/5/5.02-vtune/worker-offload.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cmath>
#include <cstdio>

__attribute__((target(mic))) double RelativeNormOfDifference(const int n, const double* v1, const double* v2) {
  // Calculates ||v1 - v2|| / (||v1|| + ||v2||)
  double norm2 = 0.0;
  double v1sq = 0.0;
  double v2sq = 0.0;
#pragma vector aligned
  for (int i = 0; i < n; i++) {
    norm2 += (v1[i] - v2[i])*(v1[i] - v2[i]);
    v1sq  += v1[i]*v1[i];
    v2sq  += v2[i]*v2[i];
  }
  return sqrt(norm2/(v1sq+v2sq));
}

__attribute__((target(mic))) int IterativeSolver(const int n, const double* M, const double* b, double* x, const double minAccuracy) {
  // Iteratively solves the equation Mx=b with accuracy of at least minAccuracy
  // using the Jacobi method
  double accuracy;
  double bTrial[n] __attribute__((aligned(64)));
  x[0:n] = 0.0; // Initial guess
  int iterations = 0;
  do {
    iterations++;
    // Jacobi method
    for (int i = 0; i < n; i++) {
      double c = 0.0;
#pragma vector aligned
      for (int j = 0; j < n; j++)
	c += M[i*n+j]*x[j];
      x[i] = x[i] + (b[i] - c)/M[i*n+i];
    }

    // Verification
    bTrial[:] = 0.0;
    for (int i = 0; i < n; i++) {
#pragma vector aligned
      for (int j = 0; j < n; j++)
	bTrial[i] += M[i*n+j]*x[j];
    }
    accuracy = RelativeNormOfDifference(n, b, bTrial);

  } while (accuracy > minAccuracy);
  return iterations;
}
