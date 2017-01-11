#include <cmath>
#include <cstdio>

double RelativeNormOfDifference(const int n, const double* v1, const double* v2) {
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

int IterativeSolver(const int n, const double* M, const double* b, double* x, const double minAccuracy) {
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
