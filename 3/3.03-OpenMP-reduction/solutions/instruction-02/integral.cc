#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>

int main(){
  const double x_upper_bound = 1.0;
  const double x_lower_bound = 0.0;

  const int nSteps = 1000000000; 
  const double dx = (x_upper_bound - x_lower_bound)/nSteps;

  const int nTrials = 10; 

  for(int trial = 0; trial < nTrials; trial++) {
    double integral = 0.0;

    const double t0 = omp_get_wtime();
#pragma omp parallel for reduction(+: integral)
    for(int i = 0; i < nSteps; i++) {
      // integrate using the midpoint rectangle method 
      const double x = x_lower_bound + dx*(double(i) + 0.5);
      integral += 1.0/sqrt(x) * dx;
    }
    const double t1 = omp_get_wtime();
    
    const double analytical_result = (sqrt(x_upper_bound) - sqrt(x_lower_bound))*2;
    const double numerical_result = integral;
    printf("Result = %.8f (Should be %.8f; err = %.8f)  Time = %f ms\n", numerical_result, analytical_result, analytical_result-numerical_result, (t1-t0)*1000.0);
  }
}
