#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>

double recursive_integral(const int minIter, const int maxIter, const double dx, const double offset) {  
  double integral = 0.0; 
  const int chunk = maxIter - minIter;
  const int chunk_threshold = 1000;
  if(chunk < chunk_threshold) {
    for(int i = minIter; i < maxIter; i++) {
      // integrate using the midpoint rectangle method 
      double x = dx*(double(i) + 0.5) + offset;
      integral += 1.0/sqrt(x) * dx;
    }

  } else {
    double child_integral = recursive_integral(minIter, minIter+chunk/2L, dx, offset);  
    integral = recursive_integral(minIter+chunk/2L, maxIter, dx, offset);
    integral+= child_integral;
  }

  return integral;
}


int main(){
  const double x_upper_bound = 1.0;
  const double x_lower_bound = 0.0;

  const int nSteps = 1000000000; 
  const double dx = (x_upper_bound - x_lower_bound)/nSteps;
  
  const int nTrials = 10; 

  for(int trial = 0; trial < nTrials; trial++) {
    double integral = 0.0;
    
    // Function to be parallelized
    const double t0 = omp_get_wtime();
    integral = recursive_integral(0, nSteps, dx, double(trial));  
    const double t1 = omp_get_wtime();
  
    const double analytical_result = (sqrt(x_upper_bound + double(trial)) - sqrt(x_lower_bound + double(trial)))*2;
    const double numerical_result = integral;
    printf("Bounds (%2f:%2f) = %.8f (Should be %.8f; err = %.8f)  Time = %f ms\n", x_lower_bound + double(trial), x_upper_bound + double(trial), numerical_result, analytical_result, analytical_result-numerical_result, (t1-t0)*1000.0);
  }
}
