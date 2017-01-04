


#include <cstdio>
#include <cstdlib>
#include <cmath>


double recursive_integral(const long minIter, const long maxIter, const double dx) {  
  double integral = 0.0; 
  const long chunk = maxIter - minIter;
  const long chunk_threshold = 1000;
  if(chunk < chunk_threshold) {
    for(long i = minIter; i < maxIter; i++) {
      // integrate using the midpoint rectangle method 
      double x = dx*(double(i) + 0.5);
      integral += 1.0/sqrt(x) * dx;
    }

  } else {
    double child_integral = 0.0;
#pragma omp task shared(child_integral)
    {
      child_integral = recursive_integral(minIter, minIter+chunk/2L, dx);
    }
    integral = recursive_integral(minIter+chunk/2L, maxIter, dx);
#pragma omp taskwait
    integral+= child_integral;
  }

  return integral;
}


int main(){
  const double x_upper_bound = 1.0;
  const double x_lower_bound = 0.0;

  const long nSteps = 1000000000; 
  const double dx = (x_upper_bound - x_lower_bound)/nSteps;
  
  double integral = 0.0;
  
#pragma omp parallel 
  {
#pragma omp master  
    integral = recursive_integral(0, nSteps, dx);  
  }
  const double analytical_result = (sqrt(x_upper_bound) - sqrt(x_lower_bound))*2;
  const double numerical_result = integral;
  printf("Result = %.8f (Should be %.8f; err = %.8f)\n", numerical_result, analytical_result, analytical_result-numerical_result);
}
