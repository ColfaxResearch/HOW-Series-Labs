


#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(){
  const double x_upper_bound = 1.0;
  const double x_lower_bound = 0.0;

  const long nSteps = 1000000000; 
  const double dx = (x_upper_bound - x_lower_bound)/nSteps;

  double integral = 0.0;
  double partial_sum = 0.0;
    
#pragma omp parallel firstprivate(partial_sum)
  {
#pragma omp for
    for(long i = 0; i < nSteps; i++) {
      // integrate using the midpoint rectangle method 
      const double x = x_lower_bound + dx*(double(i) + 0.5);
      partial_sum += 1.0/sqrt(x) * dx;
    }
#pragma omp atomic
    integral += partial_sum;
  }
  const double analytical_result = (sqrt(x_upper_bound) - sqrt(x_lower_bound))*2;
  const double numerical_result = integral;
  printf("Result = %.8f (Should be %.8f; err = %.8f)\n", numerical_result, analytical_result, analytical_result-numerical_result);
}
