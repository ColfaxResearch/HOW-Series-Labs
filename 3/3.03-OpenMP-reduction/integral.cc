


#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(){
  const double x_upper_bound = 1.0;
  const double x_lower_bound = 0.0;

  const long nSteps = 1000000000; 
  const double dx = (x_upper_bound - x_lower_bound)/nSteps;

  double integral = 0.0;

  for(long i = 0; i < nSteps; i++) {
    // integrate using the midpoint rectangle method 
    const double x = x_lower_bound + dx*(double(i) + 0.5);
    integral += 1.0/sqrt(x) * dx;
  }
  
  const double analytical_result = (sqrt(x_upper_bound) - sqrt(x_lower_bound))*2;
  const double numerical_result = integral;
  printf("Result = %.8f (Should be %.8f; err = %.8f)\n", numerical_result, analytical_result, analytical_result-numerical_result);
}
