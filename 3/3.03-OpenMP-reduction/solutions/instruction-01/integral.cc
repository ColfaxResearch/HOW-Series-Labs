/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/3/3.03-OpenMP-reduction/solutions/instruction-01/integral.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(){
  const double x_upper_bound = 1.0;
  const double x_lower_bound = 0.0;

  const long nSteps = 1000000000; 
  const double dx = (x_upper_bound - x_lower_bound)/nSteps;

  double integral = 0.0;

#pragma omp parallel for
  for(long i = 0; i < nSteps; i++) {
    // integrate using the midpoint rectangle method 
    const double x = x_lower_bound + dx*(double(i) + 0.5);
    integral += 1.0/sqrt(x) * dx;
  }
  
  const double analytical_result = (sqrt(x_upper_bound) - sqrt(x_lower_bound))*2;
  const double numerical_result = integral;
  printf("Result = %.8f (Should be %.8f; err = %.8f)\n", numerical_result, analytical_result, analytical_result-numerical_result);
}
