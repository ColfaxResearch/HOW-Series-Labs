/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/3/3.01-vectorization/solutions/instruction-03/worker.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



__attribute__((vector)) double my_scalar_add(double x1, double x2){
  return x1+x2;
}

void my_vector_add(int n, double* a, double* b){
  for(int i = 0; i < n; i++)
    a[i] += b[i];
}
