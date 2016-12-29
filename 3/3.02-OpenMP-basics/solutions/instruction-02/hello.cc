/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/3/3.02-OpenMP-basics/solutions/instruction-02/hello.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main(){
  const int total_threads = omp_get_max_threads();
  printf("There are %d available threads.\n", total_threads); fflush(stdout);

  //parallelize this part
#pragma omp parallel 
  {
    const int thread_id = omp_get_thread_num();
    printf("Hello world from thread %d\n", thread_id);
  }
}
