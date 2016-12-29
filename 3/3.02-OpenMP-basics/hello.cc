/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/3/3.02-OpenMP-basics/hello.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cstdlib>

int main(){
  const int total_threads = -1;
  printf("There are %d available threads.\n", total_threads); fflush(stdout);

  //parallelize this part
  const int thread_id = -1;
  printf("Hello world from thread %d\n", thread_id);
}
