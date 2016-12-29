/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/3/3.05-Cilk-Plus-basics/solutions/instruction-01/hello.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <cstdlib>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

int main(){

  const int total_workers = __cilkrts_get_nworkers();
  printf("There are %d available workers.\n", total_workers); fflush(stdout);

  //parallelize this part
  for(int i = 0; i < total_workers; i++) {
    const int worker_id = __cilkrts_get_worker_number();
    printf("Hello world from worker %d\n", worker_id);
  }
}

