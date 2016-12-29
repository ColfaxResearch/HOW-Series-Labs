/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.05-threading-insufficient-parallelism-sweep/solutions/instruction-03/worker.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cassert>

void SumColumns(const int m, const int n, long* M, long* s){

  const int tile = 10000;
  assert(n%tile == 0);
  s[0:m] = 0;

#pragma omp parallel
  {
    // Each thread will need a private container
    long sum[m];    sum[:] = 0;

  // Loop collapse expands iteration space
#pragma omp for collapse(2)
    for (int i = 0; i < m; i++)
      for (int jj = 0; jj < n; jj+=tile)
       	for (int j = jj; j < jj+tile; j++)
	  sum[i] += M[i*n+j];

    // Reducing from thread containers to the output array
    for (int i = 0; i < m; i++)
#pragma omp atomic
      s[i] += sum[i];

  }
}

