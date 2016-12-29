/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.05-threading-insufficient-parallelism-sweep/worker.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



void SumColumns(const int m, const int n, long* M, long* s){

  // Distribute rows across threads
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    // Private variable for reduction
    long sum = 0;

    // Vectorize inner loop
    for (int j = 0; j < n; j++)
      sum += M[i*n+j];

    s[i] = sum;
  }
}

