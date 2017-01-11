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

