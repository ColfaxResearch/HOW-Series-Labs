void SumColumns(const int m, const int n, long* M, long* s){

  for (int i = 0; i < m; i++) {
    long sum = 0;

    // Parallelizing the inner loop to have more thread parallelism
#pragma omp parallel for reduction(+: sum)
    for (int j = 0; j < n; j++)
      sum += M[i*n+j];

    s[i] = sum;
  }
}
