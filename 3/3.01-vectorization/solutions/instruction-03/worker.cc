__attribute__((vector)) double my_scalar_add(double x1, double x2){
  return x1+x2;
}

void my_vector_add(int n, double* a, double* b){
  for(int i = 0; i < n; i++)
    a[i] += b[i];
}
