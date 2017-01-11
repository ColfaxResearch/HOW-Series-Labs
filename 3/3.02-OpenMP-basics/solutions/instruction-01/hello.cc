#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main(){
  const int total_threads = omp_get_max_threads();
  printf("There are %d available threads.\n", total_threads); fflush(stdout);

  //parallelize this part
  const int thread_id = omp_get_thread_num();
  printf("Hello world from thread %d\n", thread_id);
}
