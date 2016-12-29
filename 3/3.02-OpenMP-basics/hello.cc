


#include <cstdio>
#include <cstdlib>

int main(){
  const int total_threads = -1;
  printf("There are %d available threads.\n", total_threads); fflush(stdout);

  //parallelize this part
  const int thread_id = -1;
  printf("Hello world from thread %d\n", thread_id);
}
