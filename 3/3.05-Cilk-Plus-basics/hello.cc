#include <cstdio>
#include <cstdlib>

int main(){
  const int total_workers = -1;
  printf("There are %d available workers.\n", total_workers); fflush(stdout);

  //parallelize this part
  const int worker_id = -1;
  printf("Hello world from worker %d\n", worker_id);
}
