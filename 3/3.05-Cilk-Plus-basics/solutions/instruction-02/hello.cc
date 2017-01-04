


#include <cstdio>
#include <cstdlib>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

// For usleep
#include <unistd.h>

int main(){

  const int total_workers = __cilkrts_get_nworkers();
  printf("There are %d available workers.\n", total_workers); fflush(stdout);

  //parallelize this part
  cilk_for(int i = 0; i < total_workers; i++) {
    const int worker_id = __cilkrts_get_worker_number();
    printf("Hello world from worker %d\n", worker_id);
    // waiting to ensure all workers are used
    usleep(10000);
  }
}

