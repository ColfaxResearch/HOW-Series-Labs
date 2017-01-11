#include <cstdio>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    printf("Hello world! I have %ld logical processors.\n",
            sysconf(_SC_NPROCESSORS_ONLN ));
    MPI_Finalize();
}
