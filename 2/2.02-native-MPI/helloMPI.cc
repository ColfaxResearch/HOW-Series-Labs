/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/2/2.02-native-MPI/helloMPI.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdio>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    printf("Hello world! I have %ld logical cores.\n",
            sysconf(_SC_NPROCESSORS_ONLN ));
    MPI_Finalize();
}
