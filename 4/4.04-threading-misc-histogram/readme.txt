NOTE: for the application studied in this lab, hyper-threading is counter-productive.
      Set the number of OpenMP threads as follows (assuming 16 physical cores 
      on the host and 60 physical cores on the coprocessor):

export OMP_NUM_THREADS=16
export MIC_ENV_PREFIX=MIC
export MIC_OMP_NUM_THREADS=120


Included Files:
main.cc    - Contains the main(). It initializes the array "age" with random
	     numbers and then calls histogram() from "worker.cc" to populate
	     array hist(). Finally it checks the result, and then prints the
	     performance of histogram() function.

worker.cc  - Contains histogram(), which is the subject of this lab. Contains
	     all the instructions.

worker.h   - Header file for "worker.cc".

rngutil.cc - Vectorized Random Number Generator, to speed up the initialization
	     process. Not important for the content of this lab.

rngutil.h  - Header file for rngutil.cc.



