#!/bin/bash

### TO DO THE EXERCISE YOU NEED TO CHANGE THE VALUES AND EXPRESSIONS IN FUNCTION tune_affinity()

function tune_affinity() {

    # CHANGE THIS VALUE to vary how many concurrent processes you will run:
    export NUMBER_OF_PROCESSES=1

    # CHANGE THIS VALUE to very how many threads per process you wish to use
    export OMP_NUM_THREADS=1

    # CHANGE THIS EXPRESSION to bind threads for process number i to respective cores
    # Hint: you may use the running counter OCCUPIED_THREADS in that expression
    export KMP_AFFINITY=none

}

### YOU DO NOT HAVE TO CHANGE ANYTHING BELOW THIS LINE TO DO THE EXERCISE ###



function stop_run() {

    # This function is executed when the user presses Ctrl+C
    echo
    echo "Terminating background processes..."
    echo kill $PROCID
    kill $PROCID
    exit
}



function setup_run() {

    # This is so that you can stop multiple processes by pressing Ctrl+C
    PROCID=""
    trap stop_run INT

    # Calling this function preliminarily to query the number of processes requested by the user
    tune_affinity

    # This is to prevent a crash when too many processes are requested
    V=`cat /proc/meminfo | grep MemFree | awk '{TM=1500*1000*'${NUMBER_OF_PROCESSES}'; if (TM>$2) print "1"}'`
    if [ "$V" == "1" ]; then
	echo "Not enough memory to run $NUMBER_OF_PROCESSES processes"
	exit
    fi

}



function run_benchmark() {

    # This is a counter, you may use it in the expression for KMP_AFFINITY below
    OCCUPIED_THREADS=0

    # This function launches multiple processes and benchmarks them
    STARTTIME=$(date +%s)
    for ((i=0; $i<$NUMBER_OF_PROCESSES; i++)); do

	# Before every process, call tune_affinity to set the affinity for this process
	tune_affinity

	if [ `uname -a | grep -c mpss` -eq 1 ]; then
	    # Executing on the coprocessor
	    ./app-MIC &
	else
	    # Executing on the host
	    ./app-CPU &
	fi

        # Recording the process ID to terminate it if the user presses Ctrl+C
	PROCID="$PROCID $!"

        # Incrementing the counter, do not change this line
	let OCCUPIED_THREADS=OCCUPIED_THREADS+OMP_NUM_THREADS
    done
    wait
    ENDTIME=$(date +%s)

}


function output_results() {

    echo $STARTTIME $ENDTIME | awk '{P=2.5*26*67108864*1e-9/($2-$1)*100*'${NUMBER_OF_PROCESSES}'; printf "\n\033[1mCumulative performance of '${NUMBER_OF_PROCESSES}' processes\n(including initialization and slow iterations):   \033[42m %.2f GFLOP/s\033[0m\n\n", P}'

}

# This is the main script
setup_run
run_benchmark
output_results
