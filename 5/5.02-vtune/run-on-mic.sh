#!/bin/bash

source /opt/intel/parallel_studio_xe_2015/psxevars.sh
export LD_LIBRARY_PATH=$MIC_LD_LIBRARY_PATH
export OMP_SCHEDULE=static
./app-MIC
