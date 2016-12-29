#!/bin/bash

source /opt/intel/parallel_studio_xe_2015/psxevars.sh
export OMP_SCHEDULE=static

./app-CPU
