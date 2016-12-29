#!/bin/bash
source /opt/intel/parallel_studio_xe_2015/psxevars.sh
export MIC_ENV_PREFIX=PHI
export PHI_OMP_SCHEDULE=static

./app-OFF
