CXX = icpc
CXXFLAGS=-qopenmp -mkl
CPUFLAGS = $(CXXFLAGS) -xAVX
KNCFLAGS = $(CXXFLAGS) -mmic
KNLFLAGS = $(CXXFLAGS) -xMIC_AVX512
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = dgemm.o
KNCOBJECTS = dgemm.oKNC
KNLOBJECTS = dgemm.oKNL

TARGET=app-CPU app-KNC app-KNL
CXX=icpc

.SUFFIXES: .o .cc .oKNC .oKNL

all: $(TARGET) instructions

%-CPU: $(CPUOBJECTS)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CPUFLAGS) -o $@ $(CPUOBJECTS)

%-KNC: $(KNCOBJECTS)
	$(info )
	$(info Linking the KNC executable:)
	$(CXX) $(KNCFLAGS) -o $@ $(KNCOBJECTS)

%-KNL: $(KNCOBJECTS)
	$(info )
	$(info Linking the KNL executable:)
	$(CXX) $(KNLFLAGS) -o $@ $(KNLOBJECTS)

.cc.o:
	$(info )
	$(info Compiling a CPU object file:)
	$(CXX) -c $(CPUFLAGS) $(OPTFLAGS) -o "$@" "$<"

.cc.oKNC:
	$(info )
	$(info Compiling a KNC object file:)
	$(CXX) -c $(KNCFLAGS) $(OPTFLAGS) -o "$@" "$<"

.cc.oKNL:
	$(info )
	$(info Compiling a KNL object file:)
	$(CXX) -c $(KNLFLAGS) $(OPTFLAGS) -o "$@" "$<"

instructions: run-cpu run-knc run-knl
	$(info )

run-cpu: app-CPU
	$(info )
	$(info TO EXECUTE THIS APPLICATION ON CPU, USE THE FOLLOWING COMMAND: )
	$(info OMP_NUM_THREADS=... KMP_AFFINITY=... ./app-CPU)
	$(info )

run-knc: app-KNC
	$(info )
	$(info TO EXECUTE THIS APPLICATION ON COPROCESSOR, USE THE FOLLOWING COMMANDS: )
	$(info scp app-KNC mic0:~/ )
	$(info ssh mic0)
	$(info export LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH))
	$(info KMP_PLACE_THREADS=...C,...t KMP_AFFINITY=... ./app-KNC )
	$(info )

run-knl: app-KNL
	$(info )
	$(info TO EXECUTE THIS APPLICATION ON A XEON PHI PROCESSOR, USE THE FOLLOWING COMMANDS: )
	$(info KMP_PLACE_THREADS=...C,...t KMP_AFFINITY=... ./app-KNL )
	$(info )

queue-cpu: app-CPU
	$(info )
	$(info TO EXECUTE THIS APPLICATION ON CPU, USE THE FOLLOWING COMMAND: )
	$(info OMP_NUM_THREADS=... KMP_AFFINITY=... ./app-CPU)
	$(info )
	echo "`pwd`/app-CPU" | qsub -l nodes=1:xeon -N lab-407-cpu

queue-knc: app-KNC
	$(info )
	$(info TO EXECUTE THIS APPLICATION ON COPROCESSOR, SET KMP_AFFINITY AND OMP_NUM_THREADS: )
	$(info )
	echo "scp `pwd`/app-KNC mic0:~/; \
	      ssh mic0 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) /home/guest/app-KNC $(NSIZE)" | qsub -l nodes=1:knc:coprocessor -N lab-407-knc

queue-knl: app-KNL
	echo "`numactl -m 1 pwd`/app-KNL" | qsub -l nodes=1:knl:xeonphi:flat -N lab-407-knl

clean: 
	rm -f $(CPUOBJECTS) $(KNCOBJECTS) $(KNLOBJECTS) $(TARGET) *.optrpt

