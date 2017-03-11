CXX = icpc
CXXFLAGS=-qopenmp
CPUFLAGS = $(CXXFLAGS) -xAVX
KNLFLAGS = $(CXXFLAGS) -xMIC_AVX512 -DNO_OFFLOAD
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = copy.o
KNLOBJECTS = copy.oKNL

TARGET=app-CPU app-KNL
CXX=icpc

.SUFFIXES: .o .cc .oKNL

all: $(TARGET) instructions

%-CPU: $(CPUOBJECTS)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CPUFLAGS) -o $@ $(CPUOBJECTS)

%-KNL: $(KNLOBJECTS)
	$(info )
	$(info Linking the KNL executable:)
	$(CXX) $(KNLFLAGS) -o $@ $(KNLOBJECTS)

.cc.o:
	$(info )
	$(info Compiling a CPU object file:)
	$(CXX) -c $(CPUFLAGS) $(OPTFLAGS) -o "$@" "$<"

.cc.oKNL:
	$(info )
	$(info Compiling a KNL object file:)
	$(CXX) -c $(KNLFLAGS) $(OPTFLAGS) -o "$@" "$<"

instructions: run-cpu
	$(info )

run-cpu: app-CPU
	$(info )
	$(info TO EXECUTE THIS OFFLOAD APPLICATION, USE THE FOLLOWING COMMAND: )
	$(info OMP_NUM_THREADS=... KMP_AFFINITY=... MIC_ENV_PREFIX=... ..._KMP_PLACE_THREADS=... ./app-CPU)
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

queue-knl: app-KNL
	echo "numactl -m 1 `pwd`/app-KNL" | qsub -l nodes=1:knl:xeonphi:flat -N lab-407-knl

clean: 
	rm -f $(CPUOBJECTS) $(KNLOBJECTS) $(TARGET) *.optrpt

