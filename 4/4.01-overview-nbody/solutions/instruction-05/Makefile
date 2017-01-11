CXX = mpiicpc
CXXFLAGS=-qopenmp -mkl
CPUFLAGS = $(CXXFLAGS) -xAVX
KNCFLAGS = $(CXXFLAGS) -mmic
KNLFLAGS = $(CXXFLAGS) -xMIC_AVX512 -DKNLTILE
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt -fp-model fast=2

CPUOBJECTS = nbody.o
KNCOBJECTS = nbody.oKNC
KNLOBJECTS = nbody.oKNL

NSIZE=1048576

TARGET=app-CPU app-KNC app-KNL

.SUFFIXES: .o .cc .oKNC .oKNL

all: $(TARGET) instructions

%-CPU: $(CPUOBJECTS)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CXXFLAGS) $(CPUFLAGS) -o $@ $(CPUOBJECTS)

%-KNC: $(KNCOBJECTS)
	$(info )
	$(info Linking the KNC executable:)
	$(CXX) $(CXXFLAGS) $(KNCFLAGS) -o $@ $(KNCOBJECTS)

%-KNL: $(KNLOBJECTS)
	$(info )
	$(info Linking the KNL executable:)
	$(CXX) $(CXXFLAGS) $(KNLFLAGS) -o $@ $(KNLOBJECTS)

.cc.o:
	$(info )
	$(info Compiling a CPU object file:)
	$(CXX) $(CXXFLAGS) -c $(CPUFLAGS) $(OPTFLAGS) -o "$@" "$<"

.cc.oKNC:
	$(info )
	$(info Compiling a KNC object file:)
	$(CXX) $(CXXFLAGS) -c $(KNCFLAGS) $(OPTFLAGS) -o "$@" "$<"

.cc.oKNL:
	$(info )
	$(info Compiling a KNL object file:)
	$(CXX) $(CXXFLAGS) -c $(KNLFLAGS) $(OPTFLAGS) -o "$@" "$<"

instructions: 
	$(info )
	$(info TO EXECUTE THE APPLICATION: )
	$(info "make run-cpu" to run the application on the host CPU)
	$(info "make run-knc" to run the application on the coprocessor)
	$(info "make run-knl" to run the application on the host KNL)
	$(info )
	$(info TO QUEUE THE APPLICATION FOR TORQUE: )
	$(info "make queue-cpu" to queue application for the host CPU)
	$(info "make queue-knc" this feature is currently unsupported)
	$(info "make queue-knl" to queue application for the host KNL)
	$(info )

run-cpu: app-CPU
	mpirun -host localhost -np 4 -genv KMP_AFFINITY=compact,granularity=fine ./app-CPU $(NSIZE)

run-knc: app-KNC
	scp app-KNC mic0:~/
	mpirun -host mic0 -np 4 -genv LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) -genv KMP_AFFINITY=compact,granularity=fine ./app-KNC $(NSIZE)

run-knl: app-KNL
	mpirun -host localhost -np 4 -genv KMP_AFFINITY=compact,granularity=fine ./app-KNL $(NSIZE)

queue-cpu: app-CPU
	echo "mpirun -machinefile \$$PBS_NODEFILE -np 4 -genv KMP_AFFINITY=compact,granularity=fine `pwd`/app-CPU $(NSIZE)" | qsub -l nodes=4:xeon -N lab-401-cpu

queue-knc: 
	$(info This feature is currently unsupported)

queue-knl: app-KNL
	echo "mpirun -machinefile \$$PBS_NODEFILE -np 4 -genv KMP_AFFINITY=compact,granularity=fine numactl -m 1 `pwd`/app-KNL $(NSIZE)" | qsub -l nodes=4:knl:xeonphi:flat -N lab-401-knl

clean: 
	rm -f $(CPUOBJECTS) $(KNCOBJECTS) $(KNLOBJECTS) $(TARGET) *.optrpt
