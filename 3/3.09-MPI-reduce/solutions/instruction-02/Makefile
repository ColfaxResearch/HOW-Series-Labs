CXX = mpiicpc
CXXFLAGS=
CPUFLAGS = $(CXXFLAGS) -xAVX
KNCFLAGS = $(CXXFLAGS) -mmic
KNLFLAGS = $(CXXFLAGS) -xMIC-AVX512
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = integral.o
KNCOBJECTS = integral.oKNC
KNLOBJECTS = integral.oKNL

TARGET=app-CPU app-KNC app-KNL

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

%-KNL: $(KNLOBJECTS)
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

instructions: 
	$(info )
	$(info TO EXECUTE THE APPLICATION: )
	$(info "make run-cpu" to run the application on the host CPU)
	$(info "make run-knc" to run the application on the coprocessor)
	$(info "make run-knl" to run the application on the host KNL)
	$(info )
	$(info TO QUEUE THE APPLICATION FOR TORQUE: )
	$(info "make queue-cpu" to queue application for the host CPU)
	$(info "make queue-knc" is currently not supported)
	$(info "make queue-knl" to queue application for the host KNL)
	$(info )

run-cpu: app-CPU
	mpirun -host localhost -np 4 ./app-CPU
run-knc: app-KNC
	scp app-KNC mic0:~/
	mpirun -host localhost -np 4 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) ~/app-KNC

run-knl: app-KNL
	mpirun -host localhost -np 4 ./app-KNL

queue-cpu: app-CPU
	echo "mpirun -machine \$$PBS_NODEFILE -np 4 `pwd`/app-CPU" | qsub -l nodes=4:xeon -N lab-309-cpu

queue-knc: app-KNC
	$(info MPI on KNC is not currently supported )

queue-knl: app-KNL
	echo "mpirun -machine \$$PBS_NODEFILE -np 4 `pwd`/app-KNL" | qsub -l nodes=4:knl:xeonphi -N lab-309-knl

clean: 
	rm -f $(CPUOBJECTS) $(KNCOBJECTS) $(KNLOBJECTS) $(TARGET) *.optrpt

