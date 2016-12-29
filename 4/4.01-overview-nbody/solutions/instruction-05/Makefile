CXX = mpiicpc
CXXFLAGS=-qopenmp -mkl
CPUFLAGS = $(CXXFLAGS) -xhost
MICFLAGS = $(CXXFLAGS) -mmic
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = nbody.o
MICOBJECTS = nbody.oMIC

TARGET=app-CPU app-MIC


.SUFFIXES: .o .cc .oMIC

all: $(TARGET) instructions

%-CPU: $(CPUOBJECTS)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CPUFLAGS) -o $@ $(CPUOBJECTS)

%-MIC: $(MICOBJECTS)
	$(info )
	$(info Linking the MIC executable:)
	$(CXX) $(MICFLAGS) -o $@ $(MICOBJECTS)

.cc.o:
	$(info )
	$(info Compiling a CPU object file:)
	$(CXX) -c $(CPUFLAGS) $(OPTFLAGS) -o "$@" "$<"

.cc.oMIC:
	$(info )
	$(info Compiling a MIC object file:)
	$(CXX) -c $(MICFLAGS) $(OPTFLAGS) -o "$@" "$<"

instructions: 
	$(info )
	$(info TO EXECUTE THE APPLICATION: )
	$(info "make run-mpi" to run the MPI application on host and MIC with mpirun)
	$(info )

run-cpu: app-CPU
	mpirun -host localhost -np 4 -genv "KMP_AFFINITY=compact" ./app-CPU 65536

run-mic: app-MIC
	scp app-MIC mic0:~/
	I_MPI_MIC=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(MIC_LD_LIBRARY_PATH) mpirun -host mic0 -np 4 -genv "KMP_AFFINITY=compact" ~/app-MIC 65536

clean: 
	rm -f $(CPUOBJECTS) $(MICOBJECTS) $(TARGET) *.optrpt

