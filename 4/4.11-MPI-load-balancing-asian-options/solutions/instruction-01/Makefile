CXX = mpiicpc
CXXFLAGS=-qopenmp -mkl
CPUFLAGS = $(CXXFLAGS) -xhost
MICFLAGS = $(CXXFLAGS) -mmic
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = main.o workdistribution.o pricing.o 
MICOBJECTS = main.oMIC workdistribution.oMIC pricing.oMIC 

TARGET=app
MICTARGET=app-MIC

# Modify this line to change workload balance in the heterogeneous application

.SUFFIXES: .o .cc .oMIC

all: $(TARGET) $(MICTARGET) instructions

app: $(CPUOBJECTS)
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

run-cpu: app
	$(info )
	$(info This version is not designed for CPU-only applications. Use 'make run-het')

run-mic: app-MIC
	$(info This version is not designed for MIC-only applications. Use 'make run-het')

run-het: app app-MIC
ifndef OPTIONS_ALPHA
	$(error Define the environment variable OPTIONS_ALPHA for load balancing before running the code.)
endif
	ssh mic0 mkdir -p $(PWD)
	scp app-MIC mic0:$(PWD)/
	$(info )
	$(info Using workload balancing parameter alpha=$(OPTIONS_ALPHA))
	$(info Modify variable OPTIONS_ALPHA in the Makefile to tune it)
	$(info )
	OPTIONS_ALPHA=$(OPTIONS_ALPHA) I_MPI_MIC=1 I_MPI_MIC_POSTFIX="-MIC" LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(MIC_LD_LIBRARY_PATH) mpirun -machine machines-heterogeneous.txt $(PWD)/app

instructions: 
	$(info )
	$(info TO EXECUTE THE APPLICATION: )
	$(info "make run-het" to run the application on the heterogeneous combination of CPUs and MICs)
	$(info )


clean: 
	rm -f $(CPUOBJECTS) $(MICOBJECTS) $(TARGET) $(MICTARGET) *.optrpt

