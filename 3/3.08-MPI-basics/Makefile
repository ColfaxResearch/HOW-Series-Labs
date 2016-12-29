CXX = mpiicpc
CXXFLAGS=
CPUFLAGS = $(CXXFLAGS) -xhost
MICFLAGS = $(CXXFLAGS) -mmic
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = mpi.o
MICOBJECTS = mpi.oMIC

TARGET=app-CPU app-MIC

.SUFFIXES: .o .cc .oMIC

all: $(TARGET) 

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


clean: 
	rm -f $(CPUOBJECTS) $(MICOBJECTS) $(TARGET) *.optrpt

