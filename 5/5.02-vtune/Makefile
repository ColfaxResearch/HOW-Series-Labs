CXX = icpc
CXXFLAGS=-qopenmp -mkl -g -O3
CPUFLAGS = $(CXXFLAGS) -xhost
MICFLAGS = $(CXXFLAGS) -mmic
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = main.o worker.o
MICOBJECTS = main.oMIC worker.oMIC
OFFOBJECTS = main-offload.o worker-offload.o

TARGET=app-CPU app-MIC app-OFF
CXX=icpc

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

%-OFF: $(OFFOBJECTS)
	$(info )
	$(info Linking the OFFLOAD executable:)
	$(CXX) $(CPUFLAGS) -o $@ $(OFFOBJECTS)

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
	$(info For this exercise, use VTune to run the provided scripts: )
	$(info )

clean: 
	rm -f $(CPUOBJECTS) $(MICOBJECTS) $(TARGET) *.optrpt

