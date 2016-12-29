CXX = icpc
CXXFLAGS=-qopenmp
CPUFLAGS = $(CXXFLAGS) -xhost
MICFLAGS = $(CXXFLAGS) -mmic
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = copy.o
MICOBJECTS = 

TARGET=app-CPU
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

.cc.o:
	$(info )
	$(info Compiling a CPU object file:)
	$(CXX) -c $(CPUFLAGS) $(OPTFLAGS) -o "$@" "$<"

.cc.oMIC:
	$(info )
	$(info Compiling a MIC object file:)
	$(CXX) -c $(MICFLAGS) $(OPTFLAGS) -o "$@" "$<"

instructions: run-cpu
	$(info )

run-cpu: app-CPU
	$(info )
	$(info TO EXECUTE THIS OFFLOAD APPLICATION, USE THE FOLLOWING COMMAND: )
	$(info OMP_NUM_THREADS=... KMP_AFFINITY=... MIC_ENV_PREFIX=... ..._KMP_PLACE_THREADS=... ./app-CPU)
	$(info )

clean: 
	rm -f $(CPUOBJECTS) $(MICOBJECTS) $(TARGET) *.optrpt

