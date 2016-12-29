# MODIFY THESE PARAMETERS TO OPTIMIZE THE CALCULATION:
KMP_AFFINITY=scatter
OMP_NUM_THREADS=1

MIC_KMP_AFFINITY=none
MIC_OMP_NUM_THREADS=4
# END OF PARAMETERS TO MODIFY
CXX = icpc
CXXFLAGS=-qopenmp -mkl
CPUFLAGS = $(CXXFLAGS) -xhost
MICFLAGS = $(CXXFLAGS) -mmic
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = dgemm.o
MICOBJECTS = dgemm.oMIC

TARGET=app-CPU app-MIC
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

instructions: run-cpu run-mic
	$(info )

run-cpu: app-CPU
	$(info )
	$(info TO EXECUTE THIS APPLICATION ON CPU, USE THE FOLLOWING COMMAND: )
	$(info OMP_NUM_THREADS=... KMP_AFFINITY=... ./app-CPU)
	$(info )

run-mic: app-MIC
	$(info )
	$(info TO EXECUTE THIS APPLICATION ON COPROCESSOR, USE THE FOLLOWING COMMANDS: )
	$(info scp app-MIC mic0:~/ )
	$(info ssh mic0)
	$(info export LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH))
	$(info KMP_PLACE_THREADS=...C,...t KMP_AFFINITY=... ./app-MIC )
	$(info )

clean: 
	rm -f $(CPUOBJECTS) $(MICOBJECTS) $(TARGET) *.optrpt

