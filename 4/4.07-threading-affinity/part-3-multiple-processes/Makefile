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

CPUOBJECTS = dfft.o
MICOBJECTS = dfft.oMIC

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

instructions: run-cpu run-knc
	$(info )

run-cpu: app-CPU
	$(info )
	$(info YOU MUST EDIT run-multiple-processes.sh TO OPTIMIZE THREAD AFFFINITY )
	$(info TO EXECUTE THIS APPLICATION ON CPU, USE THE FOLLOWING COMMAND: )
	$(info ./run-multiple-processes.sh )
	$(info )

run-knc: app-MIC
	$(info )
	$(info YOU MUST EDIT run-multiple-processes.sh TO OPTIMIZE THREAD AFFFINITY )
	$(info TO EXECUTE THIS APPLICATION ON COPROCESSOR, USE THE FOLLOWING COMMANDS: )
	$(info scp app-MIC ./run-multiple-processes.sh mic0:~/ )
	$(info ssh mic0)
	$(info export LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH))
	$(info ./run-multiple-processes.sh )
	$(info )

clean: 
	rm -f $(CPUOBJECTS) $(MICOBJECTS) $(TARGET) *.optrpt log-process*.txt

