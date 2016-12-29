B1;3406;0cCXX = icpc
CXXFLAGS=-qopenmp -mkl
CPUFLAGS = $(CXXFLAGS) -xhost
MICFLAGS = $(CXXFLAGS) -mmic
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = dgemm.o

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

run-cpu: app-CPU
	MIC_ENV_PREFIX=PHI PHI_KMP_AFFINITY=compact,granularity=fine ./app-CPU	

instructions: 
	$(info )
	$(info TO EXECUTE THE APPLICATION: )
	$(info "make run-cpu" to run the application on the host CPU)
	$(info )

run-mic: 
	$(info )
	$(info Offload applicaion: No native execution on the coprocessor)
	$(info )

clean: 
	rm -f $(CPUOBJECTS) $(MICOBJECTS) $(TARGET) *.optrpt

