CXX = icpc
CXXFLAGS=-qopenmp -mkl
CPUFLAGS = $(CXXFLAGS) -xhost
MICFLAGS = $(CXXFLAGS) -mmic
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = main.o worker.o rngutil.o
MICOBJECTS = main.oMIC worker.oMIC rngutil.oMIC

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
	$(info "make run-cpu" to run the application on the host CPU)
	$(info "make run-mic" to run the application on the coprocessor)
	$(info )

run-cpu: app-CPU
	./app-CPU

run-mic: app-MIC
	scp app-MIC mic0:~/
	ssh mic0 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) ./app-MIC

clean: 
	rm -f $(CPUOBJECTS) $(MICOBJECTS) $(TARGET) *.optrpt

