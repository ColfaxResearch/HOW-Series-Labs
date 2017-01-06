CXX = icpc
CXXFLAGS=
CPUFLAGS = $(CXXFLAGS) -xhost
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = matrix.o

TARGET=app-CPU
CXX=icpc

.SUFFIXES: .o .cc

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
	$(info "make run-cpu"   to run the application on the host CPU)
	$(info "make queue-cpu" to pu the job on a Torque queue)
	$(info )

run-cpu: app-CPU
	./app-CPU

queue-cpu: app-CPU
	echo "`pwd`/app-CPU" | qsub -l nodes=1:knc -N lab-204-cpu

clean: 
	rm -f $(CPUOBJECTS) $(MICOBJECTS) $(TARGET) *.optrpt

