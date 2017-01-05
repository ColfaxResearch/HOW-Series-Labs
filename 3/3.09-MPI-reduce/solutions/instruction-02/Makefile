CXX = mpiicpc
CXXFLAGS=
CPUFLAGS = $(CXXFLAGS) -xhost
KNCFLAGS = $(CXXFLAGS) -mmic
KNLFLAGS = $(CXXFLAGS) -xMIC-AVX512
OPTFLAGS = -qopt-report -qopt-report-file=$@.optrpt

CPUOBJECTS = integral.o
KNCOBJECTS = integral.oKNC
KNLOBJECTS = integral.oKNL

TARGET=app-CPU app-KNC app-KNL

.SUFFIXES: .o .cc .oKNC .oKNL

all: $(TARGET) instructions

%-CPU: $(CPUOBJECTS)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CPUFLAGS) -o $@ $(CPUOBJECTS)

%-KNC: $(KNCOBJECTS)
	$(info )
	$(info Linking the KNC executable:)
	$(CXX) $(KNCFLAGS) -o $@ $(KNCOBJECTS)

%-KNL: $(KNLOBJECTS)
	$(info )
	$(info Linking the KNL executable:)
	$(CXX) $(KNLFLAGS) -o $@ $(KNLOBJECTS)

.cc.o:
	$(info )
	$(info Compiling a CPU object file:)
	$(CXX) -c $(CPUFLAGS) $(OPTFLAGS) -o "$@" "$<"

.cc.oKNC:
	$(info )
	$(info Compiling a KNC object file:)
	$(CXX) -c $(KNCFLAGS) $(OPTFLAGS) -o "$@" "$<"

.cc.oKNL:
	$(info )
	$(info Compiling a KNL object file:)
	$(CXX) -c $(KNLFLAGS) $(OPTFLAGS) -o "$@" "$<"

instructions: 
	$(info )
	$(info SEE INSTRUCTIONS FOR HOW TO RUN APPLICATIONS: )
	$(info )

run-cpu: instructions

run-knc: instructions

run-knl: instructions

queue-cpu: instructions

queue-knc: instructions

queue-knl: instructions

clean: 
	rm -f $(CPUOBJECTS) $(KNCOBJECTS) $(KNLOBJECTS) $(TARGET) *.optrpt

