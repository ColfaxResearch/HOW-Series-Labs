CXX = icpc
CXXFLAGS = -g

OBJECTS = matrix.o

.SUFFIXES: .o .cpp

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o "$@" "$<"

all: runme

runme: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o runme $(OBJECTS)

run-cpu: runme
	./runme

queue-cpu: runme
	echo "OFFLOAD_REPORT=2 `pwd`/runme" | qsub -l nodes=1:knc -N lab-203-cpu

clean: 
	rm -f *.o runme
