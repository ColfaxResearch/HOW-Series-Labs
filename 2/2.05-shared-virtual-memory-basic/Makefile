CXX = icpc
CXXFLAGS =

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
	echo "`pwd`/runme" | qsub -l nodes=1:knc:coprocessor -N lab-205-cpu

clean: 
	rm -f *.o runme
