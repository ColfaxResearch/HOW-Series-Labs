/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/2/2.06-shared-virtual-memory-complex-objects/solutions/instruction-03/class.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cstdlib>
#include <cstdio>
#include <cstring>

class _Cilk_shared Person {
 public:
  char* c; // Pointer member - not bitwise-copyable

  Person() { c=NULL; } // Construct without memory allocation

  void Set(const char _Cilk_shared * name) {
    c=(char*)_Offload_shared_malloc(strlen(name)); // Memory alloc
    strcpy(c, name);
#ifdef __MIC__
    printf("In  Set() on coprocessor: %s\n", c);
#else
    printf("In  Set() on host: %s\n", c);
#endif
    fflush(stdout);
  }
};

Person _Cilk_shared someone;
char _Cilk_shared who[10];

_Cilk_shared void PrintString() {
#ifdef __MIC__
  printf("Who on coprocessor: %s\n", who);
#else
  printf("Who on CPU: %s\n", who);
#endif
}

int main(){
  strcpy(who, "Mary");
  PrintString();
  _Cilk_offload PrintString();
  _Cilk_offload someone.Set(who);
  printf("In main(): %s\n", someone.c);
}
