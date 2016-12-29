/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/2/2.06-shared-virtual-memory-complex-objects/class.cc,
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

class Person {
 public:
  char* c; // Pointer member - not bitwise-copyable

  Person() { c=NULL; } // Construct without memory allocation

  void Set(const char * name) {
    c=(char*) malloc(strlen(name)); // Memory alloc
    strcpy(c, name);
    printf("In  Set(): %s\n", c);
    fflush(stdout);
  }
};

Person someone;
char who[10];

void PrintString() {
#ifdef __MIC__
  printf("Who on coprocessor: %s\n", who);
#else
  printf("Who on CPU: %s\n", who);
#endif
}

int main(){
  strcpy(who, "Mary");
  PrintString();
  someone.Set(who);
  printf("In main(): %s\n", someone.c);
}
