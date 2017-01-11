#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>

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

Person * _Cilk_shared someone;
char _Cilk_shared who[10];

_Cilk_shared void PrintString() {
#ifdef __MIC__
  printf("Who on coprocessor: %s\n", who);
#else
  printf("Who on CPU: %s\n", who);
#endif
}

int main(){
  someone = (Person _Cilk_shared*) new(_Offload_shared_malloc(sizeof(Person))) Person();
  strcpy(who, "Mary");
  PrintString();
  _Cilk_offload PrintString();
  _Cilk_offload someone->Set(who);
  printf("In main(): %s\n", someone->c);
}
