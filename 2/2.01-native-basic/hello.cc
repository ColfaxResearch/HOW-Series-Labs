


#include <cstdio>
#include <unistd.h>

int main(){
    printf("Hello world! I have %ld logical cores.\n",
            sysconf(_SC_NPROCESSORS_ONLN ));
}
