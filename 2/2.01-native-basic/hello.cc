#include <cstdio>
#include <unistd.h>

int main(){
    printf("Hello world! I have %ld logical processors.\n",
            sysconf(_SC_NPROCESSORS_ONLN ));
}
