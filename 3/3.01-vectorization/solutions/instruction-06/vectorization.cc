#include <cstdio>
#include <cstdlib>

//adds two numbers and returns the sum
double my_scalar_add(double, double);
//adds the second vector to the first
void my_vector_add(int, double*, double*);

int main(){
    
  const int n=10000;
  const int maxIter = 5;
  double a[n], b[n];

  // Cilk Plus array notation
  a[:]=1.0/(double)n;
  b[:]=2.0;
  

  for(int i = 0 ; i < n ; i++)   // Addition (For instructions 0 - 2)
    b[i]=a[i]+b[i];


  for(int i = 0 ; i < n ; i++)  // Scalar function (For instruction 3)
    a[i]=my_scalar_add(a[i],b[i]);

  my_vector_add(n,a,b);    // Vector function (For instruction 4, 5)
}
