


#ifndef __INCLUDED_RNGUTIL_H__
#define __INCLUDED_RNGUTIL_H__
 
#include <mkl_vsl.h>
#include <omp.h>

class RNGutil {

  // Buffer size
  const long size;
  
  // Current index
  long index;
  
  VSLStreamStatePtr rnStream;

  // Maximum absolute value the random numbers can have
  const float magnitude;

  // Random Number container
  float* rnlist;

 public:

  // Constructor: "seed" is for the RNG seed and buffer_size sets the size of the storage buffer 
  RNGutil(const int seed, const long buffer_size, const float max_value);

  // Default constructor
  RNGutil();

  // Destructor:
  ~RNGutil();

  int Size() { return size; }

  void SetBlock(float* const blk);
  
  // Returns the next random number
  float Next();
    
  // Generates a new set of random numbers.
  void Reset();

};

#endif
