#include "rngutil.h"

#define RNGUTILPADDING 1024L

RNGutil::RNGutil() : size(2048), magnitude(1.0f) {
  index = 0L;
  rnlist = (float*) _mm_malloc(sizeof(float)*size + RNGUTILPADDING, 64);
  vslNewStream( &rnStream, VSL_BRNG_MT19937, omp_get_thread_num() );
  Reset();
}

RNGutil::RNGutil(const int seed, const long buffer_size, const float max_value) : size(buffer_size), magnitude(max_value) {
  index = 0L;
  rnlist = (float*) _mm_malloc(sizeof(float)*size + RNGUTILPADDING, 64);
  vslNewStream( &rnStream, VSL_BRNG_MT19937, seed );
  Reset();
}

RNGutil::~RNGutil(){
  _mm_free(rnlist);
}

void RNGutil::Reset(){
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, size, rnlist, 0.0, magnitude);
}

float RNGutil::Next(){
  index+=1L;
  if(index >= size) {
    Reset();
    index = 1L;
  }
  return rnlist[index-1L];
}

void RNGutil::SetBlock(float* const blk) {
  index=size;
  Reset();
  blk[0:size] = rnlist[0:size];
}
