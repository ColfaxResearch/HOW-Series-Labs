


#include <omp.h>
#include <mkl_vsl.h>
#include <cmath>

void GenerateRandomNumbers(const int m, const int n, float* const data) {
  // Filling m arrays with normally distributed n random numbers
#pragma omp parallel
  {
    VSLStreamStatePtr rng;
    const int seed = omp_get_thread_num();
    int status = vslNewStream(&rng, VSL_BRNG_MT19937, omp_get_thread_num());

#pragma omp for schedule(guided)
    for (int i = 0; i < m; i++) {
      const float mean = (float)i;
      const float stdev = 1.0f;
      status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, 
			     rng, n, &data[i*n], mean, stdev);
    }
    
    vslDeleteStream(&rng);
  }
}

void ComputeMeanAndStdev(const int m, const int n, const float* data, 
			 float* const resultMean, float* const resultStdev) {

  // Processing n arrays to compute the mean and standard deviation
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < m; i++) {
    float sumx=0.0f, sumx2=0.0f;
#pragma vector aligned
    for (int j = 0; j < n; j++) {
      sumx  += data[i*n + j];
      sumx2 += data[i*n + j]*data[i*n + j];
    }
    resultMean[i] = sumx/(float)n;
    resultStdev[i] = sqrtf(sumx2/(float)n-resultMean[i]*resultMean[i]);
  }

}

void RunStatistics(const int m, const int n, 
		   float* const resultMean, float* const resultStdev) {

  // Allocating memory for scratch space for the whole problem
  // m*n elements on heap (does not fit on stack)
  float* allData = (float*) _mm_malloc((size_t)m*(size_t)n*sizeof(float), 64);

  GenerateRandomNumbers(m, n, allData);
  ComputeMeanAndStdev(m, n, allData, resultMean, resultStdev);

  // Deallocating scratch space
  _mm_free(allData);

}

