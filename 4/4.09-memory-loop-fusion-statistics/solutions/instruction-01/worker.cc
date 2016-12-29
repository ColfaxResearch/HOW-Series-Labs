


#include <omp.h>
#include <mkl_vsl.h>
#include <cmath>

void GenerateRandomNumbers(const int n, float* const data, const float mean, const float stdev, VSLStreamStatePtr &rng) {
  
  // Filling one array with normally distributed n random numbers
  int status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, 
			     rng, n, data, mean, stdev);

}

void ComputeMeanAndStdev(const int n, const float* data, 
			 float* const resultMean, float* const resultStdev) {

  // Processing one array to compute the mean and standard deviation
  float sumx=0.0f, sumx2=0.0f;
#pragma vector aligned
  for (int j = 0; j < n; j++) {
    sumx  += data[j];
    sumx2 += data[j]*data[j];
  }
  *resultMean = sumx/(float)n;
  *resultStdev = sqrtf(sumx2/(float)n-(*resultMean)*(*resultMean));

}

void RunStatistics(const int m, const int n, 
		   float* const resultMean, float* const resultStdev) {

  // Allocating memory for scratch space for the whole problem
  // m*n elements on heap (does not fit on stack)
  float* allData = (float*) _mm_malloc((size_t)m*(size_t)n*sizeof(float), 64);

#pragma omp parallel
  {
    VSLStreamStatePtr rng;
    const int seed = omp_get_thread_num();
    int status = vslNewStream(&rng, VSL_BRNG_MT19937, omp_get_thread_num());

#pragma omp for schedule(guided)
    for (int i = 0; i < m; i++) {
      float* data = &allData[i*n];
      const float mean = (float)i;
      const float stdev = 1.0f;

      // Generating data for one array
      GenerateRandomNumbers(n, data, mean, stdev, rng);

      // Processing it immediately
      ComputeMeanAndStdev(n, data, &resultMean[i], &resultStdev[i]);
    }

    vslDeleteStream(&rng);
  }

  // Deallocating scratch space
  _mm_free(allData);

}

