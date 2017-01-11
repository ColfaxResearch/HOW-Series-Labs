// Inputs;
// float* age        - The array of size "n" containing the ages of the sample
// int* hist         - The destination array for histogram, with size "m"
// int n             - Size of "age"
// float group_width - Size of each "bin" in the histogram
// int m             - Size of array "hist" 

void Histogram(const float* age, int* const hist, const int n, const float group_width,
    const int m) {

  // Populating the histogram.
  for (int i = 0; i < n; i++) { 

    // Calculating the index of the bin age[i] goes to.
    const int j = (int) ( age[i] / group_width );

    // Incrementing the appropriate bin in the histogram.
    hist[j]++;
  }
}




