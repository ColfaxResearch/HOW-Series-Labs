/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.04-threading-misc-histogram/solutions/instruction-01/worker.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



// Inputs;
// float* age        - The array of size "n" containing the ages of the sample
// int* hist         - The destination array for histogram, with size "m"
// int n             - Size of "age"
// float group_width - Size of each "bin" in the histogram
// int m             - Size of array "hist" 

void Histogram(const float* age, int* const hist, const int n, const float group_width,
    const int m) {

  // Precomputing the reciprocal of the group width
  const float recGroupWidth = 1.0f/group_width;

  // Populating the histogram.
  for (int i = 0; i < n; i++) { 

    // Calculating the index of the bin age[i] goes to.
    // Division was replaced with multiplication to improve performance
    const int j = (int) ( age[i] * recGroupWidth );

    // Incrementing the appropriate bin in the histogram.
    hist[j]++;
  }
}
