/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.11-MPI-load-balancing-asian-options/pricing.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission 
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/     */



#include <cmath>
#include <mkl_vsl.h>
#include <omp.h>
#include "options.h"

#define ALIGNED __attribute__((aligned(64)));

void ComputeOptionPayoffs(const OptionType & option, // Parameters of option to price
			  PayoffType& payoff // Computed option payoff (output)
			  ) { 

  float payoff_geom_put=0.0f, payoff_arithm_put=0.0f;
  float payoff_geom_call=0.0f, payoff_arithm_call=0.0f;

  // Shared variables for computation
  const int batch_size = 1024;
  const int num_sims_vec = option.numPaths / batch_size; 
  const float log2e = log2f(expf(1.0f));
  const float dt = option.T / (float)option.numIntervals; // time interval between samples (in years)
  const float drift = dt*(option.r-0.5f*option.v*option.v)*log2e;
  const float vol = option.v*sqrtf(dt)*log2e;
  const float recipIntervals = 1.0f/(float)option.numIntervals;
  const float logS = logf(option.S);   
    
  // Starting a thread-parallel region
  // which will exist until the boss terminates this worker
#pragma omp parallel reduction(+: payoff_geom_put, payoff_arithm_put, payoff_geom_call, payoff_arithm_call)
  {
    // For each thread, initialize a random number stream
    VSLStreamStatePtr stream; // stream for random numbers
    int seed = omp_get_thread_num();
    int errcode = vslNewStream( &stream, VSL_BRNG_MT2203, seed );

    // For each thread, initialize scratch variables
    // These quantities will evolve multiple random paths
    // in each thread using SIMD instructions (vector operations)
    float spot_prices      [batch_size] ALIGNED;
    float rands            [batch_size] ALIGNED;
    float sumsm            [batch_size] ALIGNED;
    float sumsg            [batch_size] ALIGNED;
    float geom_mean        [batch_size] ALIGNED;
    float arithm_mean      [batch_size] ALIGNED;
    float geom_mean_put    [batch_size] ALIGNED;
    float arithm_mean_put  [batch_size] ALIGNED;
    float geom_mean_call   [batch_size] ALIGNED;
    float arithm_mean_call [batch_size] ALIGNED;


    // Team threads to process this loop in parallel 
    // The Asian option payoff calculation begins here
#pragma omp for schedule(guided)

    // Loop over the requested number of randomly simulated paths
    // (divided by the number of paths simultaneously computed in each thread
    // using vector instructions)
    for ( int i = 0; i < num_sims_vec; i++){
	    
      // Each thread carries out calculations for multiple paths.
      // Initializing the state variables for these paths.
      for ( int k = 0; k < batch_size; k++) {
	spot_prices[k] = option.S; // initialize underlying assets
	sumsm[k] = option.S;       // initialize sum vector (arithmetic mean)
	sumsg[k] = logS;    // initialize sum vector (geometric mean)
      }

      // Loop over time intervals at which the Asian option price
      // is averaged
      for ( int j = 1; j < option.numIntervals; j++){
	vsRngGaussian( VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, batch_size, rands, 0.0f, 1.0f);
	      
	// Do the calculation for batch_size paths at in each thread
	// This loop is automatically vectorized by the compiler
	// This is loop is the performance-critical part of the calculation
	for ( int k = 0; k < batch_size; k++){

	  spot_prices[k] *= exp2f(drift + vol*rands[k]); // Stochastic evolution of price
	  sumsm[k] += spot_prices[k];         // arithmetic mean
	  sumsg[k] += log2f(spot_prices[k]);   // geometric mean
	  
	}

      } // End of loop over time intervals

      // Computing the payoff for call and put options
      for ( int k = 0; k < batch_size; k++) {
	geom_mean_put[k]    = option.K - exp2f(sumsg[k] * recipIntervals); // put option
	geom_mean_call[k]   = - geom_mean_put[k];                   // call option
	arithm_mean_put[k]  = option.K - (sumsm[k] * recipIntervals);      // put option
	arithm_mean_call[k] = - arithm_mean_put[k];                 // call option
	if (geom_mean_put[k]    < 0.0f) geom_mean_put[k]     = 0.0f;
	if (geom_mean_call[k]   < 0.0f) geom_mean_call[k]    = 0.0f;
	if (arithm_mean_call[k] < 0.0f) arithm_mean_call[k]  = 0.0f;
	if (arithm_mean_put[k]  < 0.0f) arithm_mean_put[k]   = 0.0f;
      }

      // Reduction of paths calculated in vector lanes
      // into scalar variables.
      // Simultaneously with reduction across vector lanes,
      // OpenMP reduces these scalars across threads
      for ( int k=0; k < batch_size; k++) {
	payoff_geom_put    += geom_mean_put[k];
	payoff_geom_call   += geom_mean_call[k];
	payoff_arithm_put  += arithm_mean_put[k];
	payoff_arithm_call += arithm_mean_call[k];
      }

    } // End of loop over the random paths

    vslDeleteStream( &stream );

  }

  // Rescaling the quantities to convert sums into means
  const float scale = expf(-option.r*option.T)/((float)num_sims_vec*(float)batch_size);
  payoff.payoff_geom_put    = payoff_geom_put*scale;
  payoff.payoff_geom_call   = payoff_geom_call*scale;
  payoff.payoff_arithm_put  = payoff_arithm_put*scale;
  payoff.payoff_arithm_call = payoff_arithm_call*scale;
                
}
