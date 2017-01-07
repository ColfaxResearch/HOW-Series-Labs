#include <cmath>
#include <cstdio>
#include <omp.h>

struct ParticleSet { 
  float *x, *y, *z;
  float *vx, *vy, *vz; 
};

void MoveParticles(const int nParticles, ParticleSet& particle, const float dt) {

  // Loop over particles that experience force
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nParticles; i++) { 

    // Components of the gravity force on particle i
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f; 
      
    // Loop over particles that exert force: vectorization expected here
#pragma vector aligned
    for (int j = 0; j < nParticles; j++) { 
      
      // Avoid singularity and interaction with self
      const float softening = 1e-20f;

      // Newton's law of universal gravity
      const float dx = particle.x[j] - particle.x[i];
      const float dy = particle.y[j] - particle.y[i];
      const float dz = particle.z[j] - particle.z[i];
      const float rr = 1.0f/sqrtf(dx*dx + dy*dy + dz*dz + softening);
      const float drPowerN32  = rr*rr*rr;
	
      // Calculate the net force
      Fx += dx * drPowerN32;  
      Fy += dy * drPowerN32;  
      Fz += dz * drPowerN32;

    }

    // Accelerate particles in response to the gravitational force
    particle.vx[i] += dt*Fx; 
    particle.vy[i] += dt*Fy; 
    particle.vz[i] += dt*Fz;
  }

  // Move particles according to their velocities
  // O(N) work, so using a serial loop
#pragma simd
#pragma vector aligned
  for (int i = 0 ; i < nParticles; i++) { 
    particle.x[i]  += particle.vx[i]*dt;
    particle.y[i]  += particle.vy[i]*dt;
    particle.z[i]  += particle.vz[i]*dt;
  }
}

int main(const int argc, const char** argv) {

  // Problem size and other parameters
  const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  const int nSteps = 10;  // Duration of test
  const float dt = 0.01f; // Particle propagation time step

  // Particle data stored as a Structure of Arrays (SoA)
  // this may not be good in object-oriented programming,
  // however, makes vectorization much more efficient
  ParticleSet particle; 
  particle.x  = (float*) _mm_malloc(sizeof(float)*nParticles,64);
  particle.y  = (float*) _mm_malloc(sizeof(float)*nParticles,64);
  particle.z  = (float*) _mm_malloc(sizeof(float)*nParticles,64);
  particle.vx = (float*) _mm_malloc(sizeof(float)*nParticles,64);
  particle.vy = (float*) _mm_malloc(sizeof(float)*nParticles,64);
  particle.vz = (float*) _mm_malloc(sizeof(float)*nParticles,64);

  // First touch allocation
#pragma omp parallel for
  for(int i = 0; i < nParticles; i++) {
    particle.x[i] = 0;
    particle.y[i] = 0;
    particle.z[i] = 0;
    particle.vx[i] = 0;
    particle.vy[i] = 0;
    particle.vz[i] = 0;
  }

  // Initialize random number generator and particles
  srand(0);
  for(int i = 0; i < nParticles; i++) {
    particle.x[i] = rand()/RAND_MAX;
    particle.y[i] = rand()/RAND_MAX;
    particle.z[i] = rand()/RAND_MAX;
    particle.vx[i] = rand()/RAND_MAX;
    particle.vy[i] = rand()/RAND_MAX;
    particle.vz[i] = rand()/RAND_MAX;
  }
  
  // Perform benchmark
  printf("\n\033[1mNBODY Version 03\033[0m\n");
  printf("\nPropagating %d particles using %d threads on %s...\n\n", 
	 nParticles, omp_get_max_threads(),
#ifndef __MIC__
	 "CPU"
#else
	 "MIC"
#endif
	 );
  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration is warm-up on Xeon Phi coprocessor
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  for (int step = 1; step <= nSteps; step++) {

    const double tStart = omp_get_wtime(); // Start timing
    MoveParticles(nParticles, particle, dt);
    const double tEnd = omp_get_wtime(); // End timing

    const float HztoInts   = float(nParticles)*float(nParticles-1) ;
    const float HztoGFLOPs = 20.0*1e-9*float(nParticles)*float(nParticles-1);

    if (step > skipSteps) { // Collect statistics
      rate  += HztoGFLOPs/(tEnd - tStart); 
      dRate += HztoGFLOPs*HztoGFLOPs/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e %10.3e %8.1f %s\n", 
	   step, (tEnd-tStart), HztoInts/(tEnd-tStart), HztoGFLOPs/(tEnd-tStart), (step<=skipSteps?"*":""));
    fflush(stdout);
  }
  rate/=(double)(nSteps-skipSteps); 
  dRate=sqrt(dRate/(double)(nSteps-skipSteps)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");
  _mm_free( particle.x );
  _mm_free( particle.y );
  _mm_free( particle.z );
  _mm_free( particle.vx );
  _mm_free( particle.vy );
  _mm_free( particle.vz );
}
