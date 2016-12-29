


#include <cmath>
#include <cstdio>
#include <mkl_vsl.h>
#include <omp.h>

struct ParticleSet { 
  float *x, *y, *z;
  float *vx, *vy, *vz; 
};

void MoveParticles(const int nParticles, ParticleSet& particle, const float dt) {

  // Loop over particles that experience force
#pragma omp parallel for
  for (int i = 0; i < nParticles; i++) { 

    // Components of the gravity force on particle i
    float Fx = 0, Fy = 0, Fz = 0; 
      
    // Loop over particles that exert force: vectorization expected here
    for (int j = 0; j < nParticles; j++) { 
      
      // Avoid singularity and interaction with self
      const float softening = 1e-20;

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
  particle.x  = new float[nParticles];
  particle.y  = new float[nParticles];
  particle.z  = new float[nParticles];
  particle.vx = new float[nParticles];
  particle.vy = new float[nParticles];
  particle.vz = new float[nParticles];

  // Initialize random number generator and particles
  VSLStreamStatePtr rnStream;  
  vslNewStream( &rnStream, VSL_BRNG_MT19937, 1 );
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, nParticles, particle.x,  -1.0f, 1.0f);
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, nParticles, particle.y,  -1.0f, 1.0f);
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, nParticles, particle.z,  -1.0f, 1.0f);
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, nParticles, particle.vx, -1.0f, 1.0f);
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, nParticles, particle.vy, -1.0f, 1.0f);
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, nParticles, particle.vz, -1.0f, 1.0f);
  
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
  delete particle.x;
  delete particle.y;
  delete particle.z;
  delete particle.vx;
  delete particle.vy;
  delete particle.vz;
}
