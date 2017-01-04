


#ifndef __OPTIONS_H__
#define __OPTIONS_H__

// Constants for MPI communication
const int reportingRank = 0;
const int msgReportLength = 8;
const int msgReportTag = 1;
const int msgSchedLength = 8;
const int msgSchedTag    = 2;
const int msgNameTag     = 3;
const int terminate_val  = -1;
#define hostNameLen 128
typedef char HostNameType[128];

struct OptionType {

  float S; // Starting price
  float K; // Strike price
  float r; // Price drift
  float v; // Price volatility
  float T; // Option expiration time
  int numIntervals; // Number of intervals for averaging
  int numPaths; // Number of Monte Carlo paths for pricing

};

struct PayoffType {

  float payoff_geom_put; // Payoff of "put" option with geometric average
  float payoff_arithm_put; // Payoff of "put" option with arithmetic average
  float payoff_geom_call; // Payoff of "put" option with geometric average
  float payoff_arithm_call; // Payoff of "call" option with arithmetic a

};

void ComputeOptionPayoffs(const OptionType & option, PayoffType& payoff);
void ComputeOnAllNodes(const int nOptions, 
		       const OptionType* const option,
		       PayoffType* payoff,
		       const int mpiWorldSize,
		       int* rankTypes,
		       const int myRank,
		       double & comp_time,
		       int & optioncount
		       );

#endif
