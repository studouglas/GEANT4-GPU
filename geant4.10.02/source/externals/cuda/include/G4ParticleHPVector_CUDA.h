#ifndef G4ParticleHPVector_CUDA_h
#define G4ParticleHPVector_CUDA_h 1

#include <stdio.h>
#include <algorithm> // for std::max
#include <cfloat>
#include "G4ParticleHPDataPoint_CUDA.cu"

class G4ParticleHPVector_CUDA 
{
    public:
    G4ParticleHPVector_CUDA();
    G4ParticleHPVector_CUDA(int);
    void SetNEntries(int*);
    void SetNPoints(int*);
	void Times(double);
    double GetXsec(double);

    private:
	// typedef struct G4ParticleHPDataPoint_CUDA {
	// 	double x;
	// 	double y;
	// } G4ParticleHPDataPoint_CUDA;
	
    G4ParticleHPDataPoint_CUDA * theData;

	int * nEntries;
	int * nPoints;
	// int * isFreed;
	// double * label;
	// double * theIntegral;
	// double ** totalIntegral;
	// double * maxValue;
	// double * the15PercentBorderCash;
	// double * the50PercentBorderCash;
};

#endif
