#ifndef G4ParticleHPVector_CUDA_h
#define G4ParticleHPVector_CUDA_h 1

#include <stdio.h>
#include <algorithm> // for std::max
#include <cfloat>
#include "G4ParticleHPDataPoint.hh"

class G4ParticleHPVector_CUDA {
    
    /******************************************
  	 * PUBLIC
  	 ******************************************/
    public:
    G4ParticleHPVector_CUDA();
    G4ParticleHPVector_CUDA(int);
    ~G4ParticleHPVector_CUDA();
    void SetNEntries(int*);
    void SetNPoints(int*);
    void SetTheData(G4ParticleHPDataPoint**);
	void SetTheIntegral(double**);
	
	void SetTheDataChangedOnCpu();

	void CopyTheDataToGpuIfChanged();
	void CopyTheDataToCpuIfChanged();
	
	void Times(double);
    double GetXsec(double);
    // double Get15PercentBorder();
    // double Get50PercentBorder();
    // void IntegrateAndNormalize();
    // double GetMeanX();

	
	/******************************************
  	 * PRIVATE                                 
  	 ******************************************/
    private:
	void SetTheDataChangedOnGpu();

	G4ParticleHPDataPoint ** theData;
	double ** theIntegral;
	int * nEntries;
	int * nPoints;
	
	G4ParticleHPDataPoint* cudaTheData;
    double* cudaTheIntegral;
    int cudaTheDataSize;
    
    bool theDataChangedOnCpuBool;
    bool theDataChangedOnGpuBool;
};

#endif
