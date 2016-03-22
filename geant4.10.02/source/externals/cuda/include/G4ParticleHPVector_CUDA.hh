#ifndef G4ParticleHPVector_CUDA_h
#define G4ParticleHPVector_CUDA_h 1


#include <iostream>
#include <algorithm> // for std::max
#include "G4ParticleHPDataPoint_CUDA.hh"
#include "G4ParticleHPInterpolator_CUDA.hh"
#include "G4InterpolationScheme_CUDA.hh"
#include "G4InterpolationManager_CUDA.hh"
#include "G4Types_CUDA.hh"
#include "G4Pow_CUDA.hh"

#define THREADS_PER_BLOCK 32 // must be multiple of 32

typedef struct GetXsecResultStruct {
    // if -1, other elements in struct are non-null
    G4double y; 

    G4ParticleHPDataPoint pointLow;
    G4ParticleHPDataPoint pointHigh;
    G4int indexHigh;
} GetXsecResultStruct;

class G4ParticleHPVector_CUDA {
    public:
    void GetXsecList(G4double* energiesIn_xSecsOut, G4int length, G4ParticleHPDataPoint* theData, G4int nEntries);

    private:
    G4InterpolationManager theManager;
    G4ParticleHPInterpolator theInt;
};

#endif
