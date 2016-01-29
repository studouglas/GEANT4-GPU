#include <cuda.h>
#include <cuda_runtime.h>
#include "G4ParticleHPVector_CUDA.hh"

/***********************************************
*   CUDA functions
***********************************************/



/***********************************************
*   Constructors, Deconstructors
***********************************************/
G4ParticleHPVector_CUDA::G4ParticleHPVector_CUDA()      { }
G4ParticleHPVector_CUDA::G4ParticleHPVector_CUDA(int n) { }
G4ParticleHPVector_CUDA::~G4ParticleHPVector_CUDA() {
    if (theData) {
        cudaFree(theData);
    }
    if (theIntegral) {
       cudaFree(theIntegral);
    }
}

/******************************************
* Getters from .hh that use CUDA
******************************************/
const G4ParticleHPDataPoint & GetPoint(G4int i) {
    return 0;
}

G4double GetEnergy(G4int i) {
    G4ParticleHPDataPoint *p;
    cudaMemcpy(p, theData[i], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    return p.energy;
}

G4double GetX(G4int i) {
    G4ParticleHPDataPoint *p;
    cudaMemcpy(p, theData[i], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    return p.energy;
}

G4double GetXsec(G4int i) {
    G4ParticleHPDataPoint *p;
    cudaMemcpy(p, theData[i], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    return p.xSec;
}

G4double GetXsec(G4double e, G4int min) {
    return 0;
}

G4double GetY(G4double x) {
    return 0;
}

G4double GetY(G4int i) {
    G4ParticleHPDataPoint *p;
    cudaMemcpy(p, theData[i], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    return p.xSec;
}

G4double GetMeanX() {
    return 0;
}

/******************************************
* Setters from .hh that use CUDA
******************************************/
void SetData(G4int i, G4double x, G4double y) {

}

void SetX(G4int i, G4double e) {

}

void SetEnergy(G4int i, G4double e) {

}

void SetY(G4int i, G4double x) {

}

void SetXsec(G4int i, G4double x {

}


/******************************************
* Computations from .hh that use CUDA
******************************************/
void Init(std::istream & aDataFile,G4double ux=1., G4double uy=1.) {

}

void CleanUp() {

}

G4double SampleLin() {
    return 0;
}

G4double * Debug() {
    return 0;
}

void Integrate() {

}

void IntegrateAndNormalise() {

}

void Times(G4double factor) {
    cudaTimes(factor, theData, theIntegral);
}
__global__ void cudaTimes(G4double factor, G4ParticleHPDataPoint* theData, G4double* theIntegral) {
    int tid = blockIdx.x;
    theData[tid].xSec = theData[tid].xSec*factor;
    theIntegral[tid] = theIntegral[tid]*factor;
}

/******************************************
* Functions from .cc
******************************************/
G4double G4ParticleHPVector_CUDA::GetXsec(G4double e) {
    return 0;
}

void G4ParticleHPVector_CUDA::Dump() {

}

void G4ParticleHPVector_CUDA::ThinOut(G4double precision) {

}

void G4ParticleHPVector_CUDA::Merge(G4InterpolationScheme aScheme, G4double aValue, G4ParticleHPVector_CUDA * active, G4ParticleHPVector_CUDA * passive) {

}

G4double G4ParticleHPVector_CUDA::Sample() {
    return 0;
}

G4double G4ParticleHPVector_CUDA::Get15percentBorder() {
    return 0;
}

G4double G4ParticleHPVector_CUDA::Get50percentBorder() {
    return 0;
}

void G4ParticleHPVector_CUDA::Check(G4int i) {

}

G4bool G4ParticleHPVector_CUDA::IsBlocked(G4double aX) {
    return false;
}

// G4ParticleHPVector_CUDA & operatorPlus (G4ParticleHPVector & left, G4ParticleHPVector & right) { }
