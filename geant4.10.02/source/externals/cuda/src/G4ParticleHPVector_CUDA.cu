#include <cuda.h>
#include <cuda_runtime.h>
#include "G4ParticleHPVector_CUDA.hh"

/***********************************************
*   CUDA functions
***********************************************/
__global__ void cudaTimes(G4double factor, G4ParticleHPDataPoint* cudaTheData, G4double* cudaTheIntegral) {
    int tid = blockIdx.x;
    cudaTheData[tid].xSec = cudaTheData[tid].xSec*factor;
    cudaTheIntegral[tid] = cudaTheIntegral[tid]*factor;
}
__global__ void cudaGetXsecIndex(G4double e, G4ParticleHPDataPoint* cudaTheData) {
    int tid = blockIdx.x;
    if (cudaTheData[tid].xSec >= e) {
        // return tid;
    } else {
        // return -1;
    }
}
__global__ void cudaSetIfIndexValid(G4ParticleHPDataPoint* cudaTheData, G4int offset, G4int* validIndicesOnGpu, G4double e) {
    int tid = blockIdx.x;
    int index = tid*offset;
    if (cudaTheData[index].energy >= e) {
        validIndicesOnGpu[tid] = 1;
    } else {
        validIndicesOnGpu[tid] = 0;
    }
}


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
* Functions from .cc
******************************************/
// G4ParticleHPVector_CUDA & operatorPlus (G4ParticleHPVector & left, G4ParticleHPVector & right) {

// }

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
