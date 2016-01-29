#include <cuda.h>
#include <cuda_runtime.h>
#include "G4ParticleHPVector_CUDA.hh"

/***********************************************
*   CUDA functions
***********************************************/
__global__ void firstIndexGreaterThan(G4ParticleHPDataPoint * theDataArg, G4double e, int* resultIndex) {
    int startIndex = blockDim.x * blockIdx.x * threadIdx.x;
    if (theDataArg[startIndex].energy > e) {
        atomicMin(resultIndex, startIndex);
    }
}

/***********************************************
*   Constructors, Deconstructors
***********************************************/
G4ParticleHPVector_CUDA::G4ParticleHPVector_CUDA()      { 
    nPoints = 20;
    cudaMalloc(&theData, nPoints*sizeof(G4ParticleHPDataPoint));
    nEntries = 0;
    Verbose = 0;
    theIntegral = 0; // TODO: cuda malloc ?
    totalIntegral = -1;
    isFreed = 0;
    maxValue = -DBL_MAX;
    the15percentBorderCash = -DBL_MAX;
    the50percentBorderCash = -DBL_MAX;
    label = -DBL_MAX;
}

G4ParticleHPVector_CUDA::G4ParticleHPVector_CUDA(int n) { 
    nPoints = std::max(n,20);
    cudaMalloc(&theData, nPoints*sizeof(G4ParticleHPDataPoint));
    nEntries = 0;
    Verbose = 0;
    theIntegral = 0; // TODO: cuda malloc ?
    totalIntegral = -1;
    isFreed = 0;
    maxValue = -DBL_MAX;
    the15percentBorderCash = -DBL_MAX;
    the50percentBorderCash = -DBL_MAX;
    label = -DBL_MAX;
}

G4ParticleHPVector_CUDA::~G4ParticleHPVector_CUDA() {
    if (theData) {
        cudaFree(theData);
        theData = NULL;
    }
    if (theIntegral) {
       cudaFree(theIntegral);
       theIntegral = NULL;
    }
    isFreed = 1;
}

/******************************************
* Getters from .hh that use CUDA
******************************************/
// TODO: check for memory leak
const G4ParticleHPDataPoint & G4ParticleHPVector_CUDA::GetPoint(G4int i) {
    G4ParticleHPDataPoint* point;
    cudaMemcpy(point, &theData[i], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    return *(point);
}

G4double G4ParticleHPVector_CUDA::GetEnergy(G4int i) {
    G4double energy;
    cudaMemcpy(&energy, &theData[i].energy, sizeof(G4double), cudaMemcpyDeviceToHost);
    return energy;
}

G4double G4ParticleHPVector_CUDA::GetX(G4int i) {
    G4double energy;
    cudaMemcpy(&energy, &theData[i].energy, sizeof(G4double), cudaMemcpyDeviceToHost);
    return energy;
}

G4double G4ParticleHPVector_CUDA::GetXsec(G4int i) {
    G4double xSec;
    cudaMemcpy(&xSec, &theData[i].xSec, sizeof(G4double), cudaMemcpyDeviceToHost);
    return xSec;
}

// TODO: Port Me
G4double G4ParticleHPVector_CUDA::GetXsec(G4double e, G4int min) {
    return 0;
}

G4double G4ParticleHPVector_CUDA::GetY(G4double x) {
    return GetXsec(x);
}

G4double G4ParticleHPVector_CUDA::GetY(G4int i) {
    G4double xSec;
    cudaMemcpy(&xSec, &theData[i].xSec, sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    return xSec;
}

// TODO: Port Me
G4double G4ParticleHPVector_CUDA::GetMeanX() {
    return 0;
}

/******************************************
* Setters from .hh that use CUDA
******************************************/
void G4ParticleHPVector_CUDA::SetData(G4int i, G4double x, G4double y) {
    G4ParticleHPDataPoint point;
    point.energy = x;
    point.xSec = y;
    cudaMemcpy(&theData[i], &point, sizeof(G4ParticleHPDataPoint), cudaMemcpyHostToDevice);
}

void G4ParticleHPVector_CUDA::SetX(G4int i, G4double e) {
    cudaMemcpy(&theData[i].energy, &e, sizeof(G4double), cudaMemcpyHostToDevice);
}

void G4ParticleHPVector_CUDA::SetEnergy(G4int i, G4double e) {
    cudaMemcpy(&theData[i].energy, &e, sizeof(G4double), cudaMemcpyHostToDevice);
}

void G4ParticleHPVector_CUDA::SetY(G4int i, G4double x) {
    cudaMemcpy(&theData[i].xSec, &x, sizeof(G4double), cudaMemcpyHostToDevice);
}

void G4ParticleHPVector_CUDA::SetXsec(G4int i, G4double x) {
    cudaMemcpy(&theData[i].xSec, &x, sizeof(G4double), cudaMemcpyHostToDevice);
}


/******************************************
* Computations from .hh that use CUDA
******************************************/
void G4ParticleHPVector_CUDA::Init(std::istream & aDataFile, G4double ux, G4double uy) {
    G4int total;
    aDataFile >> total;
    if (theData) {
        cudaFree(theData);
    }
    cudaMalloc(&theData, sizeof(G4ParticleHPDataPoint) * total);
    nPoints = total;
    nEntries = 0;
    theManager.Init(aDataFile);
    Init(aDataFile, total, ux, uy);
}

void G4ParticleHPVector_CUDA::CleanUp() {
    nEntries = 0;
    theManager.CleanUp();
    maxValue = -DBL_MAX;
    if (theIntegral) {
        cudaFree(theIntegral);
        theIntegral = NULL;
    }
}

// TODO: Port Me
G4double G4ParticleHPVector_CUDA::SampleLin() {
    G4double result;
    if (!theIntegral) {
        IntegrateAndNormalise();
    }

    if (GetVectorLength() == 1) {
        cudaMemcpy(&result, &theData[0].energy, sizeof(G4double), cudaMemcpyDeviceToHost);
    }
    else {
        G4int i;
        G4double randNum = (G4double)rand() / (G4double)RAND_MAX; // TODO: change to G4UniformRand
        // TODO: requires 'first occurence' algorithm
        // for (i = GetVectorLength() - 1; i >= 0; i--) {
        //     if (randNum > )
        // }
    }

    return result;
}

// TODO: Port Me (should return theIntegral, but how do we return somethign we don't have ref to?)
G4double * G4ParticleHPVector_CUDA::Debug() {
    return 0;
}

// TODO: Port Me
void G4ParticleHPVector_CUDA::Integrate() {
    G4int i;
    if (nEntries == 1) {
        totalIntegral = 0;
        return;
    }

    G4double sum = 0;
    // cudaIntegrate<<<1, nEntries>>>(&sum);
    totalIntegral = sum;
}

// TODO: Port Me
void G4ParticleHPVector_CUDA::IntegrateAndNormalise() {

}

__global__ void cudaTimes(G4double factor, G4ParticleHPDataPoint* theDataArg, G4double* theIntegralArg) {
    int tid = blockIdx.x;
    theDataArg[tid].xSec = theDataArg[tid].xSec*factor;
    theIntegralArg[tid] = theIntegralArg[tid]*factor;
}
void G4ParticleHPVector_CUDA::Times(G4double factor) {
    cudaTimes<<<1, nPoints>>> (factor, theData, theIntegral);
}

/******************************************
* Functions from .cc
******************************************/
// TODO: Port Me
G4double G4ParticleHPVector_CUDA::GetXsec(G4double e) {
    int *resultIndex;
    cudaMalloc(&resultIndex, sizeof(int));
    cudaMemcpy(&resultIndex, &nEntries, sizeof(int), cudaMemcpyHostToDevice);
    
    firstIndexGreaterThan<<<1, nEntries>>> (theData, e, resultIndex);
    
    G4int i = 0;
    cudaMemcpy(&i, resultIndex, sizeof(G4int), cudaMemcpyDeviceToHost);
    G4double resultVal = 0;
    cudaMemcpy(&resultVal, &theData[i].xSec, sizeof(G4int), cudaMemcpyDeviceToHost);
    
    G4int low = i - 1;
    G4int high = i;
    if (i == 0) {
        low = 0;
        high = 1;
    }
    else if (i == nEntries) {
        low = nEntries - 2;
        high = nEntries - 1;
    }

    G4double y;
    G4ParticleHPDataPoint pointNentriesMinusOne;
    cudaMemcpy(&pointNentriesMinusOne, &theData[nEntries-1], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    
    if (e < pointNentriesMinusOne.energy) {
        G4ParticleHPDataPoint theDataLow;
        G4ParticleHPDataPoint theDataHigh;
        cudaMemcpy(&theDataLow, &theData[low], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
        cudaMemcpy(&theDataHigh, &theData[high], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);

        if((theDataHigh.energy - theDataLow.energy) / theDataHigh.energy < 0.000001) {
            y = theDataLow.xSec;
        }
        else {
            y = theInt.Interpolate(theManager.GetScheme(high), e, 
                theDataLow.energy, theDataHigh.energy,
                theDataLow.xSec, theDataHigh.xSec);
        }
    }
    else {
        y = pointNentriesMinusOne.xSec;
    }

    return y;
}

void G4ParticleHPVector_CUDA::Dump() {
    G4ParticleHPDataPoint *localTheData = (G4ParticleHPDataPoint*)malloc(nPoints * sizeof(G4ParticleHPDataPoint));
    cudaMemcpy(localTheData, theData, nPoints * sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    
    std::cout << nEntries << std::endl;
    for (G4int i = 0; i < nPoints; i++) {
        std::cout << localTheData[i].GetX() << " ";
        std::cout << localTheData[i].GetY() << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    free(localTheData);
}

// TODO: Port Me
void G4ParticleHPVector_CUDA::ThinOut(G4double precision) {

}

// TODO: Port Me
void G4ParticleHPVector_CUDA::Merge(G4InterpolationScheme aScheme, G4double aValue, G4ParticleHPVector_CUDA * active, G4ParticleHPVector_CUDA * passive) {

}

// TODO: Port Me
G4double G4ParticleHPVector_CUDA::Sample() {
    return 0;
}

// TODO: Port Me
G4double G4ParticleHPVector_CUDA::Get15percentBorder() {
    return 0;
}

// TODO: Port Me
G4double G4ParticleHPVector_CUDA::Get50percentBorder() {
    return 0;
}

// TODO: Port Me
void G4ParticleHPVector_CUDA::Check(G4int i) {

}

// Note: Geant4 doesn't ever assign private variable theBlocked,
// which means their IsBlocked function always returns false
G4bool G4ParticleHPVector_CUDA::IsBlocked(G4double aX) {
    return false;
}

// G4ParticleHPVector_CUDA:: & operatorPlus (G4ParticleHPVector & left, G4ParticleHPVector & right) { }
