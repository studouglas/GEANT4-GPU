#include <cuda.h>
#include <cuda_runtime.h>
#include "G4ParticleHPVector_CUDA.hh"

/***********************************************
*   CUDA functions
***********************************************/
__global__ void SetValueTo_CUDA(int *addressToSet, int value) {
    *(addressToSet) = value;
}

// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
__device__ double AtomicAdd_CUDA(double* address, double val) { 
    unsigned long long int* address_as_ull = (unsigned long long int*)address; 
    unsigned long long int old = *address_as_ull, assumed; 
    do { 
        assumed = old; 
        old = atomicCAS(address_as_ull, assumed, 
            __double_as_longlong(val + __longlong_as_double(assumed))); 
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
    } while (assumed != old); 
    return __longlong_as_double(old); 
}

__global__ void CopyDataPointsToBuffer_CUDA(G4ParticleHPDataPoint * fromBuffer, G4ParticleHPDataPoint * toBuffer, G4int nEntries) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nEntries) {
        toBuffer[i].energy = fromBuffer[i].energy;
        toBuffer[i].xSec = fromBuffer[i].xSec;
    }
}

/***********************************************
*   Constructors, Deconstructors
***********************************************/
G4ParticleHPVector_CUDA::G4ParticleHPVector_CUDA()      { 
    // printf("\nConstructor called");
    nPoints = 20;
    cudaMalloc(&d_theData, nPoints*sizeof(G4ParticleHPDataPoint));
    nEntries = 0;
    Verbose = 0;
    d_theIntegral = 0; // TODO: cuda malloc ?
    totalIntegral = -1;
    isFreed = 0;
    maxValue = -DBL_MAX;
    the15percentBorderCash = -DBL_MAX;
    the50percentBorderCash = -DBL_MAX;
    label = -DBL_MAX;
}

G4ParticleHPVector_CUDA::G4ParticleHPVector_CUDA(int n) { 
    // printf("\nConstructor(%d) called", n);
    nPoints = std::max(n,20);
    cudaMalloc(&d_theData, nPoints*sizeof(G4ParticleHPDataPoint));
    nEntries = 0;
    Verbose = 0;
    d_theIntegral = 0; // TODO: cuda malloc ?
    totalIntegral = -1;
    isFreed = 0;
    maxValue = -DBL_MAX;
    the15percentBorderCash = -DBL_MAX;
    the50percentBorderCash = -DBL_MAX;
    label = -DBL_MAX;
    // printf("\ndone counstructor(n)");
}

G4ParticleHPVector_CUDA::~G4ParticleHPVector_CUDA() {
    if (d_theData) {
        cudaFree(d_theData);
        d_theData = NULL;
    }
    if (d_theIntegral) {
       cudaFree(d_theIntegral);
       d_theIntegral = NULL;
    }
    isFreed = 1;
}

/******************************************
* Getters from .hh that use CUDA
******************************************/
const G4ParticleHPDataPoint & G4ParticleHPVector_CUDA::GetPoint(G4int i) {
    G4ParticleHPDataPoint point;
    cudaMemcpy(&point, &d_theData[i], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    return point; // TODO: warning: returning referecne to local variable
}

// G4double G4ParticleHPVector_CUDA::GetEnergy(G4int i) {
    
// }

G4double G4ParticleHPVector_CUDA::GetX(G4int i) {
    if (i < 0) {
        i = 0;
    }
    if (i >= GetVectorLength()) {
        i = GetVectorLength() - 1;
    }

    G4double energy;
    cudaMemcpy(&energy, &d_theData[i].energy, sizeof(G4double), cudaMemcpyDeviceToHost);

    if (energy != energy) {
        printf("\nGetEnergy(%d) = %f, nEntries=%d", i, energy, nEntries);
        // Dump();
    }
    return energy;
}

// G4double G4ParticleHPVector_CUDA::GetXsec(G4int i) {
//     if (i < 0) {
//         i = 0;
//     }
//     if (i >= GetVectorLength()) {
//         i = GetVectorLength() - 1;
//     }
//     G4double xSec;
//     cudaMemcpy(&xSec, &d_theData[i].xSec, sizeof(G4double), cudaMemcpyDeviceToHost);
//     return xSec;
// }

G4double G4ParticleHPVector_CUDA::GetY(G4double x) {
    return GetXsec(x);
}

G4double G4ParticleHPVector_CUDA::GetY(G4int i) {    
    if (i < 0) {
        i = 0;
    }
    if (i >= GetVectorLength()) {
        i = GetVectorLength() - 1;
    }

    G4double xSec;
    cudaMemcpy(&xSec, &d_theData[i].xSec, sizeof(G4double), cudaMemcpyDeviceToHost);
    return xSec;
}

// TODO: Port Me (requires 1st element predicate alg.)
G4double G4ParticleHPVector_CUDA::GetXsec(G4double e, G4int min) {
    printf("\nGetXsec(e,min) is not yetimplemented");
    return 0;
}

// TODO: Port Me (requires interpolation)
G4double G4ParticleHPVector_CUDA::GetMeanX() {
    printf("\nGET MEAN X NOT YET IMPLEMENTED");
    return 0;
}

/******************************************
* Setters from .hh that use CUDA
******************************************/
void G4ParticleHPVector_CUDA::SetData(G4int i, G4double x, G4double y) {
    Check(i);
    G4ParticleHPDataPoint point;
    point.energy = x;
    point.xSec = y;
    cudaMemcpy(&d_theData[i], &point, sizeof(G4ParticleHPDataPoint), cudaMemcpyHostToDevice);
    if (x != x || y != y) {
        printf("\nSetData got passed NAN!, SetData(%d, %0.5e, %0.5e)", i, x, y);
    }
}

void G4ParticleHPVector_CUDA::SetX(G4int i, G4double e) {
    Check(i);
    cudaMemcpy(&d_theData[i].energy, &e, sizeof(G4double), cudaMemcpyHostToDevice);
    if (e != e) {
        printf("\nSetX(%d) got passed NAN!!!", i);
    }
}

void G4ParticleHPVector_CUDA::SetEnergy(G4int i, G4double e) {
    Check(i);
    cudaMemcpy(&d_theData[i].energy, &e, sizeof(G4double), cudaMemcpyHostToDevice);
    if (e != e) {
        printf("\nSetenergy(%d) got passed NAN!!!", i);
    }
}

void G4ParticleHPVector_CUDA::SetY(G4int i, G4double x) {
    Check(i);
    cudaMemcpy(&d_theData[i].xSec, &x, sizeof(G4double), cudaMemcpyHostToDevice);
    if (x != x) {
        printf("\nSety(%d) got passed NAN!!!", i);
    }
}

void G4ParticleHPVector_CUDA::SetXsec(G4int i, G4double x) {
    Check(i);
    cudaMemcpy(&d_theData[i].xSec, &x, sizeof(G4double), cudaMemcpyHostToDevice);
    if (x != x) {
        printf("\nSetXsec(%d) got passed NAN!!!", i);
    }
}


/******************************************
* Computations from .hh that use CUDA
******************************************/
void G4ParticleHPVector_CUDA::Init(std::istream & aDataFile, G4double ux, G4double uy) {
    printf("\nInit called");
    G4int total;
    aDataFile >> total;
    if (d_theData) {
        cudaFree(d_theData);
    }
    cudaMalloc(&d_theData, sizeof(G4ParticleHPDataPoint) * total);
    nPoints = total;
    nEntries = 0;
    theManager.Init(aDataFile);
    Init(aDataFile, total, ux, uy);
}

void G4ParticleHPVector_CUDA::CleanUp() {
    printf("\nCUDA - CleanUp (nEntries: %d", nEntries);
    nEntries = 0;
    theManager.CleanUp();
    maxValue = -DBL_MAX;
    if (d_theIntegral) {
        cudaFree(d_theIntegral);
        d_theIntegral = NULL;
    }
}

__global__ void SampleLinFindLastIndex_CUDA(G4double * theIntegral, int rand, int * resultIndex, int nEntries) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nEntries) {
        return;
    }
    if (i > *resultIndex && theIntegral[i]/theIntegral[nEntries-1] < rand) {
        atomicMax(resultIndex, i);
    }
}
__global__ void SampleLinGetValues(G4ParticleHPDataPoint * theData, G4double * theIntegral, G4double * d_vals, G4int i) {
    // d_vals = [x1,x2,y1,y2]
    switch(threadIdx.x) {
        case 0: 
            d_vals[0] = theIntegral[i-1];
            break;
        case 1:
            d_vals[1] = theIntegral[i];
            break;
        case 2:
            d_vals[2] = theData[i-1].energy;
            break;
        case 3:
            d_vals[3] = theData[i].energy;
            break;
        default:
            printf("\nError -- invalid thread id in SampleLinGetValues(), returning");
    }
}

G4double G4ParticleHPVector_CUDA::SampleLin() {
    printf("\nCUDA - SampleLin (nEntries: %d", nEntries);
    G4double result;
    if (!d_theIntegral) {
        IntegrateAndNormalise();
    }

    if (GetVectorLength() == 1) {
        cudaMemcpy(&result, &d_theData[0].energy, sizeof(G4double), cudaMemcpyDeviceToHost);
    }
    else {
        // TODO: change to G4UniformRand
        G4double randNum = (G4double)rand() / (G4double)RAND_MAX; 

        int *d_resultIndex;
        cudaMalloc(&d_resultIndex, sizeof(int));
        SetValueTo_CUDA<<<1,1>>> (d_resultIndex, INT_MAX);

        int nBlocks = GetNumBlocks(nEntries);
        SampleLinFindLastIndex_CUDA<<<nBlocks, THREADS_PER_BLOCK>>> (d_theIntegral, randNum, d_resultIndex, nEntries);
        
        G4int i = 0;
        cudaMemcpy(&i, d_resultIndex, sizeof(G4int), cudaMemcpyDeviceToHost);
        if (i != GetVectorLength() - 1) {
            i++;
        }

        // vals = [x1, x2, y1, y2]
        G4double* d_vals;
        cudaMalloc(&d_vals, 4*sizeof(G4double));
        SampleLinGetValues<<<1, 4>>>(d_theData, d_theIntegral, d_vals, i);
        
        G4double vals[4];
        cudaMemcpy(vals, d_vals, 4*sizeof(G4double), cudaMemcpyDeviceToHost);
        
        result = theLin.Lin(randNum, vals[0], vals[1], vals[2], vals[3]);
        
        cudaFree(d_resultIndex);
        cudaFree(d_vals);
        free(vals);
    }

    return result;
}

// TODO: Port Me (should return d_theIntegral, but how do we return somethign we don't have ref to?)
G4double * G4ParticleHPVector_CUDA::Debug() {
    printf("\nDEBUG NOT YET IMPLEMENTED");
    return 0;
}

// TODO: test that this gives same results
__global__ void Integrate_CUDA(G4ParticleHPDataPoint * theData, G4double * sum, G4InterpolationManager theManager) {
    G4int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0) {
        return;
    }

    if (abs((theData[i].energy - theData[i-1].energy) / theData[i].energy) > 0.0000001) {
        G4double x1 = theData[i-1].energy;
        G4double x2 = theData[i].energy;
        G4double y1 = theData[i-1].xSec;
        G4double y2 = theData[i].xSec;

        double toAdd = 0;
        G4InterpolationScheme aScheme = theManager.GetScheme(i);
        if (aScheme == LINLIN || aScheme == CLINLIN || aScheme == ULINLIN) {
            toAdd += 0.5 * (y2+y1) * (x2-x1);
        }
        else if (aScheme == LINLOG || aScheme == CLINLOG || aScheme == ULINLOG) {
            G4double a = y1;
            // G4double b = (y2-y1) / (G4Log(x2) - G4Log(x1));
            // toAdd += (a-b) * (x2-x1) + b*(x2 * G4Log(x2) - x1 * G4Log(x1));
            
            // NOTE: cuda's log function requires compute capability >= 3.0 for double precision
            // make sure you are compiling for 3.0 (nvcc -arch sm_30)
            G4double b = (y2 - y1) / (log(x2) - log(x1));
            toAdd += (a-b) * (x2-x1) + b*(x2 * log(x2) - x1 * log(x1));
        }
        else if (aScheme == LOGLIN || aScheme == CLOGLIN || aScheme == ULOGLIN) {
            // G4double a = G4Log(y1);
            // G4double b = (G4Log(y2)-G4Log(y1))/(x2-x1);
            // toAdd += (G4Exp(a)/b) * (G4Exp(b * x2) - G4Exp(b * x1));

            // NOTE: cuda's log function requires compute capability >= 3.0 for double precision
            // make sure you are compiling for 3.0 (nvcc -arch sm_30)
            G4double a = log(y1);
            G4double b = (log(y2) - log(y1)) / (x2-x1);
            // toAdd += (G4Exp(a) / b) * (G4Exp(b * x2) - G4Exp(b * x1));

        }
        else if (aScheme == HISTO || aScheme == CHISTO || aScheme == UHISTO) {
            toAdd += y1 * (x2-x1);
        }
        else if (aScheme == LOGLOG || aScheme == CLOGLOG || aScheme == ULOGLOG) {
            // G4double a = G4Log(y1);
            // G4double b = (G4Log(y2) - G4Log(y1)) / (G4Log(x2) - G4Log(x1));
            // toAdd += (G4Exp(a)/(b+1)) * (G4Pow::GetInstance()->powA(x2,b+1) - G4Pow::GetInstance()->powA(x1,b+1));

            // NOTE: cuda's log function requires compute capability >= 3.0 for double precision
            // make sure you are compiling for 3.0 (nvcc -arch sm_30)
            G4double a = log(y1);
            G4double b = (log(y2) - log(y1)) / (log(x2) - log(x1));
            toAdd += (G4Exp(a)/(b+1)) * (pow(x2,b+1) - pow(x1,b+1));
        }

        if (toAdd != 0) {
            AtomicAdd_CUDA(sum, toAdd);
        }
    }
}
void G4ParticleHPVector_CUDA::Integrate() {
    printf("\nCUDA - Integrate (nEntries: %d", nEntries);
    if (nEntries == 1) {
        totalIntegral = 0;
        return;
    }
    
    G4double *d_sum;
    cudaMalloc(&d_sum, sizeof(G4double));
    Integrate_CUDA<<<1, nEntries>>>(d_theData, d_sum, theManager);
    totalIntegral = *(d_sum);
    cudaFree(d_sum);
}

// TODO: Port Me
void G4ParticleHPVector_CUDA::IntegrateAndNormalise() {
    printf("\nIntegtateandnormalize not yet implemented");
    // G4int i;
    //   if(theIntegral != 0)
    //     return;
    //   theIntegral = new G4double[nEntries];
    //   if(nEntries == 1)
    //   {
    //     theIntegral[0] = 1;
    //     return;
    //   }
    //   theIntegral[0] = 0;
    //   G4double sum = 0;
    //   G4double x1 = 0;
    //   G4double x0 = 0;
    //   for(i=1;i<GetVectorLength();i++)
    //   {
    //     x1 = theData[i].GetX();
    //     x0 = theData[i-1].GetX();
    //     if (std::abs(x1-x0) > std::abs(x1*0.0000001) )
    //     {
    //         //********************************************************************
    //         //EMendoza -> the interpolation scheme is not always lin-lin
    //         /*
    //             sum+= 0.5*(theData[i].GetY()+theData[i-1].GetY())*(x1-x0);
    //         */
    //         //********************************************************************
    //       G4InterpolationScheme aScheme = theManager.GetScheme(i);
    //       G4double y0 = theData[i-1].GetY();
    //       G4double y1 = theData[i].GetY();
    //         G4double integ=theInt.GetBinIntegral(aScheme,x0,x1,y0,y1);
    //       #if defined WIN32-VC
    //         if(!_finite(integ)){integ=0;}
    //       #elif defined __IBMCPP__
    //         if(isinf(integ)||isnan(integ)){integ=0;}
    //       #else
    //         if(std::isinf(integ)||std::isnan(integ)){integ=0;}
    //       #endif
    //         sum+=integ;
    //         //********************************************************************
    //     }
    //     theIntegral[i] = sum;
    //   }
    //   G4double total = theIntegral[GetVectorLength()-1];
    //   for(i=1;i<GetVectorLength();i++)
    //   {
    //     theIntegral[i]/=total;
    //   }
}

__global__ void Times_CUDA(G4double factor, G4ParticleHPDataPoint* theData, G4double* theIntegral, G4int nEntriesArg) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= nEntriesArg) {
        return;
    }
    theData[tid].xSec = theData[tid].xSec * factor;
    theIntegral[tid] = theIntegral[tid] * factor;
}
void G4ParticleHPVector_CUDA::Times(G4double factor) {
	printf("\nCUDA - Times (nEntries: %d", nEntries);
	int nBlocks = GetNumBlocks(nEntries);
    Times_CUDA<<<nBlocks, THREADS_PER_BLOCK>>> (factor, d_theData, d_theIntegral, nEntries);
}

/******************************************
* Functions from .cc
******************************************/
__global__ void GetXSecFirstIndex_CUDA(G4ParticleHPDataPoint * theData, G4double e, int * resultIndex, int nEntries) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nEntries && idx < *(resultIndex) && theData[idx].energy > e) {
        atomicMin(resultIndex, idx);
    }
}
G4double G4ParticleHPVector_CUDA::GetXsec(G4double e) {
	// printf("\nGetXsec called");

    if (nEntries == 0) {
        return 0;
    }

    int *d_resultIndex;
	cudaMalloc(&d_resultIndex, sizeof(int));
	SetValueTo_CUDA<<<1,1>>> (d_resultIndex, INT_MAX);
	
    int nBlocks = GetNumBlocks(nEntries);
    GetXSecFirstIndex_CUDA<<<nBlocks, THREADS_PER_BLOCK>>> (d_theData, e, d_resultIndex, nEntries);
    
    G4int i = 0;
    cudaMemcpy(&i, d_resultIndex, sizeof(G4int), cudaMemcpyDeviceToHost);
    
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
    G4ParticleHPDataPoint lastPoint;
    cudaMemcpy(&lastPoint, &d_theData[nEntries-1], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    
    if (e < lastPoint.energy) {
        G4ParticleHPDataPoint theDataLow;
        G4ParticleHPDataPoint theDataHigh;
        cudaMemcpy(&theDataLow, &d_theData[low], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
        cudaMemcpy(&theDataHigh, &d_theData[high], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);

        if ((theDataHigh.energy - theDataLow.energy) / theDataHigh.energy < 0.000001) {
            y = theDataLow.xSec;
        }
        else {
            y = theInt.Interpolate(theManager.GetScheme(high), e, 
                    theDataLow.energy, theDataHigh.energy,
                    theDataLow.xSec, theDataHigh.xSec);
        }
    }
    else {
        y = lastPoint.xSec;
    }
	
    cudaFree(d_resultIndex);
    return y;
}

void G4ParticleHPVector_CUDA::Dump() {
    printf("\nCUDA - Dump (nEntries: %d", nEntries);

    // never called, so just copy all of theData to cpu and print it out (slow, but works)
    G4ParticleHPDataPoint *localTheData = (G4ParticleHPDataPoint*)malloc(nEntries * sizeof(G4ParticleHPDataPoint));
    cudaMemcpy(localTheData, d_theData, nEntries * sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    
    std::cout << nEntries << std::endl;
    for (G4int i = 0; i < nEntries; i++) {
        printf("%0.5e %0.5e\n",localTheData[i].GetX(), localTheData[i].GetY());
    }
    std::cout << std::endl;

    free(localTheData);
}

// TODO: Make me parallel (works, but is serial so memcpy's too much)
void G4ParticleHPVector_CUDA::ThinOut(G4double precision) {
    printf("\nCUDA - ThinOut (nEntries: %d", nEntries);
    if (GetVectorLength() == 0) {
      return;
    }

    G4ParticleHPDataPoint *localTheData = (G4ParticleHPDataPoint*)malloc(nEntries*sizeof(G4ParticleHPDataPoint));
    cudaMemcpy(localTheData, d_theData, nEntries*sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    G4ParticleHPDataPoint *localBuffer = (G4ParticleHPDataPoint*)malloc(nPoints*sizeof(G4ParticleHPDataPoint));
    
    G4double x, x1, x2, y, y1, y2;
    G4int count = 0;
    G4int current = 2;
    G4int start = 1;

    // first element always goes and is never tested.
    localBuffer[0] = localTheData[0];
    // copyDataPointFromBufferToBuffer_CUDA<<<1,1>>> (d_theData, localBuffer, nEntries);

    while(current < GetVectorLength()) {
        x1 = localBuffer[count].GetX();
        y1 = localBuffer[count].GetY();
        x2 = localTheData[current].GetX();
        y2 = localTheData[current].GetY();
        
        for(G4int j=start; j<current; j++) {
            x = localTheData[j].GetX();
        
            if (x1-x2 == 0) {
                y = (y2+y1)/2.0;
            }
            else {
                y = theInt.Lin(x, x1, x2, y1, y2);
            }
            if (std::abs(y - localTheData[j].GetY()) > precision * y) {
                localBuffer[++count] = localTheData[current-1]; // for this one, everything was fine
                start = current; // the next candidate
                break;
            }
        }
        current++;
    }

    // the last one also always goes, and is never tested.
    count++;
    localBuffer[count] = localTheData[GetVectorLength() - 1];
    nEntries = count + 1;

    cudaFree(d_theData);
    cudaMemcpy(d_theData, localBuffer, nEntries * sizeof(G4ParticleHPDataPoint), cudaMemcpyHostToDevice);

    free(localTheData);
    free(localBuffer);
}

// TODO: Port Me
void G4ParticleHPVector_CUDA::Merge(G4InterpolationScheme aScheme, G4double aValue, G4ParticleHPVector_CUDA * active, G4ParticleHPVector_CUDA * passive) {
    printf("MERGE NOT YET IMPLEMENTED\n\n");
}

// TODO: Port Me
G4double G4ParticleHPVector_CUDA::Sample() {
    printf("SAMPLE NOT YET IMPLEMENTED\n\n");
    return 0;
}

// TODO: Port Me
G4double G4ParticleHPVector_CUDA::Get15percentBorder() {
    printf("Get 15 NOT YET IMPLEMENTED\n\n");
    return 0;
}

// TODO: Port Me
G4double G4ParticleHPVector_CUDA::Get50percentBorder() {
    printf("Get 50 NOT YET IMPLEMENTED\n\n");
    return 0;
}

bool G4ParticleHPVector_CUDA::doesTheDataContainNan() {
    G4ParticleHPDataPoint* localTheData = (G4ParticleHPDataPoint*) malloc(nEntries * sizeof(G4ParticleHPDataPoint));
    cudaMemcpy(localTheData, d_theData, nEntries * sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nEntries; i++) {
        if (localTheData[i].energy != localTheData[i].energy || localTheData[i].xSec != localTheData[i].xSec) {
            free(localTheData);
            return true;
        }
    }
    free(localTheData);
    return false;
}

void G4ParticleHPVector_CUDA::Check(G4int i) {
    if (i > nEntries) {
        // throw G4HadronicException(__FILE__, __LINE__, "Skipped some index numbers in G4ParticleHPVector");
    }
    if (i == nPoints) {
        nPoints = static_cast<G4int>(1.2 * nPoints);

        G4ParticleHPDataPoint* d_newTheData;
        cudaMalloc(&d_newTheData, nPoints * sizeof(G4ParticleHPDataPoint));

        int nBlocks = GetNumBlocks(nEntries);
        CopyDataPointsToBuffer_CUDA<<<nBlocks,THREADS_PER_BLOCK>>> (d_theData, d_newTheData, nEntries);
        
        cudaFree(d_theData);
        d_theData = d_newTheData;
    }
    
    if (i == nEntries) {
        nEntries = i + 1;
    }
}

// Note: Geant4 doesn't ever assign private variable theBlocked,
// which means their IsBlocked function always returns false
G4bool G4ParticleHPVector_CUDA::IsBlocked(G4double aX) {
    return false;
}

// G4ParticleHPVector_CUDA:: & operatorPlus (G4ParticleHPVector & left, G4ParticleHPVector & right) { }
