#include <cuda.h>
#include <cuda_runtime.h>
#include "G4ParticleHPVector_CUDA.hh"

/***********************************************
*   CUDA functions
***********************************************/
__global__ void firstIndexGreaterThan_CUDA(G4ParticleHPDataPoint * theDataArg, G4double e, int* resultIndex, int nEntries) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nEntries && idx < *(resultIndex) && theDataArg[idx].energy > e) {
		atomicMin(resultIndex, idx);
	}
}

// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
__device__ double atomicAdd_CUDA(double* address, double val) { 
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

__global__ void setValueToIntMax_CUDA(int *min_idx) {
    *min_idx = INT_MAX;
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

// TODO: Port Me (requires 1st element predicate alg.)
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

// TODO: Port Me (requires interpolation)
G4double G4ParticleHPVector_CUDA::GetMeanX() {
    return 0;
}

/******************************************
* Setters from .hh that use CUDA
******************************************/
void G4ParticleHPVector_CUDA::SetData(G4int i, G4double x, G4double y) {
    // printf("\nCUDA - SetData (nEntries: %d)", nEntries);
    Check(i);
    G4ParticleHPDataPoint point;
    point.energy = x;
    point.xSec = y;
    cudaMemcpy(&theData[i], &point, sizeof(G4ParticleHPDataPoint), cudaMemcpyHostToDevice);
    
}

void G4ParticleHPVector_CUDA::SetX(G4int i, G4double e) {
    Check(i);
    cudaMemcpy(&theData[i].energy, &e, sizeof(G4double), cudaMemcpyHostToDevice);
}

void G4ParticleHPVector_CUDA::SetEnergy(G4int i, G4double e) {
    Check(i);
    cudaMemcpy(&theData[i].energy, &e, sizeof(G4double), cudaMemcpyHostToDevice);
}

void G4ParticleHPVector_CUDA::SetY(G4int i, G4double x) {
    Check(i);
    cudaMemcpy(&theData[i].xSec, &x, sizeof(G4double), cudaMemcpyHostToDevice);
}

void G4ParticleHPVector_CUDA::SetXsec(G4int i, G4double x) {
    Check(i);
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
    printf("\nCUDA - CleanUp (nEntries: %d", nEntries);
    nEntries = 0;
    theManager.CleanUp();
    maxValue = -DBL_MAX;
    if (theIntegral) {
        cudaFree(theIntegral);
        theIntegral = NULL;
    }
}

__global__ void indexOfLastIntegral_CUDA(G4double * theIntegralArg, int rand, int * resultIndex, int nEntries) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nEntries) {
        return;
    }

    if (i > *resultIndex && theIntegralArg[i]/theIntegralArg[nEntries-1] < rand) {
        atomicMax(resultIndex, i);
    }
}
__global__ void sampleLinGetValues(G4ParticleHPDataPoint * theDataArg, G4double * theIntegralArg, G4double * d_vals, G4int i) {
    // d_vals = [x1,x2,y1,y2]
    switch(threadIdx.x) {
        case 0: 
            d_vals[0] = theIntegralArg[i-1];
            break;
        case 1:
            d_vals[1] = theIntegralArg[i];
            break;
        case 2:
            d_vals[2] = theDataArg[i-1].energy;
            break;
        case 3:
            d_vals[3] = theDataArg[i].energy;
            break;
        default:
            printf("\nError -- invalid thread id in sampleLinGetValues(), returning");
    }
}

G4double G4ParticleHPVector_CUDA::SampleLin() {
    printf("\nCUDA - SampleLin (nEntries: %d", nEntries);
    G4double result;
    if (!theIntegral) {
        IntegrateAndNormalise();
    }

    if (GetVectorLength() == 1) {
        cudaMemcpy(&result, &theData[0].energy, sizeof(G4double), cudaMemcpyDeviceToHost);
    }
    else {
        G4double randNum = (G4double)rand() / (G4double)RAND_MAX; // TODO: change to G4UniformRand        

        int *resultIndex;
        int block_size = 64;// testing showed block_size of 64 gave best times. -- may be different for different GPU's
        int n_blocks = nEntries/block_size + ((nEntries%block_size) == 0 ? 0:1);
        cudaMalloc(&resultIndex, sizeof(int));
        setValueToIntMax_CUDA <<<1,1>>> (resultIndex); // set the Result Index to max value so min will always set it to an actual index -- only need one thread to do this
        indexOfLastIntegral_CUDA<<<n_blocks, block_size>>> (theIntegral, randNum, resultIndex, nEntries); // find the first index who's energy is greater than e -- one thread of each index
        
        G4int i = 0;
        cudaMemcpy(&i, resultIndex, sizeof(G4int), cudaMemcpyDeviceToHost); //It is preferable to only do one memcpy
        if (i != GetVectorLength()-1) {
            i++;
        }

        // vals = [x1, x2, y1, y2]
        G4double* d_vals;
        cudaMalloc(&d_vals, 4*sizeof(G4double));
        sampleLinGetValues<<<1, 4>>>(theData, theIntegral, d_vals, i);
        
        G4double vals[4];
        cudaMemcpy(vals, d_vals, 4*sizeof(G4double), cudaMemcpyDeviceToHost);
        
        result = theLin.Lin(randNum, vals[0], vals[1], vals[2], vals[3]);
        
        cudaFree(resultIndex);
        cudaFree(d_vals);
        free(vals);
    }

    return result;
}

// TODO: Port Me (should return theIntegral, but how do we return somethign we don't have ref to?)
G4double * G4ParticleHPVector_CUDA::Debug() {
    printf("\nDEBUG NOT YET IMPLEMENTED");
    return 0;
}

// TODO: test that this gives same results
__global__ void Integrate_CUDA(G4ParticleHPDataPoint * theDataArg, G4double * sum, G4InterpolationManager theManagerArg) {
    G4int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0) {
        return;
    }

    if (abs((theDataArg[i].energy - theDataArg[i-1].energy) / theDataArg[i].energy) > 0.0000001) {
        G4double x1 = theDataArg[i-1].energy;
        G4double x2 = theDataArg[i].energy;
        G4double y1 = theDataArg[i-1].xSec;
        G4double y2 = theDataArg[i].xSec;

        double toAdd = 0;
        G4InterpolationScheme aScheme = theManagerArg.GetScheme(i);
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
            atomicAdd_CUDA(sum, toAdd);
        }
    }
}
void G4ParticleHPVector_CUDA::Integrate() {
    printf("\nCUDA - Integrate (nEntries: %d", nEntries);
    if (nEntries == 1) {
        totalIntegral = 0;
        return;
    }
    
    G4double *sum;
    cudaMalloc(&sum, sizeof(G4double));
    Integrate_CUDA<<<1, nEntries>>>(theData, sum, theManager);
    totalIntegral = *sum;
}

// TODO: Port Me
void G4ParticleHPVector_CUDA::IntegrateAndNormalise() {

}

__global__ void Times_CUDA(G4double factor, G4ParticleHPDataPoint* theDataArg, G4double* theIntegralArg, G4int nEntriesArg) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= nEntriesArg) {
        return;
    }
    theDataArg[tid].xSec = theDataArg[tid].xSec*factor;
    theIntegralArg[tid] = theIntegralArg[tid]*factor;
}
void G4ParticleHPVector_CUDA::Times(G4double factor) {
	printf("\nCUDA - Times (nEntries: %d", nEntries);
    int block_size = 64;
	int n_blocks = nEntries / block_size + (((nEntries % block_size) == 0) ? 0 : 1); 
    Times_CUDA<<<n_blocks, block_size>>> (factor, theData, theIntegral, nEntries);
}

/******************************************
* Functions from .cc
******************************************/
G4double G4ParticleHPVector_CUDA::GetXsec(G4double e) {
    // printf("\nCUDA - GetXsec (nEntries: %d", nEntries);
    //Initialize and malloc in constructor to make this more efficient
	int *resultIndex;
    int block_size = 64;// testing showed block_size of 64 gave best times. -- may be different for different GPU's
	int n_blocks = nEntries/block_size + ((nEntries%block_size) == 0 ? 0:1);
	cudaMalloc(&resultIndex, sizeof(int));
	setValueToIntMax_CUDA <<<1,1>>> (resultIndex); // set the Result Index to max value so min will always set it to an actual index -- only need one thread to do this
	
	    
    firstIndexGreaterThan_CUDA<<<n_blocks, block_size>>> (theData, e, resultIndex, nEntries); // find the first index who's energy is greater than e -- one thread of each index
    
	//getting the index and getting the xSec
    G4int i = 0;
    cudaMemcpy(&i, resultIndex, sizeof(G4int), cudaMemcpyDeviceToHost); //It is preferable to only do one memcpy
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
	cudaFree(resultIndex);// free resultIndex to avoid memory leaks -- if resultIndex is moved to init. more this to the deconstructor
    return y;
}

void G4ParticleHPVector_CUDA::Dump() {
    printf("\nCUDA - Dump (nEntries: %d", nEntries);
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

__global__ void copyDataPointsFromBufferToBuffer_CUDA(G4ParticleHPDataPoint * fromBuffer, G4ParticleHPDataPoint * toBuffer, G4int nEntries) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nEntries) {
        toBuffer[i].energy = fromBuffer[i].energy;
        toBuffer[i].xSec = fromBuffer[i].xSec;
    }
}

// TODO: Make me parallel (works, but is serial so memcpy's too much)
void G4ParticleHPVector_CUDA::ThinOut(G4double precision) {
    printf("\nCUDA - ThinOut (nEntries: %d", nEntries);
    if (GetVectorLength() == 0) {
      return;
    }

    G4ParticleHPDataPoint *localTheData = (G4ParticleHPDataPoint*)malloc(nEntries*sizeof(G4ParticleHPDataPoint));
    cudaMemcpy(localTheData, theData, nEntries*sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    G4ParticleHPDataPoint *localBuffer = (G4ParticleHPDataPoint*)malloc(nPoints*sizeof(G4ParticleHPDataPoint));
    // G4ParticleHPDataPoint * d_aBuff;
    // cudaMalloc(d_aBuff, nPoints * sizeof(G4ParticleHPDataPoint));
    
    G4double x, x1, x2, y, y1, y2;
    G4int count = 0;
    G4int current = 2;
    G4int start = 1;

    // First element always goes and is never tested.
    localBuffer[0] = localTheData[0];
    // copyDataPointFromBufferToBuffer_CUDA<<<1,1>>> (theData, aBuff, i, i, nEntries, nEntries);

    // Find the rest
    while(current < GetVectorLength()) {
        x1 = localBuffer[count].GetX();
        y1 = localBuffer[count].GetY();
        x2 = localTheData[current].GetX();
        y2 = localTheData[current].GetY();
        for(G4int j=start; j<current; j++) {
            x = localTheData[j].GetX();
            if(x1-x2 == 0) {
                y = (y2+y1)/2.;
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

    // The last one also always goes, and is never tested.
    count++;
    localBuffer[count] = localTheData[GetVectorLength() - 1];
    nEntries = count + 1;
    printf("setting nEntries to: %d\n", nEntries);

    cudaFree(theData);
    cudaMemcpy(theData, localBuffer, count+1 * sizeof(G4ParticleHPDataPoint), cudaMemcpyHostToDevice);

    free(localTheData);
    free(localBuffer);
    // copyDataPointFromBufferToBuffer_CUDA<<<1,1>>> (localTheData, localBuffer, GetVectorLength()-1, count, nEntries, nEntries);
    // cudaFree(localTheData);
    // localTheData = localBuffer;
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

void G4ParticleHPVector_CUDA::Check(G4int i) {
    if(i > nEntries) {
        // throw G4HadronicException(__FILE__, __LINE__, "Skipped some index numbers in G4ParticleHPVector");
    }
    if (i == nPoints) {
        nPoints = static_cast<G4int>(1.2 * nPoints);
        G4ParticleHPDataPoint* newTheData;
        cudaMalloc(&newTheData, nPoints*sizeof(G4ParticleHPDataPoint));
        int blockSize = 64;
        int threadsPerBlock = nEntries / blockSize + ((nEntries % blockSize == 0) ? 0 : 1);
        copyDataPointsFromBufferToBuffer_CUDA<<<blockSize,threadsPerBlock>>> (theData, newTheData, nEntries);
        cudaFree(theData);
        theData = newTheData;
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
