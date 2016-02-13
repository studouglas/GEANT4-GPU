#include <cuda.h>
#include <cuda_runtime.h>
#include "G4ParticleHPVector_CUDA.hh"
#include <time.h>
#include <curand_kernel.h>

/***********************************************
*   CUDA functions
***********************************************/
__global__ void SetValueTo_CUDA(int *addressToSet, int value) {
    *(addressToSet) = value;
}
__global__ void SetValueTo_CUDA(G4double *addressToSet, G4double value) {
    *(addressToSet) = value;
}
__global__ void SetValueTo_CUDA(G4ParticleHPDataPoint *addressToSet, G4double energy, G4double xSec) {
    addressToSet->energy = energy;
    addressToSet->xSec = xSec;
}

__device__ G4double rand_CUDA() {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() + i, 0, 0, &state);
    return curand_uniform_double(&state);
}

__global__ void SetAllNegativeXsecToZero_CUDA(G4ParticleHPDataPoint * theData, int nEntriesArg) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nEntriesArg) {
        if (theData[tid].xSec < 0) {
            theData[tid].xSec = 0;
        }
    }
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

__global__ void CopyTheIntegralToBuffer_CUDA(G4double * fromBuffer, G4double * toBuffer, G4int nEntries) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nEntries) {
        toBuffer[i] = fromBuffer[i];
    }
}

/***********************************************
*   Constructors, Deconstructors
***********************************************/
G4ParticleHPVector_CUDA::G4ParticleHPVector_CUDA()      { 
    PerformInitialization(20);
}

G4ParticleHPVector_CUDA::G4ParticleHPVector_CUDA(G4int n) {
    PerformInitialization(std::max(n,20));
}

void G4ParticleHPVector_CUDA::PerformInitialization(G4int n) {
	nPoints = n;
	cudaMalloc(&d_theData, nPoints*sizeof(G4ParticleHPDataPoint));
	
	cudaMalloc(&d_singleIntResult, sizeof(G4int));
	cudaMallocHost(&h_singleIntResult, sizeof(G4int));
	cudaMalloc(&d_singleDoubleResult, sizeof(G4double));
	cudaMallocHost(&h_singleDoubleResult, sizeof(G4double));
	cudaMalloc(&d_res, sizeof(GetXsecResultStruct));
	cudaMallocHost(&h_res, sizeof(GetXsecResultStruct));
    nEntries = 0;
    Verbose = 0;
    d_theIntegral = 0;
    totalIntegral = -1;
    isFreed = 0;
    maxValue = -DBL_MAX;
    the15percentBorderCash = -DBL_MAX;
    the50percentBorderCash = -DBL_MAX;
    label = -DBL_MAX;
}

G4ParticleHPVector_CUDA::~G4ParticleHPVector_CUDA() {
  	if (d_singleIntResult) {
  		cudaFree(d_singleIntResult);
  		d_singleIntResult = nullptr;
  	}
  	if (h_singleIntResult) {
  		cudaFreeHost(h_singleIntResult);
  		h_singleIntResult = nullptr;
  	}
  	if (d_singleDoubleResult) {
        cudaFree(d_singleDoubleResult);
        d_singleDoubleResult = nullptr;
    }
    if (h_singleDoubleResult) {
		cudaFreeHost(h_singleDoubleResult);
  		h_singleDoubleResult = nullptr;
    }
    if (d_res) {
        cudaFree(d_res);
        d_res = nullptr;
    }
    if (h_res) {
		cudaFreeHost(h_res);
  		h_res = nullptr;
    }
    if (d_theData) {
        cudaFree(d_theData);
        d_theData = nullptr;
    }
    if (d_theIntegral) {
       cudaFree(d_theIntegral);
       d_theIntegral = nullptr;
    }
    isFreed = 1;
}

void G4ParticleHPVector_CUDA::OperatorEquals(G4ParticleHPVector_CUDA * right) {
    G4int i;

    totalIntegral = right->totalIntegral;
    G4double * theIntegral = (G4double*)malloc(sizeof(G4double)*right->nEntries);
    G4double * rightTheIntegral = (G4double*)malloc(nEntries*sizeof(G4double));
    cudaMemcpy(rightTheIntegral, right->d_theIntegral, nEntries*sizeof(G4double), cudaMemcpyDeviceToHost);
    
    for (i = 0; i < right->nEntries; i++) {
        SetPoint(i, right->GetPoint(i));
        if (right->d_theIntegral != 0) {
            theIntegral[i] = rightTheIntegral[i];
        }
    }
    
    if (d_theIntegral) {
        cudaFree(d_theIntegral);
        cudaMalloc(&d_theIntegral, nEntries*sizeof(G4double));
        cudaMemcpy(d_theIntegral, theIntegral, nEntries*sizeof(G4double), cudaMemcpyHostToDevice);
    }
    
    theManager = right->theManager; 
    label = right->label;

    Verbose = right->Verbose;
    the15percentBorderCash = right->the15percentBorderCash;
    the50percentBorderCash = right->the50percentBorderCash;
    
    if (theIntegral) {
        free(theIntegral);
    }
   
    // totalIntegral = right->totalIntegral;
    // nEntries = right->nEntries;
    // nPoints = right->nPoints;

    // int numBlocks = GetNumBlocks(nEntries);

    // if (right->d_theIntegral != 0) {
    //   free(d_theIntegral);
    //   cudaMalloc(&d_theIntegral, nEntries * sizeof(G4double));
    //   CopyTheIntegralToBuffer_CUDA<<<numBlocks, THREADS_PER_BLOCK>>> (right->d_theIntegral, d_theIntegral, nEntries);
    // }
    
    // free(d_theData);
    // cudaMalloc(&d_theData, nPoints * sizeof(G4ParticleHPDataPoint));
    // CopyDataPointsToBuffer_CUDA<<<numBlocks, THREADS_PER_BLOCK>>> (right->d_theData, d_theData, nEntries);
    
    
    // // for(G4int i = 0; i < right->nEntries; i++) {
    // //   SetPoint(i, right.GetPoint(i)); // copy theData
    // //   if (right.theIntegral != 0) {
    // //     theIntegral[i] = right.theIntegral[i];
    // //   }
    // // }
    
    // theManager = right->theManager; 
    // label = right->label;

    // Verbose = right->Verbose;
    // the15percentBorderCash = right->the15percentBorderCash;
    // the50percentBorderCash = right->the50percentBorderCash;
}


/******************************************
* Getters from .hh that use CUDA
******************************************/
G4ParticleHPDataPoint & G4ParticleHPVector_CUDA::GetPoint(G4int i) {
    G4ParticleHPDataPoint point;
    cudaMemcpy(&point, &d_theData[i], sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    G4ParticleHPDataPoint *res  = new G4ParticleHPDataPoint(point.energy, point.xSec);
    return *res;
}

G4double G4ParticleHPVector_CUDA::GetX(G4int i) {
    if (i < 0) {
        i = 0;
    }
    if (i >= GetVectorLength()) {
        i = GetVectorLength() - 1;
    }
    // G4double energy;
    cudaMemcpy(h_singleDoubleResult, &d_theData[i].energy, sizeof(G4double), cudaMemcpyDeviceToHost);
    if (*(h_singleDoubleResult) != *(h_singleDoubleResult)) { printf("\nGetEnergy(%d) = %f, nEntries=%d", i, *h_singleDoubleResult, nEntries); }
    return *(h_singleDoubleResult);
}

G4double G4ParticleHPVector_CUDA::GetY(G4int i) {    
    if (i < 0) {
        i = 0;
    }
    if (i >= GetVectorLength()) {
        i = GetVectorLength() - 1;
    }
    // G4double xSec;
    cudaMemcpy(h_singleDoubleResult, &d_theData[i].xSec, sizeof(G4double), cudaMemcpyDeviceToHost);
    return *(h_singleDoubleResult);
}

G4double G4ParticleHPVector_CUDA::GetY(G4double x) {
    return GetXsec(x);
}

// TODO: Port Me (requires 1st element predicate alg.)
G4double G4ParticleHPVector_CUDA::GetXsec(G4double e, G4int min) {
    printf("\nGETXSEC(e,min) NOT YET IMPLEMENTED");
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
    SetValueTo_CUDA<<<1,1>>> (&d_theData[i], x, y);
    if (x != x || y != y) { printf("\nSetData got passed NAN!, SetData(%d, %0.5e, %0.5e)", i, x, y); }
}

void G4ParticleHPVector_CUDA::SetX(G4int i, G4double e) {
    Check(i);
    SetValueTo_CUDA<<<1,1>>> (&d_theData[i].energy, e);
    if (e != e) { printf("\nSetX(%d) got passed NAN!!!", i); }
}

void G4ParticleHPVector_CUDA::SetEnergy(G4int i, G4double e) {
    SetX(i,e);
}

void G4ParticleHPVector_CUDA::SetY(G4int i, G4double x) {
    Check(i);
    SetValueTo_CUDA<<<1,1>>> (&d_theData[i].xSec, x);
    if (x != x) { printf("\nSety(%d) got passed NAN!!!", i); }
}

void G4ParticleHPVector_CUDA::SetXsec(G4int i, G4double x) {
    SetY(i,x);
}


/******************************************
* Computations from .hh that use CUDA
******************************************/
void G4ParticleHPVector_CUDA::Init(std::istream & aDataFile, G4double ux, G4double uy) {
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
        d_theIntegral = nullptr;
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
    printf("\nCUDA - SampleLin (nEntries: %d)", nEntries);
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
    printf("\nCUDA - Integrate (nEntries: %d)", nEntries);
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

__global__ void TimesTheIntegral_CUDA(G4double * theIntegral, G4int nEntries, G4double factor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nEntries) {
        theIntegral[tid] *= factor;
    }
}
void G4ParticleHPVector_CUDA::IntegrateAndNormalise() {
    G4int i;
    if (d_theIntegral != 0) {
        return;
    }
    cudaMalloc(&d_theIntegral, nEntries * sizeof(G4double));
    
    if (nEntries == 1) {
        G4double one = 1.0;
        SetValueTo_CUDA<<<1,1>>> (&d_theIntegral[0], 1.0);
        return;
    }

    G4double sum = 0;
    G4double x1 = 0;
    G4double x0 = 0;
    G4double localTheIntegral[nEntries];
    localTheIntegral[0] = 0;

    G4ParticleHPDataPoint *localTheData = (G4ParticleHPDataPoint*) malloc(nEntries * sizeof(G4ParticleHPDataPoint));
    cudaMemcpy(localTheData, d_theData, nEntries*sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);

    for (G4int i = 1; i < GetVectorLength(); i++) {
        x1 = localTheData[i].GetX();
        x0 = localTheData[i-1].GetX();
        if (std::abs(x1 - x0) > std::abs(x1 * 0.0000001)) {
            G4InterpolationScheme aScheme = theManager.GetScheme(i);
            G4double y0 = localTheData[i-1].GetY();
            G4double y1 = localTheData[i].GetY();
            G4double integ = theInt.GetBinIntegral(aScheme,x0,x1,y0,y1);
            #if defined WIN32-VC
                if(!_finite(integ)){integ=0;}
            #elif defined __IBMCPP__
                if(isinf(integ)||isnan(integ)){integ=0;}
            #else
                if(std::isinf(integ)||std::isnan(integ)){integ=0;}
            #endif

            sum += integ;
        }
        localTheIntegral[i] = sum;
    }
    G4double total = localTheIntegral[GetVectorLength()-1];
    cudaMemcpy(d_theIntegral, localTheIntegral, nEntries * sizeof(G4double), cudaMemcpyHostToDevice);
    free(localTheData);

    int nBlocks = GetNumBlocks(nEntries);
    TimesTheIntegral_CUDA<<<nBlocks, THREADS_PER_BLOCK>>> (d_theIntegral, nEntries, 1.0/total);
}

__global__ void Times_CUDA(G4double factor, G4ParticleHPDataPoint* theData, G4double* theIntegral, G4int nEntriesArg) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= nEntriesArg) {
        return;
    }
    theData[tid].xSec = theData[tid].xSec * factor;
    if (theIntegral != 0) {
    	theIntegral[tid] = theIntegral[tid] * factor;
    }
}
void G4ParticleHPVector_CUDA::Times(G4double factor) {
    int nBlocks = GetNumBlocks(nEntries);
    Times_CUDA<<<nBlocks, THREADS_PER_BLOCK>>> (factor, d_theData, d_theIntegral, nEntries);
}

/******************************************
* Functions from .cc
******************************************/
__global__ void GetXSecFirstIndex_CUDA(G4ParticleHPDataPoint * theData, G4double e, int * resultIndex, int nEntries) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nEntries && idx < *(resultIndex) && theData[idx].energy >= e) {
        atomicMin(resultIndex, idx);
    }
}
__global__ void GetYForXSec_CUDA(G4ParticleHPDataPoint * theData, G4double e, G4int * singleIntResult, GetXsecResultStruct * resultsStruct, int nEntries) {
	G4int low = *(singleIntResult) - 1;
	G4int high = *(singleIntResult);
	if (*(singleIntResult) == 0) {
		low = 0;
		high = 1;
	} else if (*(singleIntResult) == nEntries) {
		low = nEntries - 2;
		high = nEntries - 1;
	}

    if (e < theData[nEntries - 1].energy) {
        if ((theData[high].energy != 0) && (abs((theData[high].energy - theData[low].energy) / theData[high].energy) < 0.000001)) {
            resultsStruct->y = theData[low].xSec;
        }
        else {
            resultsStruct->y = -1;
            resultsStruct->pointLow.energy = theData[low].energy;
            resultsStruct->pointLow.xSec = theData[low].xSec;
            resultsStruct->pointHigh.energy = theData[high].energy;
            resultsStruct->pointHigh.xSec = theData[high].xSec;
            resultsStruct->indexHigh = high;
        }
    }
    else {
        resultsStruct->y = theData[nEntries - 1].xSec;
    }
}
G4double G4ParticleHPVector_CUDA::GetXsec(G4double e) {
	// printf("\nGetXsec called");
 
   //  // === SLOW SERIAL CODE (works) ==============================================
   //  G4ParticleHPDataPoint* localTheData = (G4ParticleHPDataPoint*)malloc(nEntries*sizeof(G4ParticleHPDataPoint));
   //  cudaMemcpy(localTheData, d_theData, nEntries * sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
   //  if(nEntries == 0) {
   //    return 0;
   //  }
   //  G4int min = 0;
   //  G4int i;
   //  for(i=min ; i<nEntries; i++){
   //    if(localTheData[i].GetX() >= e) {
   //      break;
   //    }
   //  }
   //  G4int low = i-1;
   //  G4int high = i;
   //  if(i==0){
   //    low = 0;
   //    high = 1;
   //  }
   //  else if(i==nEntries){
   //    low = nEntries-2;
   //    high = nEntries-1;
   //  }
   //  G4double y;
   //  if(e < localTheData[nEntries-1].GetX()) {
   //    if (localTheData[high].GetX() !=0 &&( std::abs( (localTheData[high].GetX()-localTheData[low].GetX())/localTheData[high].GetX()) < 0.000001 ) ) {
   //      serialY = localTheData[low].GetY();
   //    }
   //    else {
   //      serialY = theInt.Interpolate(theManager.GetScheme(high), e, 
	  //       localTheData[low].GetX(), localTheData[high].GetX(),
	  //       localTheData[low].GetY(), localTheData[high].GetY());
	//    }
	//  }
	// else {
	//    serialY = localTheData[nEntries-1].GetY();
	// }
	// free(localTheData);
    // return y;
    // === END SLOW SERIAL CODE ==============================================

    // === SLOW GPU CODE ==============================================
    if (nEntries == 0) {
        return 0;
    }

	SetValueTo_CUDA<<<1,1>>> (d_singleIntResult, nEntries);
    int nBlocks = GetNumBlocks(nEntries);
    GetXSecFirstIndex_CUDA<<<nBlocks, THREADS_PER_BLOCK>>> (d_theData, e, d_singleIntResult, nEntries);
    
    // GetXsecResultStruct res;
    GetYForXSec_CUDA<<<1, 1>>> (d_theData, e, d_singleIntResult, d_res, nEntries);
    cudaMemcpy(h_res, d_res, sizeof(GetXsecResultStruct), cudaMemcpyDeviceToHost);
    GetXsecResultStruct res = *(h_res);
    if (res.y != -1) {
    	return res.y;
    } 
    else {
    	G4double y = theInt.Interpolate(theManager.GetScheme(res.indexHigh), e, 
                res.pointLow.energy, res.pointHigh.energy,
                res.pointLow.xSec, res.pointHigh.xSec);
    	if (nEntries == 1) {
    		return 0.0;
    	}
    	return y;
    }
	// === END SLOW GPU CODE ==============================================  
}

void G4ParticleHPVector_CUDA::Dump() {
    printf("\nCUDA - Dump (nEntries: %d)", nEntries);
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
    if (GetVectorLength() == 0) {
        return;
    }
    G4ParticleHPDataPoint *localTheData = (G4ParticleHPDataPoint*)malloc(nEntries * sizeof(G4ParticleHPDataPoint));
    cudaMemcpy(localTheData, d_theData, nEntries * sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
    G4ParticleHPDataPoint * aBuff = new G4ParticleHPDataPoint[nPoints];
    
    G4double x, x1, x2, y, y1, y2;
    G4int count = 0, current = 2, start = 1;

    // first element always goes and is never tested
    aBuff[0] = localTheData[0];

    // find the rest
    while (current < GetVectorLength()) {
        x1 = aBuff[count].GetX();
        y1 = aBuff[count].GetY();
        x2 = localTheData[current].GetX();
        y2 = localTheData[current].GetY();
        for (G4int j = start; j < current; j++) {
            x = localTheData[j].GetX();
            if (x1 - x2 == 0) {
                y = (y2 + y1) / 2.;
            }
            else {
                y = theInt.Lin(x, x1, x2, y1, y2);
            }
            if (std::abs(y - localTheData[j].GetY()) > precision * y) {
                aBuff[++count] = localTheData[current-1]; // for this one, everything was fine
                start = current; // the next candidate
                break;
            }
        }
        current++ ;
    }

    // the last one also always goes, and is never tested
    aBuff[++count] = localTheData[GetVectorLength()-1];
    nEntries = count+1;

    cudaFree(d_theData);
    cudaMalloc(&d_theData, nPoints * sizeof(G4ParticleHPDataPoint));
    cudaMemcpy(d_theData, aBuff, nEntries * sizeof(G4ParticleHPDataPoint), cudaMemcpyHostToDevice);
    free(localTheData);
    delete [] aBuff;
}

// TODO: Port Me
void G4ParticleHPVector_CUDA::Merge(G4InterpolationScheme aScheme, G4double aValue, G4ParticleHPVector_CUDA * active, G4ParticleHPVector_CUDA * passive) {
    printf("MERGE NOT YET IMPLEMENTED\n\n");
}

__device__ int SampleGetFirstIndex_CUDA(G4double * theIntegral, G4double myRand, G4int nEntries) {
    for (int i = 0; i < nEntries; i++) {
        if (theIntegral[i] > myRand) {
            return i;
        }
    }
    return -1;
}
__global__ void SampleGetResult_CUDA(G4ParticleHPDataPoint * theData, G4double * theIntegral, G4int nEntries, G4double * result) {
    G4double myRand;
    G4double value;
    G4double test;
    
    G4int jcounter = 0;
    G4int jcounter_max = 1024;
    do {
        jcounter++;
        if (jcounter > jcounter_max) {
            printf("Loop-counter exceeded the threshold value.\n");
            break;
        }
        myRand = rand_CUDA();
        G4int ibin = SampleGetFirstIndex_CUDA(theIntegral, myRand, nEntries);
        
        if (ibin < 0) {
            printf("TKDB 080807 %f\n", myRand);
        }
    
        // result 
        myRand = rand_CUDA();
        G4double x1, x2;
        if (ibin == 0) {
            x1 = theData[ibin].energy; 
            value = x1; 
            break;
        }
        else {
            x1 = theData[ibin-1].energy;
        }

        x2 = theData[ibin].energy;
        value = myRand * (x2 - x1) + x1;
    
        // EMendoza - Always linear interpolation:
        G4double y1 = theData[ibin-1].xSec;
        G4double y2 = theData[ibin].xSec;
        G4double mval = (y2-y1) / (x2-x1);
        G4double bval = y1 - mval * x1;
        test = (mval * value + bval) / max(theData[ibin-1].xSec, theData[ibin].xSec); 
    } while (rand_CUDA() > test);
    *(result) = value;
}
G4double G4ParticleHPVector_CUDA::Sample() {    
    G4double result;
    
    int nBlocks = GetNumBlocks(nEntries);
    SetAllNegativeXsecToZero_CUDA<<<nBlocks,THREADS_PER_BLOCK>>> (d_theData, nEntries);

    if (GetVectorLength() == 1) {
        cudaMemcpy(&result, &d_theData[0].energy, sizeof(G4double), cudaMemcpyHostToDevice);
    }
    else {
        if (d_theIntegral == 0) { 
            IntegrateAndNormalise(); 
        }
        SampleGetResult_CUDA<<<1, 1>>> (d_theData, d_theIntegral, nEntries, d_singleDoubleResult);
        cudaMemcpy(&result, d_singleDoubleResult, sizeof(G4double), cudaMemcpyDeviceToHost);
    }

    return result;
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
    if (i > nEntries) {
        printf("ERROR\n\n\nskipped some index numbers in Cuda::CHECK\n\n");
        return;
    }
    if (i == nPoints) {
        nPoints = static_cast<G4int>(1.2 * nPoints);

        G4ParticleHPDataPoint * d_newTheData;
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
