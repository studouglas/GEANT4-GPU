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
__global__ void SetValueTo_CUDA(G4ParticleHPDataPoint *addressToSet, G4double energy,
    G4double xSec) {
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

__global__ void CopyDataPointsToBuffer_CUDA(G4ParticleHPDataPoint * fromBuffer,
    G4ParticleHPDataPoint * toBuffer, G4int nEntries) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nEntries) {
        toBuffer[i].energy = fromBuffer[i].energy;
        toBuffer[i].xSec = fromBuffer[i].xSec;
    }
}

__global__ void CopyTheIntegralToBuffer_CUDA(G4double * fromBuffer, G4double * toBuffer,
    G4int nEntries) {
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
    isDataDirtyHost = true;
    nPointsHost = nPoints;
    h_theData = (G4ParticleHPDataPoint*)malloc(nPoints * sizeof(G4ParticleHPDataPoint));
    if (!h_theData) {
        printf("\nMALLOC FAILED IN PERFORM INITIALIZATION");
    }
    cudaMalloc(&d_theData, nPoints * sizeof(G4ParticleHPDataPoint));
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
    if (d_theData) {
        cudaFree(d_theData);
        d_theData = NULL;
    }
    if (h_theData) {
        free(h_theData);
        h_theData = NULL;
    }
    if (d_singleIntResult) {
        cudaFree(d_singleIntResult);
        d_singleIntResult = NULL;
    }
    if (h_singleIntResult) {
        cudaFreeHost(h_singleIntResult);
        h_singleIntResult = NULL;
    }
    if (d_singleDoubleResult) {
        cudaFree(d_singleDoubleResult);
        d_singleDoubleResult = NULL;
    }
    if (h_singleDoubleResult) {
        cudaFreeHost(h_singleDoubleResult);
        h_singleDoubleResult = NULL;
    }
    if (d_res) {
        cudaFree(d_res);
        d_res = NULL;
    }
    if (h_res) {
        cudaFreeHost(h_res);
        h_res = NULL;
    }

    if (d_theIntegral) {
       cudaFree(d_theIntegral);
       d_theIntegral = NULL;
    }
    isFreed = 1;
}

void G4ParticleHPVector_CUDA::OperatorEquals(G4ParticleHPVector_CUDA * right) {
    totalIntegral = right->totalIntegral;
    nEntries = right->nEntries;
    nPoints = right->nPoints;

    int numBlocks = GetNumBlocks(nEntries);

    if (right->d_theIntegral != 0) {
        if (d_theIntegral) {
            free(d_theIntegral);
        }
        cudaMalloc(&d_theIntegral, nEntries * sizeof(G4double));
        CopyTheIntegralToBuffer_CUDA<<<numBlocks, THREADS_PER_BLOCK>>> (right->d_theIntegral, d_theIntegral, nEntries);
    }

    cudaMalloc(&d_theData, nPoints * sizeof(G4ParticleHPDataPoint));
    CopyDataPointsToBuffer_CUDA<<<numBlocks, THREADS_PER_BLOCK>>> (right->d_theData, d_theData, nEntries);

    theManager = right->theManager;
    label = right->label;

    Verbose = right->Verbose;
    the15percentBorderCash = right->the15percentBorderCash;
    the50percentBorderCash = right->the50percentBorderCash;

    isDataDirtyHost = true;
}

void G4ParticleHPVector_CUDA::CopyToCpuIfDirty() {
    if (isDataDirtyHost) {
        if (nPointsHost != nPoints) {
            h_theData = (G4ParticleHPDataPoint*)realloc(h_theData, nPoints * sizeof(G4ParticleHPDataPoint));
            if (!h_theData) { printf("\nMALLOC FAILED IN COPY TO CPU"); }
            nPointsHost = nPoints;
        }

        cudaMemcpy(h_theData, d_theData, nEntries * sizeof(G4ParticleHPDataPoint), cudaMemcpyDeviceToHost);
        isDataDirtyHost = false;
    }
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

    if (!isDataDirtyHost) {
        return h_theData[i].GetX();
    }
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
    
    if (!isDataDirtyHost) {
        return h_theData[i].GetY();
    }
    cudaMemcpy(h_singleDoubleResult, &d_theData[i].xSec, sizeof(G4double), cudaMemcpyDeviceToHost);
    return *(h_singleDoubleResult);
}

G4double G4ParticleHPVector_CUDA::GetY(G4double x) {
    return GetXsec(x);
}

// TODO: Port Me
G4double G4ParticleHPVector_CUDA::GetXsec(G4double e, G4int min) {
    printf("\nGETXSEC(e,min) NOT YET IMPLEMENTED");
    return 0;
}

// TODO: Port Me (requires interpolation)
G4double G4ParticleHPVector_CUDA::GetMeanX() {
    printf("\nGETMEANX NOT YET IMPLEMENTED");
    return 0;
}


/******************************************
* Setters from .hh that use CUDA
******************************************/
void G4ParticleHPVector_CUDA::SetData(G4int i, G4double x, G4double y) {
    Check(i);
    SetValueTo_CUDA<<<1,1>>> (&d_theData[i], x, y);
    if (x != x || y != y) { printf("\nSetData got passed NAN!, SetData(%d, %0.5e, %0.5e)", i, x, y); }
    isDataDirtyHost = true;
}

void G4ParticleHPVector_CUDA::SetX(G4int i, G4double e) {
    Check(i);
    SetValueTo_CUDA<<<1,1>>> (&d_theData[i].energy, e);
    if (e != e) { printf("\nSetX(%d) got passed NAN!!!", i); }
    isDataDirtyHost = true;
}

void G4ParticleHPVector_CUDA::SetY(G4int i, G4double x) {
    Check(i);
    SetValueTo_CUDA<<<1,1>>> (&d_theData[i].xSec, x);
    if (x != x) { printf("\nSety(%d) got passed NAN!!!", i); }
    isDataDirtyHost = true;
}

void G4ParticleHPVector_CUDA::SetEnergy(G4int i, G4double e) {
    SetX(i,e);
}

void G4ParticleHPVector_CUDA::SetXsec(G4int i, G4double x) {
    SetY(i,x);
}


/******************************************
* Computations from .hh that use CUDA
******************************************/
void G4ParticleHPVector_CUDA::Init(std::istream & aDataFile, G4int total, G4double ux, G4double uy) {
    G4double x, y;
    printf("Init!: total: %d\n", total);
    
    // TODO: change to realloc, had some problems when it was realloc before
    //h_theData = (G4ParticleHPDataPoint*)realloc(h_theData, total * sizeof(G4ParticleHPDataPoint));
    free(h_theData);
    h_theData = (G4ParticleHPDataPoint*)malloc(total * sizeof(G4ParticleHPDataPoint));
    if (!h_theData) { printf("MALLOC FAILURE - 296"); }
    
    for (G4int i = 0; i < total; i++) {
        aDataFile >> x >> y;
        x *= ux;
        y *= uy;
        h_theData[i] = G4ParticleHPDataPoint(x,y);
    }
    nPoints = total;
    nEntries = total;
    nPointsHost = total;

    if (d_theData) {
        cudaFree(d_theData);
    }
    cudaMalloc(&d_theData, nPoints * sizeof(G4ParticleHPDataPoint));
    cudaMemcpy(d_theData, h_theData, nEntries * sizeof(G4ParticleHPDataPoint), cudaMemcpyHostToDevice);

    isDataDirtyHost = false;
}

void G4ParticleHPVector_CUDA::Init(std::istream & aDataFile, G4double ux, G4double uy) {
    G4int total;
    aDataFile >> total;

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
    isDataDirtyHost = true;
}

__global__ void SampleLinFindLastIndex_CUDA(G4double * theIntegral, int rand, int * resultIndex,
    int nEntries) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nEntries) {
        return;
    }
    if (i > *(resultIndex) && (theIntegral[i] / theIntegral[nEntries-1]) < rand) {
        atomicMax(resultIndex, i);
    }
}

__global__ void SampleLinGetValues(G4ParticleHPDataPoint * theData, G4double * theIntegral,
    G4double * d_vals, G4int i) {
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
    G4double result;
    if (!d_theIntegral) {
        IntegrateAndNormalise();
    }

    if (GetVectorLength() == 1) {
        cudaMemcpy(&result, &d_theData[0].energy, sizeof(G4double), cudaMemcpyDeviceToHost);
    }
    else {
        G4double randNum = GetUniformRand();

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
    }

    return result;
}

// TODO: Port Me (should return d_theIntegral, but how do we return somethign we don't have ref to?)
G4double * G4ParticleHPVector_CUDA::Debug() {
    printf("\nDEBUG NOT YET IMPLEMENTED");
    return 0;
}

// TODO: test that this gives same results
__global__ void Integrate_CUDA(G4ParticleHPDataPoint * theData, G4double * sum,
    G4InterpolationManager theManager) {
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

        // NOTE: cuda's log function requires compute capability >= 3.0 for double precision
        // make sure you are compiling for 3.0 (nvcc -arch sm_30)
        if (aScheme == LINLIN || aScheme == CLINLIN || aScheme == ULINLIN) {
            toAdd += 0.5 * (y2+y1) * (x2-x1);
        }
        else if (aScheme == LINLOG || aScheme == CLINLOG || aScheme == ULINLOG) {
            G4double a = y1;
            G4double b = (y2 - y1) / (log(x2) - log(x1));
            toAdd += (a-b) * (x2-x1) + b*(x2 * log(x2) - x1 * log(x1));
        }
        else if (aScheme == LOGLIN || aScheme == CLOGLIN || aScheme == ULOGLIN) {
            G4double a = log(y1);
            G4double b = (log(y2) - log(y1)) / (x2-x1);
            toAdd += (G4Exp(a) / b) * (G4Exp(b * x2) - G4Exp(b * x1));
        }
        else if (aScheme == HISTO || aScheme == CHISTO || aScheme == UHISTO) {
            toAdd += y1 * (x2-x1);
        }
        else if (aScheme == LOGLOG || aScheme == CLOGLOG || aScheme == ULOGLOG) {
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

// TODO: Parallelize
void G4ParticleHPVector_CUDA::IntegrateAndNormalise() {
    if (d_theIntegral != 0) {
        return;
    }
    cudaMalloc(&d_theIntegral, nEntries * sizeof(G4double));

    if (nEntries == 1) {
        SetValueTo_CUDA<<<1,1>>> (&d_theIntegral[0], 1.0);
        return;
    }

    G4double sum = 0;
    G4double x1 = 0;
    G4double x0 = 0;
    G4double localTheIntegral[nEntries];
    localTheIntegral[0] = 0;

    CopyToCpuIfDirty();

    for (G4int i = 1; i < GetVectorLength(); i++) {
        x1 = h_theData[i].GetX();
        x0 = h_theData[i-1].GetX();
        if (std::abs(x1 - x0) > std::abs(x1 * 0.0000001)) {
            G4InterpolationScheme aScheme = theManager.GetScheme(i);
            G4double y0 = h_theData[i-1].GetY();
            G4double y1 = h_theData[i].GetY();
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

    int nBlocks = GetNumBlocks(nEntries);
    TimesTheIntegral_CUDA<<<nBlocks, THREADS_PER_BLOCK>>> (d_theIntegral, nEntries, 1.0/total);
}

__global__ void Times_CUDA(G4double factor, G4ParticleHPDataPoint* theData, G4double* theIntegral,
    G4int nEntriesArg) {
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
    isDataDirtyHost = true;
}

/******************************************
* Functions from .cc
******************************************/
__global__ void GetXSecFirstIndex_CUDA(G4ParticleHPDataPoint * theData, G4double e,
        int * resultIndex, int numThreads, int nEntries) {
    int start = (blockDim.x * blockIdx.x + threadIdx.x);
    for (int i = start; i < nEntries; i += numThreads) {
        if (theData[i].energy >= e) {
            atomicMin(resultIndex, i);
            return;
        }
    }
}

__global__ void GetYForXSec_CUDA(G4ParticleHPDataPoint * theData, G4double e,
    G4int * singleIntResult, GetXsecResultStruct * resultsStruct, int nEntries) {
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

__global__ void getXsecBuffer_CUDA(G4ParticleHPDataPoint * theData, int nEntries, G4double * d_queryList, GetXsecResultStruct * d_resArray, G4int querySize){
	int idx = blockDim.x*blockIdx.x + threadIdx.x;	// determine thread ID
	if(idx < querySize){
		G4double e = d_queryList[idx];
		//printf("Looking for %f\n", e);
		int i;
		for(i = 0; i < nEntries; i++){ // find the first DataPoint whose energy is greater
			//printf("energy: %f\n",theData[i].energy);
			if(theData[i].energy >= e){
				//printf("found xSec %f for %f\n", theData[i].energy, e);
				break;
			}
		}
		
		// get the index before 
		G4int low = i - 1;
		G4int high = i;
		if(i == 0){// first dataPoint doesn't have an index before
			low = 0;
			high = 1;
		} 
		else if(i == nEntries){// if no dataPoint was good, just use the last datapoint
			d_resArray[idx].y = theData[nEntries-1].xSec;
			return;
		}
		if ((theData[high].energy != 0) && (abs((theData[high].energy - theData[low].energy) / theData[high].energy) < 0.000001)) {
			d_resArray[idx].y = theData[low].xSec;
		}
		else { // can't interpolate on device -- need cpu to do this
			d_resArray[idx].y = -1;
			d_resArray[idx].pointLow.energy = theData[low].energy;
			d_resArray[idx].pointLow.xSec = theData[low].xSec;
			d_resArray[idx].pointHigh.energy = theData[high].energy;
			d_resArray[idx].pointHigh.xSec = theData[high].xSec;
			d_resArray[idx].indexHigh = high;
		}
	}
}

void G4ParticleHPVector_CUDA::GetXsecBuffer(G4double * queryList, G4int length){	
	GetXsecResultStruct * h_resArray;	// Array of result for host
	GetXsecResultStruct * d_resArray;	// Array for where the results are placed on the GPU
	G4double * d_queryList;				// device copy of the queryList

	// Allocate memory for everything
	cudaMallocHost(&h_resArray, sizeof(GetXsecResultStruct) * length);
	cudaMalloc(&d_resArray, sizeof(GetXsecResultStruct) * length);
	cudaMalloc(&d_queryList, sizeof(G4double) * length);

	// Copy the queryList to the device
	cudaMemcpy(d_queryList, queryList, sizeof(G4double)*length, cudaMemcpyHostToDevice);
	
	// Determine how many blocks we need to allocate 
	int block_size =  32;
	int queryBlocks = length/block_size + (length%block_size == 0 ? 0:1);
	// Get GPU to do its thing
	getXsecBuffer_CUDA <<< queryBlocks, block_size >>> (d_theData, nEntries, d_queryList, d_resArray, length);
	
	// Copy the computed results back to the Host
	cudaMemcpy(h_resArray, d_resArray, sizeof(GetXsecResultStruct)*length, cudaMemcpyDeviceToHost);
	
	// need to interpolate the xSecs using CPU, for now
	for(int i = 0; i < length; i++){
	    GetXsecResultStruct res = h_resArray[i];
		if (res.y != -1) {
			queryList[i] = res.y;
		}
		else {
			G4double y = theInt.Interpolate(theManager.GetScheme(res.indexHigh), queryList[i],
            res.pointLow.energy, res.pointHigh.energy,
            res.pointLow.xSec, res.pointHigh.xSec);
			if (nEntries == 1) {
				queryList[i] = 0.0;
			}
			queryList[i] = y;
		}
	}
	// Free the temporary data to avoid memory leaks
	 cudaFree(d_resArray);
	 cudaFree(d_queryList);
	 cudaFree(h_resArray);
}

G4double G4ParticleHPVector_CUDA::GetXsec(G4double e) {
    if (nEntries == 0) {
        return 0;
    }

    // Note: this was causing some crashing / finishing in 0.01s pre-Mar-3 commit, if it crops up
    // again try copying d_theData to a new local array and using that (every GetXSec call)
    CopyToCpuIfDirty();
      
    G4int min = 0;
    G4int i;
    for (i = min; i < nEntries; i++) {
        if (h_theData[i].GetX() >= e) {
            break;
        }
    }

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
    if (e < h_theData[nEntries-1].GetX()) {
        if (h_theData[high].GetX() != 0
                && (std::abs((h_theData[high].GetX() - h_theData[low].GetX()) / h_theData[high].GetX()) < 0.000001)) {
            y = h_theData[low].GetY();
        }
        else {
            y = theInt.Interpolate(theManager.GetScheme(high), e,
                                   h_theData[low].GetX(), h_theData[high].GetX(),
                                   h_theData[low].GetY(), h_theData[high].GetY());
        }
    }
    else {
        y = h_theData[nEntries-1].GetY();
    }

    return y;

    /* ===== Run GetXSec using CUDA ===========================================
    SetValueTo_CUDA<<<1,1>>> (d_singleIntResult, nEntries);

    // GetXSecFirstIndex = 0.000005s
    int elementsPerThread = 2;
    int nBlocks = GetNumBlocks(nEntries / elementsPerThread);
    int numThreads = nBlocks * THREADS_PER_BLOCK;
    GetXSecFirstIndex_CUDA<<<nBlocks, THREADS_PER_BLOCK>>>
        (d_theData, e, d_singleIntResult, numThreads, nEntries);

    // GetYForXSec = 0.000005s
    GetYForXSec_CUDA<<<1,1>>> (d_theData, e, d_singleIntResult, d_res, nEntries);

    // Performing memcpy (singleIntResult) = 0.00003s
    // Performing memcpy (h_res) = 0.00004s
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
    } ===================================================================== */
}

void G4ParticleHPVector_CUDA::Dump() {
    printf("\nCUDA - Dump (nEntries: %d)", nEntries);
    CopyToCpuIfDirty();

    std::cout << nEntries << std::endl;
    for (G4int i = 0; i < nEntries; i++) {
        printf("%0.5e %0.5e\n", h_theData[i].GetX(), h_theData[i].GetY());
    }
    std::cout << std::endl;
}

// TODO: Parallelize
void G4ParticleHPVector_CUDA::ThinOut(G4double precision) {
    if (GetVectorLength() == 0) {
        return;
    }
    CopyToCpuIfDirty();
    G4ParticleHPDataPoint * aBuff = new G4ParticleHPDataPoint[nPoints];

    G4double x, x1, x2, y, y1, y2;
    G4int count = 0, current = 2, start = 1;

    // first element always goes and is never tested
    aBuff[0] = h_theData[0];

    // find the rest
    while (current < GetVectorLength()) {
        x1 = aBuff[count].GetX();
        y1 = aBuff[count].GetY();
        x2 = h_theData[current].GetX();
        y2 = h_theData[current].GetY();
        for (G4int j = start; j < current; j++) {
            x = h_theData[j].GetX();
            if (x1 - x2 == 0) {
                y = (y2 + y1) / 2.;
            }
            else {
                y = theInt.Lin(x, x1, x2, y1, y2);
            }
            if (std::abs(y - h_theData[j].GetY()) > precision * y) {
                aBuff[++count] = h_theData[current-1]; // for this one, everything was fine
                start = current; // the next candidate
                break;
            }
        }
        current++ ;
    }

    // the last one also always goes, and is never tested
    aBuff[++count] = h_theData[GetVectorLength()-1];
    nEntries = count + 1;

    cudaFree(d_theData);
    cudaMalloc(&d_theData, nPoints * sizeof(G4ParticleHPDataPoint));
    cudaMemcpy(d_theData, aBuff, nEntries * sizeof(G4ParticleHPDataPoint), cudaMemcpyHostToDevice);
    delete [] aBuff;
    isDataDirtyHost = true;
}

// TODO: Port Me
void G4ParticleHPVector_CUDA::Merge(G4InterpolationScheme aScheme, G4double aValue,
    G4ParticleHPVector_CUDA * active, G4ParticleHPVector_CUDA * passive) {
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

__global__ void SampleGetResult_CUDA(G4ParticleHPDataPoint * theData, G4double * theIntegral,
    G4int nEntries, G4double * result) {
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

    isDataDirtyHost = true;
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
        printf("ERROR - skipped some index numbers in Cuda::CHECK\n\n");
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

    // don't set data dirty as we haven't changed anything
}

// Geant4 doesn't ever assign private variable theBlocked,
// which means their IsBlocked function always returns false
G4bool G4ParticleHPVector_CUDA::IsBlocked(G4double aX) {
    return false;
}

G4double G4ParticleHPVector_CUDA::GetUniformRand() {
    return (G4double)rand() / (G4double)RAND_MAX;
}
