#include <cuda.h>
#include <cuda_runtime.h>
#include "G4ParticleHPVector_CUDA.hh"

/***********************************************
*   Device Methods
***********************************************/
__global__ void SetArrayTo(int *array, int length, int setValue)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < length) {
    array[idx] = setValue;
  }
}

__global__ void GetXSecFirstIndexArray_CUDA(G4ParticleHPDataPoint *d_theData, G4double *d_queryList, int *d_resArray, int numThreads, int querySize, int nEntries)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = 0; i < querySize; i++){
    G4double queryEnergy = d_queryList[i];
    
    for (int j = idx; j <= nEntries; j += numThreads) {
      if (d_theData[j].energy >  queryEnergy) {
        atomicMin(&d_resArray[i], j);
      }
    }
  }
}

__global__ void GetYForXSecArray_CUDA(G4ParticleHPDataPoint *theData, int nEntries,  int *indexArray, GetXsecResultStruct * d_resArray, G4int querySize){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < querySize) {
    G4int low = indexArray[idx] -1;
    G4int high = indexArray[idx];
    
    if (indexArray[idx] == 0) {
      low = 0;
      high = 1;
    } else if (indexArray[idx] == nEntries) {
      d_resArray[idx].y = theData[nEntries-1].xSec;
    }
    
    if ((theData[high].energy != 0) && (abs((theData[high].energy - theData[low].energy) / theData[high].energy) < 0.000001)) {
      d_resArray[idx].y = theData[low].xSec;
    } else {
      d_resArray[idx].y = -1;
      d_resArray[idx].pointLow.energy = theData[low].energy;
      d_resArray[idx].pointLow.xSec = theData[low].xSec;
      d_resArray[idx].pointHigh.energy = theData[high].energy;
      d_resArray[idx].pointHigh.xSec = theData[high].xSec;
      d_resArray[idx].indexHigh = high;
    }
  }
}

/***********************************************
*   Host Methods
***********************************************/
void G4ParticleHPVector_CUDA::GetXsecList(G4double* energiesIn_xSecsOut, G4int numQueries, G4ParticleHPDataPoint* theData, G4int nEntries) {  
  G4ParticleHPDataPoint * d_theData;
  G4double              * d_energiesIn_xSecsOut;
  G4int                 * d_minIndices;
  GetXsecResultStruct   * d_resArray;
  GetXsecResultStruct   * h_resArray;

  cudaMalloc((void**)&d_theData,             sizeof(G4double)            * nEntries);
  cudaMalloc((void**)&d_energiesIn_xSecsOut, sizeof(G4double)            * numQueries);
  cudaMalloc((void**)&d_minIndices,          sizeof(G4int)               * numQueries);
  cudaMalloc((void**)&d_resArray,            sizeof(GetXsecResultStruct) * numQueries);
  cudaMallocHost(&h_resArray,                sizeof(GetXsecResultStruct) * numQueries);

  cudaMemcpy(d_theData,             theData,             sizeof(G4ParticleHPDataPoint) * nEntries,   cudaMemcpyHostToDevice);
  cudaMemcpy(d_energiesIn_xSecsOut, energiesIn_xSecsOut, sizeof(G4double)              * numQueries, cudaMemcpyHostToDevice);

  // need to add 1 block if doesn't divide evenly (e.g 32 T_P_B, 36 numQueries we need 1+1=2 blocks to get those last 4 queries)
  int numBlocksSingleElement = numQueries/THREADS_PER_BLOCK + ((numQueries % THREADS_PER_BLOCK == 0) ? 0 : 1);
  
  // each thread will work on multiple elements
  int elementsPerThread = 2;
  int totalNumThreads = nEntries / elementsPerThread;
  int numBlocksMultipleElements = elementsPerThread / THREADS_PER_BLOCK + ((elementsPerThread % THREADS_PER_BLOCK == 0) ? 0 : 1);
  
  // initialize each index in array to last index of theData
  SetArrayTo <<<numBlocksSingleElement, THREADS_PER_BLOCK>>> 
    (d_minIndices, numQueries, nEntries - 1);

  // populate minIndices with the index of the first data point in theData with minimum energy
  GetXSecFirstIndexArray_CUDA <<<numBlocksMultipleElements, THREADS_PER_BLOCK>>>
    (d_theData, d_energiesIn_xSecsOut, d_minIndices, totalNumThreads, numQueries, nEntries);
  
  // fill resArray with struct containing either result if computed directly, or data points needed for interpolation
  GetYForXSecArray_CUDA <<<numBlocksSingleElement, THREADS_PER_BLOCK>>>
    (d_theData, nEntries,  d_minIndices, d_resArray, numQueries);
  
  cudaMemcpy(h_resArray, d_resArray, sizeof(GetXsecResultStruct)*numQueries, cudaMemcpyDeviceToHost);
  
  // interpolate the values (if needed) on CPU (for now)
  for (int i = 0; i < numQueries; i++) {
      GetXsecResultStruct res = h_resArray[i];
    if (res.y != -1) {
      energiesIn_xSecsOut[i] = res.y;
    } else {
      G4double y = theInt.Interpolate(theManager.GetScheme(res.indexHigh), energiesIn_xSecsOut[i],
            res.pointLow.energy, res.pointHigh.energy,
            res.pointLow.xSec, res.pointHigh.xSec);
      if (nEntries == 1) {
        energiesIn_xSecsOut[i] = 0.0;
      }
      energiesIn_xSecsOut[i] = y;
    }
  }

  cudaFree(d_theData);
  cudaFree(d_energiesIn_xSecsOut);
  cudaFree(d_minIndices);
  cudaFree(d_resArray);
  cudaFreeHost(h_resArray);
}
