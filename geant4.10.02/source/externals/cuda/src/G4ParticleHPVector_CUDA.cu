// #include <time.h>
// #include <sys/time.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include "G4ParticleHPVector_CUDA.hh"
// #include <thrust/device_vector.h>
// #include <stdio.h>
// #include <iostream>
// #include <math.h>

// __global__ void SetArrayTo(int *resArray, int numQueries, int setValue)
// {
//   int idx = blockDim.x*blockIdx.x + threadIdx.x;
//   if (idx < numQueries) {
//     resArray[idx] = setValue;
//   }
// }

// __global__ void findMinArray2(G4ParticleHPDataPoint *theData_d, G4double *queryArray_d, int *resArray_d, int numThreads, int numQueries, int nEntries)
// {
//   int idx = blockDim.x*blockIdx.x + threadIdx.x;
//   for (int i = 0; i < numQueries; i++) {
//     G4double queryEnergy = queryArray_d[i];
    
//     // search through data points in thread's range 
//     for (int j = idx; j <= nEntries; j+= numThreads) {
//       if (theData_d[j].energy >  queryEnergy) {
//         atomicMin(&resArray_d[i], j);
//         break;
//       }
//     }
  
//   }

//   // slower, 13s for highest test (and seg fault too)
//   // int start = blockIdx.x * queriesPerBlock;
//   // int i = start;
//   // do {
//   //   G4double queryEnergy = queryArray_d[i];

//   //   for (int j = idx; j < nEntries; j += numThreads) {
//   //     if (theData_d[j].energy >  queryEnergy) {
//   //       atomicMin(&resArray_d[i], j);
//   //       break;
//   //     }
//   //   }
//   //   i = ++i % numQueries;
//   // } while (i != start);
// }

// /***********************************************
// *   Device Methods
// ***********************************************/
// void G4ParticleHPVector_CUDA::SetInterpolationManager(G4InterpolationManager & aManager) {
//   theManager = aManager;
// }
// void G4ParticleHPVector_CUDA::SetInterpolationManager(const G4InterpolationManager & aManager) {
//   theManager = aManager;
// }

// double getWallTime() {
//   struct timeval time;
//   gettimeofday(&time, NULL);
//   return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
// }

// /***********************************************
// *   Host Methods
// ***********************************************/
// void G4ParticleHPVector_CUDA::GetXsecList(G4double* energiesIn_xSecsOut, G4int numQueries, G4ParticleHPDataPoint* theData, G4int nEntries) {  
//   if (nEntries == 0) {
//     for (int i = 0; i < numQueries; i++) {
//       energiesIn_xSecsOut[i] = 0.0;
//     }
//     return;
//   }

//   G4ParticleHPDataPoint * d_theData;
//   G4double              * d_energiesIn_xSecsOut;
//   G4int                 * d_minIndices;
  
//   cudaMalloc((void**)&d_theData,             sizeof(G4ParticleHPDataPoint)            * nEntries);
//   cudaMalloc((void**)&d_energiesIn_xSecsOut, sizeof(G4double)            * numQueries);
//   cudaMalloc((void**)&d_minIndices,          sizeof(G4int)               * numQueries);
//   G4int *minIndices = (G4int*)malloc(numQueries * sizeof(G4int));

//   cudaMemcpy(d_theData, theData, sizeof(G4ParticleHPDataPoint) * nEntries, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_energiesIn_xSecsOut, energiesIn_xSecsOut, sizeof(G4double) * numQueries, cudaMemcpyHostToDevice);
  
//   int queryBlocks = numQueries/THREADS_PER_BLOCK + (numQueries % THREADS_PER_BLOCK == 0 ? 0:1);
//   int dataChunk = 1;
//   int threadNum = nEntries/dataChunk;
//   int arrayBlocks = threadNum/THREADS_PER_BLOCK + (threadNum % THREADS_PER_BLOCK == 0 ? 0:1);
//   int queriesPerBlock = numQueries / arrayBlocks;
  
//   double a = getWallTime();
//   SetArrayTo <<< queryBlocks, THREADS_PER_BLOCK >>>(d_minIndices, numQueries, nEntries-1);
//   findMinArray2 <<< arrayBlocks, THREADS_PER_BLOCK >>> (d_theData, d_energiesIn_xSecsOut, d_minIndices, threadNum, numQueries, nEntries);
//   cudaDeviceSynchronize();
//   printf("Time (nEntries = %d, numQueries = %d): %f\n", nEntries, numQueries, getWallTime() - a);
  
//   cudaMemcpy(minIndices, d_minIndices, numQueries * sizeof(G4int), cudaMemcpyDeviceToHost);

//   for (int i = 0; i < numQueries; i++) {
//     int minIndex = minIndices[i];
   
//     G4int low = minIndex - 1;
//     G4int high = minIndex;
//     G4double e = energiesIn_xSecsOut[i];
    
//     if (minIndex == 0)
//     {
//       low = 0;
//       high = 1;
//     }
//     else if (minIndex == nEntries)
//     {
//       low = nEntries - 2;
//       high = nEntries - 1;
//     }

//     if (e < theData[nEntries-1].GetX())
//     {
//       if (theData[high].GetX() != 0 
//         && (std::abs((theData[high].GetX() - theData[low].GetX()) / theData[high].GetX()) < 0.000001))
//       {
//         energiesIn_xSecsOut[i] = theData[low].GetY();
//       }
//       else
//       {
//         energiesIn_xSecsOut[i] = 
//           theInt.Interpolate(theManager.GetScheme(high), e, 
//                              theData[low].GetX(), theData[high].GetX(),
//                              theData[low].GetY(), theData[high].GetY());
//       }
//     }
//     else
//     {
//       energiesIn_xSecsOut[i] = theData[nEntries-1].GetY();
//     }
//   }

//   cudaFree(d_theData);
//   cudaFree(d_energiesIn_xSecsOut);
//   cudaFree(d_minIndices);
//   free(minIndices);
// }


#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "G4ParticleHPVector_CUDA.hh"

/***********************************************
*   Device Methods
***********************************************/
__global__ void GetMinIndices_CUDA(G4ParticleHPDataPoint *d_theData, int nEntries, 
                                   double *d_energiesIn_xSecsOut, int numQueries, int *d_minIndices) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int stepSize = (int)sqrt((float)nEntries);

  if (idx < numQueries) {
    int i = 0;
    double e = d_energiesIn_xSecsOut[idx];
    
    for (i = 0; i < nEntries; i += stepSize) {
      if (d_theData[i].energy >= e) {
        break;
      }
    }
    
    i = (i - (stepSize - 1) >= 0) ? i - (stepSize - 1) : 0; 
    for (; i < nEntries; i++) {
      if (d_theData[i].energy >= e) {
        break;
      }
    }

    d_minIndices[idx] = i;
  }
}

void G4ParticleHPVector_CUDA::SetInterpolationManager(G4InterpolationManager & aManager) {
  theManager = aManager;
}
void G4ParticleHPVector_CUDA::SetInterpolationManager(const G4InterpolationManager & aManager) {
  theManager = aManager;
}

double getWallTime() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

/***********************************************
*   Host Methods
***********************************************/
void G4ParticleHPVector_CUDA::GetXsecList(G4double* energiesIn_xSecsOut, G4int numQueries, G4ParticleHPDataPoint* theData, G4int nEntries) {  
  if (nEntries == 0) {
    for (int i = 0; i < numQueries; i++) {
      energiesIn_xSecsOut[i] = 0.0;
    }
    return;
  }

  G4ParticleHPDataPoint * d_theData;
  G4double              * d_energiesIn_xSecsOut;
  G4int                 * d_minIndices;
  
  cudaMalloc((void**)&d_theData,             sizeof(G4ParticleHPDataPoint)            * nEntries);
  cudaMalloc((void**)&d_energiesIn_xSecsOut, sizeof(G4double)            * numQueries);
  cudaMalloc((void**)&d_minIndices,          sizeof(G4int)               * numQueries);
  G4int *minIndices = (G4int*)malloc(numQueries * sizeof(G4int));

  cudaMemcpy(d_theData, theData, sizeof(G4ParticleHPDataPoint) * nEntries, cudaMemcpyHostToDevice);
  cudaMemcpy(d_energiesIn_xSecsOut, energiesIn_xSecsOut, sizeof(G4double) * numQueries, cudaMemcpyHostToDevice);
  
  // need to add 1 block if doesn't divide evenly (e.g 32 T_P_B, 36 numQueries we need 1+1=2 blocks to get those last 4 queries)
  int numBlocksSingleElement = numQueries/THREADS_PER_BLOCK + ((numQueries % THREADS_PER_BLOCK == 0) ? 0 : 1);

  GetMinIndices_CUDA <<<numBlocksSingleElement, THREADS_PER_BLOCK>>>
    (d_theData, nEntries, d_energiesIn_xSecsOut, numQueries, d_minIndices);

  cudaMemcpy(minIndices, d_minIndices, sizeof(G4int) * numQueries, cudaMemcpyDeviceToHost);

  for (int i = 0; i < numQueries; i++) {
    int minIndex = minIndices[i];
   
    G4int low = minIndex - 1;
    G4int high = minIndex;
    G4double e = energiesIn_xSecsOut[i];
    
    if (minIndex == 0)
    {
      low = 0;
      high = 1;
    }
    else if (minIndex == nEntries)
    {
      low = nEntries - 2;
      high = nEntries - 1;
    }

    if (e < theData[nEntries-1].GetX())
    {
      if (theData[high].GetX() != 0 
        &&(std::abs((theData[high].GetX() - theData[low].GetX()) / theData[high].GetX()) < 0.000001))
      {
        energiesIn_xSecsOut[i] = theData[low].GetY();
      }
      else
      {
        energiesIn_xSecsOut[i] = theInt.Interpolate(theManager.GetScheme(high), e, 
                               theData[low].GetX(), theData[high].GetX(),
                               theData[low].GetY(), theData[high].GetY());
      }
    }
    else
    {
      energiesIn_xSecsOut[i] = theData[nEntries-1].GetY();
    }
  }

  cudaFree(d_theData);
  cudaFree(d_energiesIn_xSecsOut);
  cudaFree(d_minIndices);
  free(minIndices);
}