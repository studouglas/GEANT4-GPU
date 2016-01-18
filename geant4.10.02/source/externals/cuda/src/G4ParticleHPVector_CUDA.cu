#include <cuda.h>
#include <cuda_runtime.h>
#include "G4ParticleHPVector_CUDA.h"

// CUDA kernel
__global__
void sumArrays(int* arr1, int* arr2, int* res, int n)
{
    int tid = blockIdx.x;
    if (tid < n) {
        res[tid] = arr1[tid] + arr2[tid];
    }
}

// void CUDA_sumArrays(int* arr1, int* arr2, int* res, int n) {
//     int *gpu_arr1, *gpu_arr2, *gpu_res;

//     cudaMalloc((void**)&gpu_arr1, n*sizeof(int));
//     cudaMalloc((void**)&gpu_arr2, n*sizeof(int));
//     cudaMalloc((void**)&gpu_res, n*sizeof(int));
    
//     cudaMemcpy(gpu_arr1, arr1, n*sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(gpu_arr2, arr2, n*sizeof(int), cudaMemcpyHostToDevice);

//     sumArrays<<<n,1>>>(gpu_arr1, gpu_arr2, gpu_res, n);

//     cudaMemcpy(res, gpu_res, n*sizeof(int), cudaMemcpyDeviceToHost);

//     cudaFree(gpu_arr1);
//     cudaFree(gpu_arr2);
//     cudaFree(gpu_res);
// }

void G4ParticleHPVector_CUDA::SetNEntries(int * nEntriesPointer) {
    nEntries = nEntriesPointer;
}
void G4ParticleHPVector_CUDA::SetNPoints(int * nPointsPointer) {
    nPoints = nPointsPointer;
}

G4ParticleHPVector_CUDA::G4ParticleHPVector_CUDA() {
    cudaMalloc((void**)&theData, (20) * sizeof(G4ParticleHPDataPoint_CUDA));
}

G4ParticleHPVector_CUDA::G4ParticleHPVector_CUDA(int n) {
    int nPoints = std::max(20, n);
    cudaMalloc((void**)&theData, (nPoints) * sizeof(G4ParticleHPDataPoint_CUDA));
}

void G4ParticleHPVector_CUDA::Times(double factor) {

}

double G4ParticleHPVector_CUDA::GetXsec(double e) {
    // if (nEntries == 0) {
        return 0;
    // }
    
    // TODO: FIGURE OUT THIS FUNCTION
    //int min = theHash.GetMinIndex(e);

    // int i;
    // for (i = min; i < nEntries; i++)
    // {
    //     if (theData[i].x >= e) {
    //         break;
    //     }
    // }
    
    // int low = i - 1;
    // int high = i;
    // if (i == 0)
    // {
    //     low = 0;
    //     high = 1;
    // }
    // else if (i == nEntries)
    // {
    //     low = nEntries-2;
    //     high = nEntries-1;
    // }
    
    // double y;
    // if (e < theData[nEntries-1].x) 
    // {
    //     if (theData[high].x !=0 
    //         && (std::abs((theData[high].x - theData[low].x) / theData[high].x) < 0.000001))
    //     {
    //         y = theData[low].y;
    //     }
    //     else
    //     {
    //         // TODO: FIGURE OUT WHAT TO DO HERE
    //         //y = theInt.Interpolate(theManager.GetScheme(high), e, theData[low].x, theData[high].x, theData[low].y, theData[high].y;
    //         return -1;
    //     }
    // }
    // else
    // {
    //     y = theData[nEntries-1].y;
    // }
    // return y;
}





