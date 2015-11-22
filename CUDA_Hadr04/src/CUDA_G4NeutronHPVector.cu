#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include "CUDA_G4NeutronHPVector.h"

// CUDA kernel
__global__
void sumArrays(int* arr1, int* arr2, int* res, int n)
{
  int tid = blockIdx.x;
  if (tid < n) 
      res[tid] = arr1[tid] + arr2[tid];
}

void CUDA_sumArrays(int* arr1, int* arr2, int* res, int n) {
    int *gpu_arr1, *gpu_arr2, *gpu_res;

    cudaMalloc((void**)&gpu_arr1, n*sizeof(int));
    cudaMalloc((void**)&gpu_arr2, n*sizeof(int));
    cudaMalloc((void**)&gpu_res, n*sizeof(int));
    
    cudaMemcpy(gpu_arr1, arr1, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_arr2, arr2, n*sizeof(int), cudaMemcpyHostToDevice);

    sumArrays<<<n,1>>>(gpu_arr1, gpu_arr2, gpu_res, n);

    cudaMemcpy(res, gpu_res, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpu_arr1);
    cudaFree(gpu_arr2);
    cudaFree(gpu_res);
}
