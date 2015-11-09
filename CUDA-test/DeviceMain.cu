#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "DeviceMain.h"

// Kernel that executes on the CUDA device
__global__ 
void square_array(float *a, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) 
      a[idx] = a[idx] * a[idx];
}

float squareArray(int N) {
    float *a_h, *a_d;  // Pointer to host & device arrays
  	size_t size = N * sizeof(float);
  	
  	a_h = (float *)malloc(size);        // Allocate array on host
  	cudaMalloc((void **) &a_d, size);   // Allocate array on device
  
  	// Initialize host array and copy it to CUDA device
  	for (int i=0; i<N; i++) {
  		a_h[i] = (float)i;
  	}
  	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  
  	// Do calculation on device:
  	int block_size = 4;
  	int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
  
  	square_array <<< n_blocks, block_size >>> (a_d, N);
  
  	// Retrieve result from device and store it in host array
  	cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

  	// Cleanup
  	free(a_h);
  	cudaFree(a_d);

    return a_h[N-1];
}