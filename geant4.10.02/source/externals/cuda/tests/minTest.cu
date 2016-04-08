#include <thrust/device_vector.h>
#include <stdio.h>
#include "G4NeutronHPDataPoint.cu"
#include <iostream>
#include <math.h>


__global__ void resolveMinIndexArray(G4NeutronHPDataPoint *theData_d, int *resArray_d, G4double *queryArray_d, int querySize){
	int idx = blockDim.x*blockIdx.x + threadIdx.x;	// determine threads ID
	if(idx < querySize){
		//printf("index: %i\n", resArray_d[idx]);
		queryArray_d[idx] = theData_d[resArray_d[idx]].xSec;
	}
}

__global__ void SetArrayTo(int *resArray, int querySize, int setValue)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;	// determine threads ID
	if(idx < querySize){
		resArray[idx] = setValue;
	}
	
}


__global__ void findMinArray2(G4NeutronHPDataPoint *theData_d, G4double *queryArray_d, int *resArray_d, int numThreads, int querySize, int nEntries)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;	// determine threads ID
	for (int i = 0; i < querySize; i++){// foreach query in the query List
		G4double queryEnergy = queryArray_d[i];
		for(int j = idx; j <= nEntries; j+= numThreads){// check threads designated chunk of data
			if(theData_d[j].energy >  queryEnergy){
				atomicMin(&resArray_d[i], j);
			}
		}
	}
}

__global__ void findMinArray(G4NeutronHPDataPoint *theData_d, G4double *queryArray_d, int nEntries, int querySize)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;	// determine threads ID
	G4double queryEnergy = queryArray_d[idx];			// thread gets it's query energy value 
	
	if(idx < querySize){
		// thread finds the xSec for that energy Value
		for(int i = 0; i < nEntries; i++){
			if(theData_d[i].energy >  queryEnergy) {
			queryArray_d[idx] = theData_d[i].xSec; // put the result into the output array
			break;
			}
		}
	}
}

__global__ void findMin2(G4NeutronHPDataPoint *a, G4NeutronHPDataPoint *minArray, G4double value, int nEntries)
{
	int nTotalThreads = nEntries;	// Total number of active threads
	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	G4NeutronHPDataPoint temp;
	if(nTotalThreads > 1)
	{
		//printf("nTotalThreads: %i\n", idx + nTotalThreads);
		//First itteration needs to take data from main array and put it into a smaller working array
		//This array is going to have values overwritten which is why we can't do this to theData_d
		
		int halfPoint = (int)ceil((double)nTotalThreads/2);	// divide by two\

		temp = a[idx + (int)ceil((double)nTotalThreads/2)];
		if(idx < halfPoint){

			if(a[idx].energy < value){
				minArray[idx] = temp;
			}
			else{
				minArray[idx] = a[idx];
			}
		}
		nTotalThreads = (int)ceil((double)nTotalThreads/2);	// divide by two
		__syncthreads(); // make sure minArray is properly set
		
		while(nTotalThreads > 1)
		{
			halfPoint = (int)ceil((double)nTotalThreads/2);	// divide by two\
			// only the first half of the threads will be active.
			if(idx < halfPoint)
			{
				// Get the shared value stored by another thread
				temp = minArray[idx + halfPoint];
									//printf("Total: %i, tempI: %i tempEnergy: %f, idx: %i, a[idx].energy: %f\n", nTotalThreads, idx + (int)ceil((double)nTotalThreads/2) ,temp.energy, idx, minArray[idx].energy);

				if(minArray[idx].energy >= value && temp.energy >= value){	// both energies are good
					if(minArray[idx].energy > temp.energy){					// the energy in temp is smaller.
						minArray[idx] = temp;
					}
				}
				else if(minArray[idx].energy < value && temp.energy >= value) // temp good, minArray bad
				{
					minArray[idx] = temp;
				}

				//else keep minArray as it is
			}
			printf("waiting to sync %i \n", idx);
			__syncthreads();
			nTotalThreads = (int)ceil((double)nTotalThreads/2);	// divide by two.
		}
	}
	printf("Thread Done %i\n", idx);
}
__global__ void findMin(G4NeutronHPDataPoint *a, G4double value, int *min_idx)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if(idx < *min_idx)
	{
		if(a[idx].energy > value)
		{
			atomicMin(min_idx, idx);
		}
	}
} 

__global__ void setMin(int *min_idx)
{
	//printf("blockIdx.x :%d blockDim.x: %d  threadIdx.x %d\n", blockIdx.x, blockDim.x, threadIdx.x);
	*min_idx = INT_MAX;
}

__global__ void timesCUDA(G4NeutronHPDataPoint *a, G4double factor)
{
	int idx = blockDim.x *blockIdx.x + threadIdx.x;
	a[idx].energy = a[idx].energy*factor;
}

int main(int argc, char* argv[])
{
    int N = atoi(argv[1]);
	int events = 1600;
	int querySize = 1000;
	size_t size = N*sizeof(G4NeutronHPDataPoint);
	
	G4double *queryArray_d, *queryArray_h;				// Arrays used for holding query values and results
	int *resArray_d;					// Arrays used to store results
	queryArray_h = (G4double *)malloc(querySize*sizeof(G4double));
	
	cudaMalloc((void **) &queryArray_d, sizeof(G4double)*querySize);
	cudaMalloc((void **) &resArray_d, sizeof(int)*querySize);
	
	G4NeutronHPDataPoint *minArray_d;				// temp array we can use to find min with reduction.
	G4NeutronHPDataPoint *theData_h, *theData_d; // Pointer to host & device arrays
	int *minIdx_d;// index used to store index for first occuring element
	theData_h = (G4NeutronHPDataPoint *)malloc(size);// Allocate array on host
	cudaMalloc((void **) &theData_d, size);
	cudaMalloc((void **) &minIdx_d, sizeof(int));
	cudaMalloc((void **) &minArray_d, size/2);
	
	// Initialize host array and copy it to CUDA device
	for(int i = 0; i < N; i++)
	{
		theData_h[i] = G4NeutronHPDataPoint(i,i);
	}

	 cudaMemcpy(theData_d, theData_h, size, cudaMemcpyHostToDevice);

	 
	int block_size = 64;
	//int n_blocks = (N)/block_size + ((N)%block_size == 0 ? 0:1);	 
	 
	 //Generate an array of query values
	 for(int i = 0; i < querySize; i++){
		queryArray_h[i] = N - i - 1;
	 }
	 // put the Array on GPU
	 
	 int queryBlocks = querySize/block_size + (querySize%block_size == 0 ? 0:1);

	 
	//printf("PRE: %f\n", theData_h[20].energy);
	// for(int k = 0; k < events; k++)
	// {
		// timesCUDA<<< n_blocks, block_size>>>(theData_d, 2);
	// }
	cudaMemcpy(theData_h, theData_d, size, cudaMemcpyDeviceToHost);
	printf("POST: %f \n", theData_h[20].energy);
	G4double resultVal = 0;
	int dataChunk = 2;
	int threadNum = N/dataChunk;
	int arrayBlocks = threadNum/block_size + (threadNum%block_size == 0 ? 0:1);
	for(int j = 0; j < events; j++){
		 SetArrayTo <<< queryBlocks, block_size >>>(resArray_d, querySize, N-1);										// Set the resArray values all to N + 1
		 cudaMemcpy(queryArray_d, queryArray_h, querySize*sizeof(G4double), cudaMemcpyHostToDevice);					// copy the query List to the GPU
		 findMinArray2 <<< arrayBlocks, block_size >>> (theData_d, queryArray_d, resArray_d, threadNum, querySize, N);	// find the first index that whose energy is greater than the query energy
		 resolveMinIndexArray <<<queryBlocks, block_size >>>(theData_d, resArray_d, queryArray_d, querySize);							// find the xSec values for those indexes
		 cudaMemcpy(queryArray_h, queryArray_d, querySize*sizeof(G4double), cudaMemcpyDeviceToHost);						// copy the result indexes back to the host
		 //cudaMemcpy(queryArray_h, queryArray_d, querySize*sizeof(G4double), cudaMemcpyDeviceToHost);

		// //printf("Hello\n");
		// //findMin2 <<< n_blocks, block_size >>> (theData_d, minArray_d, 500, N);
		// //findMin <<< n_blocks, block_size >>> (theData_d, 500, minIdx_d);
	 }

	//cudaMemcpy(&resultVal, &minArray_d[0].energy, sizeof(G4double), cudaMemcpyDeviceToHost);
	//printf("Result %f\n", resultVal);
	// G4double *resultVal;
	// for(int i = 0; i < events; i++)
	// {
		// // // Do calculation on device:
		 // setMin <<< 1,1>>> (minIdx_d);
		 // findMin <<< n_blocks, block_size >>> (theData_d, 50, m-inIdx_d);
	
		 // //cudaMemcpy(resultVal, &theData_d[minIdx_d], sizeof(G4double), cudaMemcpyDeviceToHost);
	// }
	

	//CPU Implementation 
	for(int j = 0; j < events; j++){			// events
		for(int k = 0; k < querySize; k++){ 	// for each item in the queryArray
			for(int i=0 ; i<N; i++){			// find the element
				if(theData_h[i].energy > queryArray_h[k]) {
					queryArray_h[k] = theData_h[i].xSec;
					break;
				}
			}
		}
	}

	free(queryArray_h);
	cudaFree(queryArray_d);
	
	free(theData_h);
	//free(resArray_h);
	cudaFree(theData_d);
	cudaFree(minIdx_d);
	cudaFree(minArray_d);
	cudaFree(resArray_d);

    return 0;
}