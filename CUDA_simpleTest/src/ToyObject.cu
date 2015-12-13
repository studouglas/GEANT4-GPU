#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "DeviceMain.h"
#include <vector>

class ToyClass
{
	public:
	int* data;
	
	ToyClass(int x)
	{
		data = new int[1];
		data[0] = x;
	}
	void add_one()
	{
		data[0] = data[0] + 1;
	}
};

__global__ void useClass(ToyClass *toyClass)
{
	printf("%d\n", toyClass->data[0]);
}

int main()
{
	ToyClass c(1);
	// create class storage on device and copy top level class
	ToyClass *d_c;
	cudaMalloc((void **)&d_c, sizeof(ToyClass));
	cudaMemcpy(d_c, &c, sizeof(ToyClass), cudaMemcpyHostToDevice);
	// make an allocated region on device for use by pointer in class
	int *hostdata;
	cudaMalloc((void **)&hostdata, sizeof(int));
	cudaMemcpy(hostdata, c.data, sizeof(int), cudaMemcpyHostToDevice);
	// copy pointer to allocated device storage to device class
	cudaMemcpy(&(d_c->data), &hostdata, sizeof(int *), cudaMemcpyHostToDevice);
	useClass<<<1,1>>>(d_c);
	cudaDeviceSynchronize();
	
	return 0;



}