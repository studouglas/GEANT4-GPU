#include "G4NeutronHPDataPoint.cu"
#include <stdlib.h>
#include <stdio.h>

__global__ 
void advanceParticles(float dt, G4NeutronHPDataPoint * pArray, int nParticles)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nParticles) { pArray[idx].SetX(dt); } 
} 

int main(int argc, char ** argv) 
{     
    int n = 100;     
    if(argc > 1) { n = atoi(argv[1]);}     // Number of particles
    if(argc > 2) { srand(atoi(argv[2])); } // Random seed

    G4NeutronHPDataPoint * theData = new G4NeutronHPDataPoint[n];
    G4NeutronHPDataPoint * devPArray = NULL;
    cudaMalloc(&devPArray, n*sizeof(G4NeutronHPDataPoint));
    cudaMemcpy(devPArray, theData, n*sizeof(G4NeutronHPDataPoint), cudaMemcpyHostToDevice);
    for(int i=0; i<10000000; i++)
    {   // Random distance each step
        float dt = (float)rand()/(float) RAND_MAX;
   	advanceParticles<<< 1 +  n/256, 256>>>(dt, devPArray, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(theData, devPArray, n*sizeof(G4NeutronHPDataPoint), cudaMemcpyDeviceToHost);
    G4double totalDistance = 0;
    G4double temp;
    for(int i=0; i<n; i++)
    {
        temp = theData[i].GetX();
        totalDistance += temp;
    }
    printf("Moved %d particles 100 steps. Total energy is (%f)", n, totalDistance);
    return 0;
}
