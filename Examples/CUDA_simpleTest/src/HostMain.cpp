#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include "DeviceMain.h"

int main(int argc, const char * argv[]) {
	int n = 1000000;
	
	std::clock_t start = std::clock();
	
	printf("\nRunning on CPU: \n");
	
	float *a_h;
	a_h = (float *)malloc(sizeof(float)*n);

	for (int i = 0; i < n; i++) {
		a_h[i] = (float)i;
	}
	for (int i = 0; i < n; i++) {
		a_h[i] = a_h[i] * a_h[i];
	}

	std::clock_t end = std::clock();
	
	double elapsed = double(end - start) / CLOCKS_PER_SEC;

	printf("Result: %f | Time: %f\n\n", a_h[n-1], elapsed);

	std::clock_t start2 = std::clock();
	
	printf("\nRunning on GPU:\n");
	float res;
	res = squareArray(n);

	std::clock_t end2 = std::clock();
	double elapsed2 = double(end2 - start2) / CLOCKS_PER_SEC;
	printf("Result: %f | Time: %f\n\n", res, elapsed2);
}
