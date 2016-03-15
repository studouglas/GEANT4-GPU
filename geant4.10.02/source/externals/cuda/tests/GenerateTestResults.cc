#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include "G4ParticleHPVector.hh"

G4ParticleHPVector** vectors;
std::ofstream resultsFile;
std::ofstream timesFile;

typedef char Bytef;
typedef unsigned long uLongf;
typedef unsigned long uLong;

void printArrayToFile(double* arr, int count, std::ofstream & file) {
	// empty array
	if (count == 0) {
		file << "[]";
		return;
	}

	// print each element, comma-separated
	file << "[";
	for (int i = 0; i < count - 1; i++) {
		file << arr[i] << ",";
	}
	file << arr[count-1] << "]";
}

void writeOutTestName(const char* testName, int caseNum) {
	// printf("Testing '%s'...\n", testName);
	// printf("here");

	char testNameToPrint[128];
	sprintf(testNameToPrint, "Test#%s\n", testName);
	resultsFile << testNameToPrint;
	timesFile << testNameToPrint;
}

void writeOutTheData(int caseNum) {
	int nEntries = vectors[caseNum]->GetVectorLength();
	double xVals[nEntries];
	double yVals[nEntries];
	for (int i = 0; i < nEntries; i++) {
		xVals[i] = vectors[caseNum]->GetX(i);
		yVals[i] = vectors[caseNum]->GetY(i);
	}
	printArrayToFile(xVals, nEntries, resultsFile);
	resultsFile << "\n";
	printArrayToFile(yVals, nEntries, resultsFile);
	resultsFile << "\n";
}

void testInitializeVector(int caseNum) {
	writeOutTestName("Init", caseNum);

	*(vectors[caseNum]) = G4ParticleHPVector();
	std::filebuf fileBuffer;

	G4String *data = NULL;
	G4String dataFileName;

	switch (caseNum) {
		case 0:
			dataFileName = "/Users/stuart/Desktop/test.txt";
			break;
		default:
			dataFileName = "/Users/stuart/Desktop/test.txt";
	}
	if (fileBuffer.open(dataFileName, std::ios::in)) {
		std::istream dataStream(&fileBuffer);
		
		int n;
		dataStream >> n;

		vectors[caseNum]->Init(dataStream, n, 1, 1);
		fileBuffer.close();
	} else {
		printf("\n\n***ERROR READING FILE***\n\n");
	}

	writeOutTheData(caseNum);
}

void testSetters(int caseNum) {
	printf("Testing setters...\n");
	resultsFile << "Setters\n";
	timesFile << "Setters\n";


}

void testGetters(int caseNum) {
	printf("Testing getters...\n");
	resultsFile << "Getters\n";
	timesFile << "Getters\n";


}

void testIntegration(int caseNum) {
	printf("Testing integration...\n");
	resultsFile << "Integration\n";
	timesFile << "Integration\n";


}

// usage: ./GenerateTestResults 0
int main(int argc, char** argv) {
	printf("\n\n");
	if (argc < 2) {
		printf("Usage: './GenerateTestResults N' where N is 1 if Geant4 compiled with CUDA, 0 otherwise.\n");
		return 1;
	}
	int cudaEnabled = atoi(argv[1]);

	char resultsFileName[128];
	char timesFileName[128];
	sprintf(resultsFileName, "UnitTest_Results_%s.txt", ((cudaEnabled) ? "GPU" : "CPU"));
	sprintf(timesFileName, "UnitTest_Times_%s.txt", ((cudaEnabled) ? "GPU" : "CPU"));
	
	printf("Geant4 compiled with CUDA: %s\n\n", ((cudaEnabled) ? "ON" : "OFF"));

	resultsFile.open(resultsFileName);
	timesFile.open(timesFileName);
	
	int numCases = 1;
	vectors = (G4ParticleHPVector**)malloc(numCases * sizeof(G4ParticleHPVector*));

	// run tests
	for (int i = 0; i < numCases; i++) {
		testInitializeVector(i);
		testSetters(i);
		testGetters(i);
		testIntegration(i);
	}

	printf("\nAll tests complete.\n\n");

	// close our file
	resultsFile.close();
	timesFile.close();
	return 0;
}