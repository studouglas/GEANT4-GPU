#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include "G4ParticleHPVector.hh"

G4ParticleHPVector** vectors;
std::ofstream resultsFile;
std::ofstream timesFile;

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

void writeOutTestName(std::string testName, int caseNum) {
	std::cout << "Testing '" << testName << "'...\n";

	char testNameToPrint[128];
	sprintf(testNameToPrint, "Test#%s\n", testName.c_str());
	
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

void writeOutTime(clock_t diff) {
	double secondsElapsed = diff;
	timesFile << secondsElapsed << " cycles\n";
}

void testInitializeVector(int caseNum) {
	writeOutTestName("Init", caseNum);

	vectors[caseNum] = new G4ParticleHPVector();
	std::filebuf fileBuffer;

	G4String *data = NULL;
	std::string dataFileName;

	switch (caseNum) {
		case 0:
			dataFileName = "Lead_66.txt";
			break;
		default:
			dataFileName = "Lead_66.txt";
	}
	if (fileBuffer.open(dataFileName, std::ios::in)) {
		std::istream dataStream(&fileBuffer);
		
		int n;
		dataStream >> n;

		clock_t t1 = clock();
		vectors[caseNum]->Init(dataStream, n, 1, 1);
		clock_t t2 = clock();
		writeOutTime(t2-t1);

		fileBuffer.close();
	} else {
		std::cout << "\n\n***ERROR READING FILE***\n\n";
	}

	writeOutTheData(caseNum);
}

void testSetters(int caseNum) {
	writeOutTestName("Setters", caseNum);
}

void testGetters(int caseNum) {
	writeOutTestName("Getters", caseNum);
}

void testIntegration(int caseNum) {
	writeOutTestName("Integration", caseNum);
}

// usage: ./GenerateTestResults 0
int main(int argc, char** argv) {
	std::cout << "\n\n";
	if (argc < 2) {
		std::cout << "Usage: './GenerateTestResults N' where N is 1 if Geant4 compiled with CUDA, 0 otherwise.\n";
		return 1;
	}
	int cudaEnabled = atoi(argv[1]);

	char resultsFileName[128];
	char timesFileName[128];
	sprintf(resultsFileName, "UnitTest_Results_%s.txt", ((cudaEnabled) ? "GPU" : "CPU"));
	sprintf(timesFileName, "UnitTest_Times_%s.txt", ((cudaEnabled) ? "GPU" : "CPU"));
	
	std::cout << "Geant4 compiled with CUDA: " << ((cudaEnabled) ? "ON" : "OFF") << "\n\n";

	// check if file exists, confirm overwrite
	if (access(resultsFileName, F_OK) != -1) {
		char userResponse;
		do {
			std::cout << "'" << resultsFileName << "' already exists, do you want to overwrite it [y/n]?\n> ";
			std::cin >> userResponse;
		} while (!std::cin.fail() && userResponse != 'y' && userResponse != 'n');
		std::cout << "\n";
		if (userResponse == 'n') {
			return 1;
		}
	}

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

	std::cout << "\nAll tests complete.\n\n";

	// close our file
	resultsFile.close();
	timesFile.close();
	return 0;
}
