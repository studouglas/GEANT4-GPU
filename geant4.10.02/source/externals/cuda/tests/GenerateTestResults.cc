#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include "G4ParticleHPVector.hh"

#define NUM_TEST_INPUTS 5

G4ParticleHPVector** vectors;
std::ofstream resultsFile;
std::ofstream timesFile;

/***********************************************
* Write out to files
***********************************************/
void printArrayToFile(double* arr, int count) {
	// empty array
	if (count == 0) {
		resultsFile << "[]";
		return;
	}

	// print each element, comma-separated
	resultsFile << "[";
	for (int i = 0; i < count - 1; i++) {
		resultsFile << arr[i] << ",";
	}
	resultsFile << arr[count-1] << "]";
}
void writeOutTestName(std::string testName, int caseNum) {
	std::cout << "Testing '" << testName << "' case " << caseNum << "...\n";

	char testNameToPrint[128];
	sprintf(testNameToPrint, "#%s_%d\n", testName.c_str(), caseNum);
	
	resultsFile << testNameToPrint;
	timesFile << testNameToPrint;
}
void writeOutInt(int val) {
	resultsFile << val << "\n";
}
void writeOutDouble(double val) {
	resultsFile << val << "\n";
}
void writeOutString(std::string str) {
	resultsFile << str << "\n";
}
void writeOutPoint(G4ParticleHPDataPoint point) {
	resultsFile << "(" << point.GetX() << "," << point.GetY() << ")\n";
}
void writeOutIntInput(std::string inputName, int val) {
	resultsFile << "@" << inputName << "=" << val << "\n";
}
void writeOutDoubleInput(std::string inputName, double val) {
	resultsFile << "@" << inputName << "=" << val << "\n";	
}
void writeOutTheData(int caseNum) {
	int nEntries = vectors[caseNum]->GetVectorLength();
	double xVals[nEntries];
	double yVals[nEntries];
	for (int i = 0; i < nEntries; i++) {
		xVals[i] = vectors[caseNum]->GetX(i);
		yVals[i] = vectors[caseNum]->GetY(i);
	}
	
	printArrayToFile(xVals, nEntries);
	resultsFile << "\n";
	printArrayToFile(yVals, nEntries);
	resultsFile << "\n";
}
void writeOutTheIntegral(int caseNum) {
	double *integral = vectors[caseNum]->Debug();
	printArrayToFile(integral, vectors[caseNum]->GetVectorLength());
}
void writeOutTime(clock_t diff) {
	double secondsElapsed = diff;
	timesFile << secondsElapsed << " cycles\n";
}

/***********************************************
* Helper funtions
***********************************************/
double randDouble() {
	return (double)(rand() / RAND_MAX);
}
int* testValuesForI(int caseNum) {
	int* testVals = (int*)malloc(NUM_TEST_INPUTS * sizeof(int));
	testVals[0] = -1;
	testVals[1] = 0;
	testVals[2] = vectors[caseNum]->GetVectorLength() / 2;
	testVals[3] = vectors[caseNum]->GetVectorLength() - 1;
	testVals[4] = vectors[caseNum]->GetVectorLength();
	return testVals;
}

/***********************************************
* Run unit tests
***********************************************/
void testInitializeVector(int caseNum) {
	writeOutTestName("void Init(std::istream & aDataFile, G4int total, G4double ux=1., G4double uy=1.)", caseNum);

	vectors[caseNum] = new G4ParticleHPVector();
	std::filebuf fileBuffer;

	G4String *data = NULL;
	std::string dataFileName;

	// different data files for different cases
	switch (caseNum) {
		case 0:
			writeOutTheData(caseNum);
			return;
		case 1:
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
void testSettersAndGetters(int caseNum) {
	int* testVals = testValuesForI(caseNum);

	const int NUM_SETTER_TESTS = 11;
	std::string testNames[NUM_SETTER_TESTS];
	testNames[0] = "void SetPoint(G4int i, const G4ParticleHPDataPoint & it)";
	testNames[1] = "void SetData(G4int i, G4double x, G4double y)";
	testNames[2] = "void SetX(G4int i, G4double e)";
	testNames[3] = "void SetEnergy(G4int i, G4double e)";
	testNames[4] = "void SetY(G4int i, G4double x)";
	testNames[5] = "void SetXsec(G4int i, G4double x)";
	testNames[6] = "const G4ParticleHPDataPoint GetPoint(G4int i)";
	testNames[7] = "G4double GetEnergy(G4int i)";
	testNames[8] = "G4double GetX(G4int i)";
	testNames[9] = "G4double GetXsec(G4int i)";
	testNames[10] = "G4double GetY(G4int i)";

	for (int testType = 0; testType < NUM_SETTER_TESTS; testType++) {
		writeOutTestName(testNames[testType], caseNum);
		
		for (int inputIndex = 0; inputIndex < NUM_TEST_INPUTS; inputIndex++) {
			writeOutIntInput("i", testVals[inputIndex]);
			G4ParticleHPDataPoint point = G4ParticleHPDataPoint(randDouble(), randDouble());
			try {
				switch(testType) {
					case 0:
						vectors[caseNum]->SetPoint(testVals[inputIndex], point);
						break;
					case 1:
						vectors[caseNum]->SetData(testVals[inputIndex], randDouble(), randDouble());
						break;
					case 2:
						vectors[caseNum]->SetX(testVals[inputIndex], randDouble());
						break;
					case 3:
						vectors[caseNum]->SetEnergy(testVals[inputIndex], randDouble());
						break;
					case 4:
						vectors[caseNum]->SetY(testVals[inputIndex], randDouble());
						break;
					case 5:
						vectors[caseNum]->SetXsec(testVals[inputIndex], randDouble());
						break;
					case 6:
						writeOutPoint(vectors[caseNum]->GetPoint(testVals[inputIndex]));
						break;
					case 7:
						writeOutDouble(vectors[caseNum]->GetEnergy(testVals[inputIndex]));
						break;
					case 8:
						writeOutDouble(vectors[caseNum]->GetX(testVals[inputIndex]));
						break;
					case 9:
						writeOutDouble(vectors[caseNum]->GetXsec(testVals[inputIndex]));
						break;
					case 10:
						writeOutDouble(vectors[caseNum]->GetY(testVals[inputIndex]));
						break;
				}
				if (testType <= 5) {
					writeOutTheData(caseNum);
				}
			} catch (G4HadronicException e) {
				writeOutString("Caught G4HadronicException");
			}
		}
	}
}
void testGetXSec(int caseNum) {
	writeOutTestName("G4double GetXsec(G4double e)", caseNum);
	writeOutTestName("G4double GetXsec(G4double e, G4int min)", caseNum);
	int* testVals = testValuesForI(caseNum);
	for (int i = 0; i < NUM_TEST_INPUTS; i++) {

	}
}
void testThinOut(int caseNum) {
	writeOutTestName("void ThinOut(G4double precision)", caseNum);
	int* testVals = testValuesForI(caseNum);
	for (int i = 0; i < NUM_TEST_INPUTS; i++) {

	}
}
void testMerge(int caseNum) {
	writeOutTestName("void Merge(G4ParticleHPVector * active, G4ParticleHPVector * passive)", caseNum);
	writeOutTestName("void Merge(G4InterpolationScheme aScheme, G4double aValue, G4ParticleHPVector * active, G4ParticleHPVector * passive)", caseNum);
}
void testSample(int caseNum) {
	writeOutTestName("G4double SampleLin()", caseNum);
	writeOutTestName("G4double Sample()", caseNum);
}
void testGetBorder(int caseNum) {
	writeOutTestName("G4double Get15PercentBorder", caseNum);
	writeOutTestName("G4double Get50PercentBorder", caseNum);
}
void testIntegral(int caseNum) {
	writeOutTestName("void Integrate()", caseNum);
	writeOutTestName("void IntegrateAndNormalise()", caseNum);
}
void testTimes(int caseNum) {
	double testVals[NUM_TEST_INPUTS];
	for (int i = 0; i < NUM_TEST_INPUTS; i++) {
		testVals[i] = randDouble();
	}

	writeOutTestName("void Times(G4double factor)", caseNum);
	writeOutTheData(caseNum);
	// writeOutTheIntegral(caseNum);
}
void testAssignment(int caseNum) {

}

/***********************************************
* usage: ./GenerateTestResults 0
***********************************************/
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
	// if (access(resultsFileName, F_OK) != -1) {
	// 	char userResponse;
	// 	do {
	// 		std::cout << "'" << resultsFileName << "' already exists, do you want to overwrite it [y/n]?\n> ";
	// 		std::cin >> userResponse;
	// 	} while (!std::cin.fail() && userResponse != 'y' && userResponse != 'n');
	// 	std::cout << "\n";
	// 	if (userResponse == 'n') {
	// 		return 1;
	// 	}
	// }

	resultsFile.open(resultsFileName);
	timesFile.open(timesFileName);
	
	int numCases = 2;
	vectors = (G4ParticleHPVector**)malloc(numCases * sizeof(G4ParticleHPVector*));

	// run tests
	for (int i = 0; i < numCases; i++) {
		testInitializeVector(i);
		testSettersAndGetters(i);
		testGetXSec(i);
		testThinOut(i);
		testMerge(i);
		testSample(i);
		testGetBorder(i);
		testIntegral(i);
		testTimes(i);
		testAssignment(i);
	}

	std::cout << "\nAll tests complete.\n\n";

	// close our file
	resultsFile.close();
	timesFile.close();
	return 0;
}
