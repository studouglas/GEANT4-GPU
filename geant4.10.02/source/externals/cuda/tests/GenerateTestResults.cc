#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include <sys/time.h>
#include "G4ParticleHPVector.hh"

// test will detect if CPU and GPU result files generated from different versions
// of the unit test program
#define VERSION_NUMBER "1.0"

// number of different G4ParticleHPVectors to create
#define NUM_TEST_CASES 7

// number of different input values to test (including 'edge case' values)
#define NUM_TEST_INPUTS 5 

// 10^n where n is number of digits to keep
// TODO: try more (17 is to much)
#define DOUBLE_PRECISION 10000000000

// some tests based on functions that use random values, run these multiple times and take average
#define REPETITIONS_RAND_TESTS 1000

// Geant4 doesn't use rand() on CUDA, but does on CPU so rands become mismatched
// To fix this, we generate a number of rands at init (we only need 221, but might as be safe)
#define NUM_RANDS 5000
int randCounter = 0;
double* rands;

G4ParticleHPVector** vectors;
std::ofstream resultsFile;
std::ofstream timesFile;

// used by analyzer to see what line in file is
const char variableId = '@';
const char methodId = '#';
const char nEntriesId = '!';
const char randomResultId = '$';

/***********************************************
* Write out to files
***********************************************/
void writeOutArray(double* arr, int count) {
	// empty array
	if (count == 0) {
		resultsFile << "[]";
		return;
	}
	std::cout << "print arr\n";
	// round our doubles within tolerance before calculating hash
	for (int i = 0; i < count; i++) {
		arr[i] = round(arr[i]*DOUBLE_PRECISION)/DOUBLE_PRECISION;
	}

	// convert array into string (double* -> char* -> std::string)
	char* charArr = reinterpret_cast<char*>(arr);
	std::string arrAsString(reinterpret_cast<char const*>(charArr), count * sizeof(double));

	// use std function to hash the array (don't want to print out each element for big arrays)
	std::size_t arrayHash = std::hash<std::string>{}(arrAsString);

	resultsFile << "hash: " << arrayHash;
}
void writeOutTestName(std::string testName, int caseNum) {
	std::cout << "Testing '" << testName << "' case " << caseNum << "...\n";

	char testNameToPrint[128];
	sprintf(testNameToPrint, "%c%s_%d\n", methodId, testName.c_str(), caseNum);
	
	resultsFile << testNameToPrint;
	timesFile << testNameToPrint;
}
void writeOutInt(int val) {
	resultsFile << val << "\n";
}
void writeOutDouble(double val) {
	resultsFile << val << "\n";
}
void writeOutRandomDouble(double val) {
	resultsFile << randomResultId << val << "\n";
}
void writeOutString(std::string str) {
	resultsFile << str << "\n";
}
void writeOutPoint(G4ParticleHPDataPoint point) {
	resultsFile << "(" << point.GetX() << "," << point.GetY() << ")\n";
}
void writeOutIntInput(std::string inputName, int val) {
	resultsFile << variableId << inputName << "=" << val << "\n";
	timesFile << variableId << inputName << "=" << val << "\n";
	std::cout << "    Input: " << inputName << "=" << val <<"\n";
}
void writeOutDoubleInput(std::string inputName, double val) {
	resultsFile << variableId << inputName << "=" << val << "\n";	
	timesFile << variableId << inputName << "=" << val << "\n";	
	std::cout << "    Input: " << inputName << "=" << val <<"\n";
}
void writeOutTheData(int caseNum) {
	int nEntries = vectors[caseNum]->GetVectorLength();
	double xVals[nEntries];
	double yVals[nEntries];
	for (int i = 0; i < nEntries; i++) {
		xVals[i] = vectors[caseNum]->GetX(i);
		yVals[i] = vectors[caseNum]->GetY(i);
	}

	resultsFile << "theData xVals ";
	writeOutArray(xVals, nEntries);
	resultsFile << "\n";

	resultsFile << "theData yVals ";
	writeOutArray(yVals, nEntries);
	resultsFile << "\n";
}
void writeOutTheIntegral(int caseNum) {
	double *integral = vectors[caseNum]->Debug();
	resultsFile << "theIntegral ";
	if (integral == NULL) {
		writeOutArray(integral, 0);
		resultsFile << "\n";
	} else {
		if (integral[0] == 0.0 || integral[0] == -0.0) {
			integral[0] = 1.0-1.0;
		}
		writeOutArray(integral, vectors[caseNum]->GetVectorLength());
		resultsFile << "\n";
	}
}
void writeOutTime(double diff) {
	// double secondsElapsed = diff;
	timesFile << diff << "\n";
}


/***********************************************
* Helper funtions
***********************************************/
double randDouble() {
	randCounter = (randCounter + 1) % NUM_RANDS;
	return rands[randCounter];
}
int* testValuesForI(int caseNum) {
	int* testVals = (int*)malloc(NUM_TEST_INPUTS * sizeof(int));
	
	// we want edge cases, and a couple normal cases
	testVals[0] = -1;
	testVals[1] = 0;
	testVals[2] = vectors[caseNum]->GetVectorLength() / 2;
	testVals[3] = vectors[caseNum]->GetVectorLength() - 1;
	testVals[4] = vectors[caseNum]->GetVectorLength();
	
	return testVals;
}
double* testValuesForDoubles() {
	double* testVals = (double*)malloc(NUM_TEST_INPUTS * sizeof(double));
	
	// cover the edge cases, and some random numbers for normal cases
	testVals[0] = -1.0;
	testVals[1] = 0.0;
	testVals[2] = 0.00051234;
	testVals[3] = 1.5892317;
	testVals[4] = 513.1869340;

	return testVals;
}
double getWallTime() {
	struct timeval time;
	gettimeofday(&time, NULL);
	return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
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
			timesFile << nEntriesId << caseNum << nEntriesId << 0 << "\n";
			return;
		case 1:
			dataFileName = "58_141_Cerium_80.txt";
			break;
		case 2:
			dataFileName = "90_228_Thorium_1509.txt";
			break;
		case 3:
			dataFileName = "92_232_Uranium_8045.txt";
			break;
		case 4:
			dataFileName = "92_236_Uranium_41854.txt";
			break;
		case 5:
			dataFileName = "90_232_Thorium_98995.txt";
			break;
		case 6:
			dataFileName = "92_235_Uranium_242594.txt";
			break;
		default:
			std::cout << "Error. Filename not set for case " << caseNum << ".\n";
			exit(1);
	}

	if (fileBuffer.open(dataFileName, std::ios::in)) {
		std::istream dataStream(&fileBuffer);
		
		int n;
		dataStream >> n;
		double t1 = getWallTime();
		vectors[caseNum]->Init(dataStream, n, 1, 1);
		double t2 = getWallTime();
		writeOutTime(t2-t1);

		fileBuffer.close();
	} else {
		std::cout << "\n\n***ERROR READING FILE '" << dataFileName << "'***\n\n";
	}
	timesFile << nEntriesId << caseNum << nEntriesId << vectors[caseNum]->GetVectorLength() << "\n";
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
			
			try {
				double x = randDouble();
				double y = randDouble();
				G4ParticleHPDataPoint point = G4ParticleHPDataPoint(x, y); // must be outside switch statement for compiler reasons
				switch(testType) {
					case 0:
						writeOutDoubleInput("x", point.GetX());
						writeOutDoubleInput("y", point.GetY());
						vectors[caseNum]->SetPoint(testVals[inputIndex], point);
						break;
					case 1:
						writeOutDoubleInput("x", x);
						writeOutDoubleInput("y", y);
						vectors[caseNum]->SetData(testVals[inputIndex], x, y);
						break;
					case 2:
						writeOutDoubleInput("x", x);
						vectors[caseNum]->SetX(testVals[inputIndex], x);
						break;
					case 3:
						writeOutDoubleInput("x", x);
						vectors[caseNum]->SetEnergy(testVals[inputIndex], x);
						break;
					case 4:
						writeOutDoubleInput("x", x);
						vectors[caseNum]->SetY(testVals[inputIndex], x);
						break;
					case 5:
						writeOutDoubleInput("x", x);
						vectors[caseNum]->SetXsec(testVals[inputIndex], x);
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
	
	// remove point we just added for our 0-vector so can continue to test on empty vector
	if (caseNum == 0) {
		// vectors[caseNum]->nEntries = 0;
	}
	free(testVals);
}
void testGetXSec(int caseNum) {
	double* testVals = testValuesForDoubles();
	
	writeOutTestName("G4double GetXsec(G4double e)", caseNum);
	for (int i = 0; i < NUM_TEST_INPUTS; i++) {
		writeOutDoubleInput("e", testVals[i]);
		try {
			double t1 = getWallTime();
			writeOutDouble(vectors[caseNum]->GetXsec(testVals[i]));
			double t2 = getWallTime();
			writeOutTime(t2-t1);
		} catch (G4HadronicException e)  {
			resultsFile << "Caught G4HadronicException" << "\n";
		}
	}

	writeOutTestName("G4double GetXsec(G4double e, G4int min)", caseNum);
	int* minVals = testValuesForI(caseNum);
	for (int i = 0; i < NUM_TEST_INPUTS; i++) {
		for (int j = 0; j < NUM_TEST_INPUTS; j++) {
			writeOutDoubleInput("e", testVals[i]);
			writeOutDoubleInput("min", minVals[j]);
			try {
				double t1 = getWallTime();
				std::cout << "pre write double\n";
				writeOutDouble(vectors[caseNum]->GetXsec(testVals[i], minVals[j]));
				std::cout << "post write double\n";
				double t2 = getWallTime();
				writeOutTime(t2-t1);
			} catch (G4HadronicException e)  {
				resultsFile << "Caught G4HadronicException" << "\n";
			}
		}
	}
	free(testVals);
	free(minVals);
}
void testThinOut(int caseNum) {
	writeOutTestName("void ThinOut(G4double precision)", caseNum);
	double* testVals = testValuesForDoubles();
	for (int i = 0; i < NUM_TEST_INPUTS; i++) {
		writeOutDoubleInput("precision", testVals[i]);
		double t1 = getWallTime();
		vectors[caseNum]->ThinOut(testVals[i]);
		writeOutTime(getWallTime() - t1);
		writeOutTheData(caseNum);
	}
	free(testVals);
}
void testSample(int caseNum) {
	writeOutTestName("G4double SampleLin()", caseNum);
	
	// includes random numbers, so run multiple times and take average
	double t1 = getWallTime();
	double sum = 0.0;
	for (int i = 0; i < REPETITIONS_RAND_TESTS; i++) {
		sum += vectors[caseNum]->SampleLin();
	}
	writeOutTime(getWallTime() - t1);
	writeOutRandomDouble(sum/REPETITIONS_RAND_TESTS);

	// includes random numbers, so run multiple times and take average
	writeOutTestName("G4double Sample()", caseNum);
	t1 = getWallTime();
	sum = 0.0;
	for (int i = 0; i < REPETITIONS_RAND_TESTS; i++) {
		sum += vectors[caseNum]->Sample();
	}
	writeOutTime(getWallTime() - t1);
	writeOutRandomDouble(sum/REPETITIONS_RAND_TESTS);
}
void testGetBorder(int caseNum) {
	writeOutTestName("G4double Get15PercentBorder()", caseNum);
	double t1 = getWallTime();
	writeOutDouble(vectors[caseNum]->Get15percentBorder());
	writeOutTime(getWallTime() - t1);

	writeOutTestName("G4double Get50PercentBorder()", caseNum);
	t1 = getWallTime();
	writeOutDouble(vectors[caseNum]->Get50percentBorder());
	writeOutTime(getWallTime() - t1);
}
// todo: re-enable (crashes on GPU)
void testIntegral(int caseNum) {
	writeOutTestName("void Integrate()", caseNum);
	double t1 = getWallTime();
	vectors[caseNum]->Integrate();
	writeOutTime(getWallTime() - t1);
	writeOutTheIntegral(caseNum);
	

	writeOutTestName("void IntegrateAndNormalise()", caseNum);
	t1 = getWallTime();
	vectors[caseNum]->IntegrateAndNormalise();
	writeOutTime (getWallTime() - t1);
	writeOutTheIntegral(caseNum);
	
}
void testTimes(int caseNum) {
	writeOutTestName("void Times(G4double factor)", caseNum);
	
	// vectors[caseNum]->Dump();
	// multiply by several random factors, then by their 
	// inverse to get back to original values
	const int numTimesInputs = 4;
	double testVals[numTimesInputs] = {-4.67, -0.00001525, 0.000909820, 1.0}; // final val will store inverse
	for (int i = 0; i < numTimesInputs; i++) {
		if (i == numTimesInputs - 1) {
			testVals[i] = 1.0/testVals[i];
		}
		
		writeOutDoubleInput("factor", testVals[i]);
		double t1 = getWallTime();
		vectors[caseNum]->Times(testVals[i]);
		double t2 = getWallTime();
		// vectors[caseNum]->Dump();
		writeOutTime(t2-t1);

		writeOutTheData(caseNum);
		writeOutTheIntegral(caseNum);

		testVals[numTimesInputs-1] *= testVals[i];
	}
}
void testAssignment(int caseNum) {
	writeOutTestName("G4ParticleHPVector & operator = (const G4ParticleHPVector & right)", caseNum);
	double t1 = getWallTime();
	G4ParticleHPVector vec = *(vectors[caseNum]);
	double t2 = getWallTime();
	writeOutTime(t2-t1);

	int nEntries = vec.GetVectorLength();
	double xVals[nEntries];
	double yVals[nEntries];
	for (int i = 0; i < nEntries; i++) {
		xVals[i] = vec.GetX(i);
		yVals[i] = vec.GetY(i);
	}
	
	writeOutArray(xVals, nEntries);
	resultsFile << "\n";
}

/***********************************************
* usage: ./GenerateTestResults 0
***********************************************/
int main(int argc, char** argv) {
	std::cout << "\n\n";
	
	if (argc < 2 || (atoi(argv[1]) != 0 && atoi(argv[1]) != 1)) {
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
	
	resultsFile << "G4ParticleHPVector_CUDA Unit Test Version: " << VERSION_NUMBER << "\n";

	vectors = (G4ParticleHPVector**)malloc(NUM_TEST_CASES * sizeof(G4ParticleHPVector*));

	// populate rands with doubles between 0 and 1
	srand(1);
	rands = (G4double*)malloc(NUM_RANDS * sizeof(double));
	for (int i = 0; i < NUM_RANDS; i++) {
		rands[i] = (double)((double)rand()/(double)RAND_MAX);
	}

	// run tests
	for (int i = 0; i < NUM_TEST_CASES; i++) {
		testInitializeVector(i);
		testGetXSec(i);
		testSample(i);
		testGetBorder(i);
		// testIntegral(i);
		testTimes(i);
		testSettersAndGetters(i);
		testThinOut(i);
		testAssignment(i);
	}

	std::cout << "\nAll tests complete.\n\n";

	// close our file
	resultsFile.close();
	timesFile.close();

	return 0;
}
