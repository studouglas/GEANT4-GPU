#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include "G4ParticleHPVector.hh"

// number of different G4ParticleHPVectors to create
#define NUM_TEST_CASES 2

// number of different input values to test (including 'edge case' values)
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
	timesFile << "@" << inputName << "=" << val << "\n";
}
void writeOutDoubleInput(std::string inputName, double val) {
	resultsFile << "@" << inputName << "=" << val << "\n";	
	timesFile << "@" << inputName << "=" << val << "\n";	
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
	if (integral == NULL) {
		printArrayToFile(integral, 0);
		resultsFile << "\n";
	} else {
		printArrayToFile(integral, vectors[caseNum]->GetVectorLength());
		resultsFile << "\n";
	}
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
	timesFile << "!Case_" << caseNum << ":nEntries=" << vectors[caseNum]->GetVectorLength() << "\n";
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
	free(testVals);
}
void testGetXSec(int caseNum) {
	double* testVals = testValuesForDoubles();
	
	writeOutTestName("G4double GetXsec(G4double e)", caseNum);
	for (int i = 0; i < NUM_TEST_INPUTS; i++) {
		writeOutDoubleInput("e", testVals[i]);
		try {
			clock_t t1 = clock();
			writeOutDouble(vectors[caseNum]->GetXsec(testVals[i]));
			clock_t t2 = clock();
			writeOutTime(t2-t1);
		} catch (G4HadronicException e)  {
			resultsFile << "Caught G4HadronicException" << "\n";
		}
	}

	writeOutTestName("G4double GetXsec(G4double e, G4int min)", caseNum);
	int* minVals = testValuesForI(caseNum);
	for (int i = 0; i < NUM_TEST_INPUTS; i++) {
		writeOutDoubleInput("e", testVals[i]);
		writeOutDoubleInput("min", minVals[i]);
		try {
			clock_t t1 = clock();
			writeOutDouble(vectors[caseNum]->GetXsec(testVals[i], minVals[i]));
			clock_t t2 = clock();
			writeOutTime(t2-t1);
		} catch (G4HadronicException e)  {
			resultsFile << "Caught G4HadronicException" << "\n";
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
		vectors[caseNum]->ThinOut(testVals[i]);
		writeOutTheData(caseNum);
	}
	free(testVals);
}
void testMerge(int caseNum) {
	// merge each of our test case vectors with each other one, including itself
	writeOutTestName("void Merge(G4ParticleHPVector * active, G4ParticleHPVector * passive)", caseNum);
	for (int i = 0; i < NUM_TEST_CASES; i++) {
		for (int j = 0; j < NUM_TEST_CASES; j++) {
			if (i != caseNum && j != caseNum) {
				// set up parameters
				writeOutIntInput("active_caseNum", i);
				writeOutIntInput("passive_caseNum", j);
				
				// perform merge
				clock_t t1 = clock();
				// vectors[caseNum]->Merge(new G4ParticleHPVector(vectors[i]), new G4ParticleHPVector(vectors[j]));
				clock_t t2 = clock();
				
				// record results
				writeOutTime(t2-t1);
				writeOutTheData(caseNum);
			}
		}
	}

	writeOutTestName("void Merge(G4InterpolationScheme aScheme, G4double aValue, G4ParticleHPVector * active, G4ParticleHPVector * passive)", caseNum);
	for (int i = 0; i < NUM_TEST_CASES; i++) {
		for (int j = 0; j < NUM_TEST_CASES; j++) {
			if (i != caseNum && j != caseNum) {
				// set up parameters
				G4InterpolationScheme scheme = G4InterpolationScheme();
				double val = randDouble();
				writeOutIntInput("aValue", val);
				writeOutIntInput("active_caseNum", i);
				writeOutIntInput("passive_caseNum", j);
				
				// perform merge
				clock_t t1 = clock();
				// vectors[caseNum]->Merge(vectors[i], vectors[j]);
				clock_t t2 = clock();
				
				// record results
				writeOutTime(t2-t1);
				writeOutTheData(caseNum);
			}
		}
	}
}
void testSample(int caseNum) {
	writeOutTestName("G4double SampleLin()", caseNum);
	writeOutDouble(vectors[caseNum]->SampleLin());

	writeOutTestName("G4double Sample()", caseNum);
	writeOutDouble(vectors[caseNum]->Sample());
}
void testGetBorder(int caseNum) {
	writeOutTestName("G4double Get15PercentBorder()", caseNum);
	writeOutDouble(vectors[caseNum]->Get15percentBorder());

	writeOutTestName("G4double Get50PercentBorder()", caseNum);
	writeOutDouble(vectors[caseNum]->Get50percentBorder());
}
void testIntegral(int caseNum) {
	writeOutTestName("void Integrate()", caseNum);
	clock_t t1 = clock();
	vectors[caseNum]->Integrate();
	clock_t t2 = clock();
	writeOutTheIntegral(caseNum);
	writeOutTime(t2-t1);

	writeOutTestName("void IntegrateAndNormalise()", caseNum);
	t1 = clock();
	vectors[caseNum]->IntegrateAndNormalise();
	t2 = clock();
	writeOutTheIntegral(caseNum);
	writeOutTime (t2-t1);
}
void testTimes(int caseNum) {
	writeOutTestName("void Times(G4double factor)", caseNum);
	
	// multiply by several random factors, then by their 
	// inverse to get back to original values
	const int numTimesInputs = 4;
	double testVals[numTimesInputs] = {-4.67, -0.00001525, 0.000909820, 1.0}; // final val will store inverse
	for (int i = 0; i < numTimesInputs; i++) {
		if (i == numTimesInputs - 1) {
			testVals[i] = 1.0/testVals[i];
		}
		
		writeOutDoubleInput("factor", testVals[i]);
		clock_t t1 = clock();
		vectors[caseNum]->Times(testVals[i]);
		clock_t t2 = clock();
		
		writeOutTime(t2-t1);
		writeOutTheData(caseNum);
		writeOutTheIntegral(caseNum);
		testVals[numTimesInputs-1] *= testVals[i];
	}
}
void testAssignment(int caseNum) {
	writeOutTestName("G4ParticleHPVector & operator = (const G4ParticleHPVector & right)", caseNum);
	clock_t t1 = clock();
	G4ParticleHPVector vec = *(vectors[caseNum]);
	clock_t t2 = clock();
	
	int nEntries = vec.GetVectorLength();
	double xVals[nEntries];
	double yVals[nEntries];
	for (int i = 0; i < nEntries; i++) {
		xVals[i] = vec.GetX(i);
		yVals[i] = vec.GetY(i);
	}
	
	printArrayToFile(xVals, nEntries);
	resultsFile << "\n";
	writeOutTime(t2-t1);
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
	
	vectors = (G4ParticleHPVector**)malloc(NUM_TEST_CASES * sizeof(G4ParticleHPVector*));

	// run tests
	for (int i = 0; i < NUM_TEST_CASES; i++) {
		testInitializeVector(i);
		testSettersAndGetters(i);
		testGetXSec(i);
		testMerge(i);
		testSample(i);
		testGetBorder(i);
		testIntegral(i);
		testTimes(i);
		testThinOut(i);
		testAssignment(i);
	}

	std::cout << "\nAll tests complete.\n\n";

	// close our file
	resultsFile.close();
	timesFile.close();
	return 0;
}
