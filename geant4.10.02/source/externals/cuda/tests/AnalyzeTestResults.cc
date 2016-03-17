#include <iostream>
#include <string>
#include <stdbool.h>
#include <unistd.h>
#include <fstream>
#include <stdio.h>

std::ifstream cpuResults;
std::ifstream gpuResults;

void compareTestResults() {
	int testsPassed = 0;
	int testsFailed = 0;

	char variableId = '@';
	char methodId = '#';

	std::string cpuLine;
	std::string gpuLine;

	std::string currentMethod = "";
	std::string currentCaseNum = "";
	std::string currentVariableName = "";
	
	while (true) {
		// read the next line
		if (!std::getline(cpuResults, cpuLine)) {
			break; 
		}
		if (!std::getline(gpuResults, gpuLine)) {
			break;
		}
		
		// method name identifier
		if (cpuLine.at(0) == methodId) {
			int caseNumSeparatorIndex = cpuLine.find_last_of("_");
			currentCaseNum = cpuLine.substr(caseNumSeparatorIndex+1, cpuLine.length());
			currentMethod = cpuLine.substr(1, caseNumSeparatorIndex-1);
			currentVariableName = "";
		} 

		// variable identifier
		else if (cpuLine.at(0) == variableId) {
			currentVariableName = cpuLine.substr(1);
		} 

		// result (i.e. array of theData, returned double, etc)
		else {
			if (cpuLine.compare(gpuLine) != 0) {
				std::cout << "FAILED: ";
				testsFailed++;
			} else {
				std::cout << "PASSED: ";
				testsPassed++;
			}
			
			std::cout << "case " << currentCaseNum << ", ";
			if (currentVariableName != "") {
				std::cout << currentVariableName << ", ";
			}
			std::cout << "'" << currentMethod << "'\n";
		}
	}

	// print aggregated results
	std::cout << "\n-------------------------------\n";
	std::cout << (double)(testsPassed/(testsPassed+testsFailed))*100.0 << "\% passed\n"; 
	std::cout << testsPassed << " tests passed out of " << testsFailed+testsPassed << "\n";
	std::cout << "-------------------------------\n\n";
}

void generateTimesCsv() {
	std::ofstream timesOutput;
}

int main(int argc, char** argv) {
	char cpuResultsFilename[128] = "UnitTest_Results_CPU.txt";
	char gpuResultsFilename[128] = "UnitTest_Results_GPU.txt";
	
	// check files exist
	if (access(cpuResultsFilename, F_OK) == -1) {
		std::cout << "\nError. File '" << cpuResultsFilename << "' not found. Refer to README.txt for usage instructions.\n";
		return 1;
	}
	if (access(gpuResultsFilename, F_OK) == -1) {
		std::cout << "\nError. File '" << gpuResultsFilename << "' not found. Refer to README.txt for usage instructions.\n";
		return 1;
	}

	cpuResults.open(cpuResultsFilename);
	gpuResults.open(gpuResultsFilename);
	if (cpuResults.fail() || gpuResults.fail()) {
		std::cout << "\nError. Could not open file.\n";
		return 1;
	} 

	std::cout << "\nAnalyzing test results...\n";	
	compareTestResults();
	generateTimesCsv();

	cpuResults.close();
	gpuResults.close();
	
	return 0;
}
