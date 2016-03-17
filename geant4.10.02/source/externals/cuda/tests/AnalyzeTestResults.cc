#include <iostream>
#include <string>
#include <algorithm>
#include <stdbool.h>
#include <unistd.h>
#include <fstream>
#include <stdio.h>

std::ifstream cpuResults;
std::ifstream gpuResults;

const char variableId = '@';
const char methodId = '#';
const char nEntriesId = '!';

void compareTestResults() {
	int testsPassed = 0;
	int testsFailed = 0;

	std::string cpuLine;
	std::string gpuLine;

	std::string currentMethod = "";
	std::string currentCaseNum = "";
	std::string currentVariableName = "";
	
	while (std::getline(cpuResults, cpuLine) && std::getline(gpuResults, gpuLine)) {
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
		else if (cpuLine.at(0) != nEntriesId) {
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
	std::cout << (double)(testsPassed*100.0/(double)(testsPassed+testsFailed)) << "\% passed\n"; 
	std::cout << testsPassed << " tests passed out of " << testsFailed+testsPassed << "\n";
	std::cout << "-------------------------------\n\n";
}

std::string replaceCommas(std::string str) {
	std::string res = "";
	for (int i = 0; i < str.length(); i++) {
		if (str.at(i) == ',') {
			res.append(".");
		} else {
			res.append(std::string(1, str.at(i)));
		}
	}
	return res;
}
void generateTimesCsv() {
	std::ifstream cpuTimes("UnitTest_Times_CPU.txt");
	std::ifstream gpuTimes("UnitTest_Times_CPU.txt");
	std::ofstream timesOutput("UnitTest_Times.csv");
	
	std::string cpuLine = "";
	std::string gpuLine = "";
	
	std::string currentMethod = "";
	std::string currentCaseNum = "";
	std::string currentVariableName = "";
	std::string nEntriesPerCase[128]; // won't have more than 128 cases !

	timesOutput << "Method Signature,Case Number,nEntries,Input,CPU Time,GPU Time\n";
	while (std::getline(cpuTimes, cpuLine) && std::getline(gpuTimes, gpuLine)) {
		if (cpuLine.at(0) == methodId) {
			int caseNumSeparatorIndex = cpuLine.find_last_of("_");
			currentCaseNum = cpuLine.substr(caseNumSeparatorIndex+1, cpuLine.length());
			currentMethod = cpuLine.substr(1, caseNumSeparatorIndex-1);
			currentMethod = replaceCommas(currentMethod);
			currentVariableName = "";
		} else if (cpuLine.at(0) == variableId) {
			currentVariableName = cpuLine.substr(1);
		} else if (cpuLine.at(0) == nEntriesId) {
			std::string caseNum = cpuLine.substr(1, cpuLine.find_last_of("!")-1);
			std::string nEntries = cpuLine.substr(cpuLine.find_last_of("!") + 1, cpuLine.length());
			nEntriesPerCase[stoi(caseNum)] = nEntries;
		} else {
			timesOutput << currentMethod << "," << currentCaseNum << "," << nEntriesPerCase[stoi(currentCaseNum)] << "," << currentVariableName << "," << cpuLine << "," << gpuLine << "\n";
		}
	}

	cpuTimes.close();
	gpuTimes.close();
	timesOutput.close();
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
