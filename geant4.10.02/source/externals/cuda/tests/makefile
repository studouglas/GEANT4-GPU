LIBRARY_PATH_GEANT='../../../../../geant4.10.02-install/lib/'

HEADER_PATH_CUDA='../include'
HEADER_PATH_CPU='../../../processes/hadronic/models/particle_hp/include'
HEADER_PATH_GEANT='../../../../../geant4.10.02-install/include/Geant4'

LINK_GEANT_LIBS_CPU=-lG4processes -lG4clhep -lG4global

all: GenerateTestResults AnalyzeTestResults

GenerateTestResults:
	g++ -std=c++11 -I $(HEADER_PATH_CPU) -I $(HEADER_PATH_GEANT) -L $(LIBRARY_PATH_GEANT) $(LINK_GEANT_LIBS_CPU) GenerateTestResults.cc -o GenerateTestResults

AnalyzeTestResults:
	g++ -std=c++11 AnalyzeTestResults.cc -o AnalyzeTestResults

clean:
	rm AnalyzeTestResults GenerateTestResults