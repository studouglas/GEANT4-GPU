/*
StorkParseInput.hh

Created by:		Liam Russell
Date:			13-06-2011
Modified:		11-03-2013

Header file for StorkParseInput class.

StorkParseInput class is designed to read the input file and store all the
information so that it can be used by other classes later (through Get
functions).  The class also prints a formatted table of the input to a given
output stream.

*/

#ifndef STORKPARSEINPUT_H
#define STORKPARSEINPUT_H

#include <fstream>
#include <dirent.h>
#include "Randomize.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"
#include "BZR_Version_Info.hh"
#include "G4String.hh"
#include "StorkInterpVector.hh"
#include <map>
#include "StorkMatPropManager.hh"
#include "StorkContainers.hh"
#include "StorkMatPropChange.hh"
#include "StorkInterpManager.hh"
#include "G4SystemOfUnits.hh"
#include "StorkSixVector.hh"



class StorkParseInput
{
	public:
        // Public member functions

        // Constructor and destructor
		StorkParseInput(G4bool master=true);
		~StorkParseInput();

        // Read and print the parsed input file
		G4bool ReadInputFile(G4String filename);
		G4int ConvertSide(G4String side);
		void PrintInput(std::ostream *output) const;	// Print input values to stream
		G4bool FinalizeInputs();

        G4bool DirectoryExists( const char* pzPath );

		// Add world with default values to ParseInput
		void AddWorldType(G4String name, StorkMatPropChangeVector defaults);

        // Set the version of Geant4
		void SetG4VersionString(G4String ver) { g4Version = ver; }

		// Get/Set functions
		void SetRequiredInputs(G4String worldNam, G4int numRuns, G4int numEvents, G4int numPrimariesPerEvent, G4double runDur);


		G4String GetWorld() const { return worldName; }
		void SetWorld(G4String worldNam);

		G4int GetReactorMaterial() const { return reactorMat; }
		void SetReactorMaterial(G4int reactorMaterial);

		G4ThreeVector GetWorldDimensions() const;
		void SetWorldDimensions(G4double dim) { theWorldProps[MatPropPair(all,dimension)]=dim; }

		G4int GetKCalcType() const { return kCalcType; }
		void SetKCalcType(G4int KCalcType) { kCalcType = KCalcType; }

		G4String GetCSDirName() const { return csDirName; }
		void SetCSDirName(G4String CSDirName) { csDirName=CSDirName; }

		G4String GetFSDirName() const { return fsDirName; }
		void SetFSDirName(G4String FSDirName) { fsDirName=FSDirName; }

		G4double GetInitialEnergy() const { return initEnergy; }
		void SetInitialEnergy(G4double initialEnergy) { initEnergy=initialEnergy; }

		G4ThreeVector GetInitialSourcePos() const { return initialSourcePos; }
		void SetInitialSourcePos(G4ThreeVector initSourcePosition) { initialSourcePos=initSourcePosition; }

		G4String GetUniformDistributionShape() const { return uniDisShape; }
		void SetUniformDistributionShape(G4String uniformDisShape) { uniDisShape=uniformDisShape; }


		StorkSixVector<G4double> GetUniformDistDim() const
		{
            return uniDisDim;
        }
		void SetUniformDistDim(StorkSixVector<G4double> uniformDisDim)
		{
            uniDisDim=uniformDisDim;
        }

		G4int GetNumberOfRuns() const { return numberOfRuns; }
		void SetNumberOfRuns(G4int numOfRuns) { numberOfRuns=numOfRuns; }

		G4int GetNumberOfEvents() const { return numberOfEvents; }
		void SetNumberOfEvents(G4int numOfEvents) { numberOfEvents=numOfEvents; }

		G4int GetNumberOfPrimariesPerEvent() const
			{ return numberOfPrimariesPerEvent; }
        void SetNumberOfPrimariesPerEvent(G4int numPrimaries) { numberOfPrimariesPerEvent=numPrimaries; }

		G4double GetRunDuration() const { return runDuration; }
		void SetRunDuration(G4double runDur) { runDuration=runDur; }

		const G4int* GetMeshSteps() const { return numSteps; }
		void SetMeshSteps(G4int numberSteps[]) { numSteps[0]=numberSteps[0]; numSteps[1]=numberSteps[1]; numSteps[2]=numberSteps[2];}

		G4int GetNumberOfConvergenceRuns() const { return convRuns; }
		void SetNumberOfConvergenceRuns(G4int convergRuns) { convRuns=convergRuns; }

		G4double GetConvergenceLimit() const { return limitSE; }
		void SetConvergenceLimit(G4double limSE) { limitSE=limSE; }


		// Log/Initial Source Files and Options
		G4bool OverrideInitRandomSeed() const { return overrideInitRandomSeed; }
		G4long GetRandomSeed() const { return randomSeed; }
		void SetRandomSeed(G4long randSeed);

		G4bool SaveResultsToFile() const { return logData; }
		G4String GetResultsFile() const { return logFile; }
		void SetResultsFile(G4String logFil);

		G4int SaveSourcesInterval() const { return saveSources; }
		void SetSaveSourcesInterval(G4int sourceInterval) { saveSources=sourceInterval; }

		G4String GetSourceFile() const { return sourceFile; }
		void SetSourceFile(G4String sourceFil) { sourceFile=sourceFil; }

		G4bool LoadInitialSources() const { return loadSources; }

		G4String GetInitialSourceFile() const { return initialSourceFile; }
		void SetInitialSourceFile(G4String SourceFile);

		G4bool GetPrecursorDelayed() const { return precursorDelayed; }
		G4String GetInitialDelayedFile() const { return initialfissionDataFile; }
		void SetInitialDelayedFile(G4String DelayedFile);

		G4bool SaveFissionData() const { return saveFissionData; }
		G4String GetFissionDataFile() const { return fissionDataFile; }
		void SetFissionDataFile(G4String fissionFile);

        G4bool SaveTemperature() const { return saveTempData; }
        G4String GetTemperatureDataFile() const { return tempFileName; }
        void SetTemperatureDataFile(G4String temperatureFile);

		// Interpolation and world property changes
		G4bool GetInterpStartCond() const {return interpStart;}
		void SetInterpStartCond(G4bool interpStartCond) { interpStart=interpStartCond; }

        const StorkInterpManager* GetNSInterpolationManager() const
        {
        	return &theMPInterpMan;
		}
		void SetNSInterpolationManager(G4String mat, G4String prop, G4String interpDataFile);

		StorkMatPropChangeVector GetIntialWorldProperties() const
		{
			return initialWorldProps;
		}
		void SetIntialWorldProperties(G4String mat, G4String prop, G4double val);

		// Simulation options
		G4bool GetInstantDelayed() const{return instantDelayed;}
		void SetInstantDelayed(G4bool instantDelay) { instantDelayed=instantDelay; }

		void GetPeriodicBCVector(std::vector<G4int>* &perBC) const{perBC = periodicBC;}
		void SetPeriodicBCVector(std::vector<G4int>* pBC) { (*periodicBC)=(*pBC); }

		void GetReflectBCVector(std::vector<G4int>* &refBC) const{refBC = reflectBC;}
		void SetReflectBCVector(std::vector<G4int>* rBC) { (*reflectBC)=(*rBC); }

		G4bool GetRenormalizeAfterRun() const {return normalize;}
		void SetRenormalizeAfterRun(G4bool renormalize) { normalize=renormalize; }

		G4bool GetUniformDistribution() const {return uniformDis;}
		void SetUniformDistribution(G4bool uniformSourceDis) { uniformDis=uniformSourceDis; }

        G4bool GetRunThermalModel() const { return RunThermalModel; }
        void SetRunThermalModel(G4bool tracking) {RunThermalModel=tracking;}

		G4bool GetUniformDistWithDim() const {return uniformDisWithDim;}
		void SetUniformDistWithDim(G4bool uniDisWithDim) { uniformDisWithDim=uniDisWithDim; }

        G4bool GetInterp() const { return interp; }
        void SetInterp(G4bool yesno) {interp=yesno;}

        G4bool GetNeutronFluxCalc() const { return neutronFluxCalc;}
        void SetNeutronFluxCalc(G4bool neutronFlux) { neutronFluxCalc = neutronFlux;}

        G4bool GetSourceFileDelayed() const {return sourcefileDelayed;}

        const G4double* GetEnergyRange() const { return energyRange;}

        const G4double* GetFluxRegion() const {return fluxCalcRegion;}

        const G4String GetFluxCalcShape() const {return fluxCalcShape;}

        G4ThreeVector GetOrigin() const {return Origin;}
        void SetOrigin(G4ThreeVector orig) { Origin = orig;}

        // Reactor Power Output
        G4double GetReactorPower() const { return reactorPower; }
        void SetReactorPower(G4double power) {reactorPower = power;}

        //Heat Transfer
        G4double GetHeatTransferCoefficient() const {return htc;}

        G4double GetAmbientTemperature() const {return T_infty;}

        G4double GetBaselineFissionRate() const {return baselineFissionRate;}

        G4double GetFissionToEnergyCoefficient() const {return FissionToEnergy;}


		// Logging output stream
		std::ostream* GetLogOutputStream() const { return logStream; }



	private:
        // Private member functions

		// Check the input for errors
		G4bool ErrorChecking();

		// Helper function for setting initial world properties
		void SetInitialPropertiesVector();

        // Print the material-property pair (as a string) with units
		G4String PrintMatPropPair(MatPropPair mpPair) const;

	private:

		// Flag for master process
		G4bool isMaster;

		// GEANT4 version string
		G4String g4Version;

        // Flag to keep track of fission energy deposition
        G4bool RunThermalModel;
        G4bool interp;
        // Reactor Power
        G4double reactorPower;

		// World choice and seed for random number generator
		InitialPropertyMap theWorldProps;

		// World properties
		G4String worldName;
		G4int reactorMat;
		G4int kCalcType;
		G4String csDirName;
		G4String fsDirName;
		G4long randomSeed;
		G4bool overrideInitRandomSeed;

		// World options
		G4bool uniformDis;
		G4String uniDisShape;
		G4bool uniformDisWithDim;
		StorkSixVector<G4double> uniDisDim;
		G4bool interpStart;
		G4bool instantDelayed;
		std::vector<G4int> *reflectBC;
        std::vector<G4int> *periodicBC;
		G4bool normalize;
        G4bool neutronFluxCalc;

        //Delayed neutron options.
        G4int theDelayedOption;
        G4bool sourcefileDelayed;
        G4bool precursorDelayed;

		// Number of runs, events and primaries per event
		G4int numberOfRuns;
		G4int numberOfEvents;
		G4int numberOfPrimariesPerEvent;
		G4double runDuration;

        //Energy range and volume name for neutron flux calculation.
        G4double energyRange[2];
        G4String fluxCalcShape;
        G4double fluxCalcRegion[4];

        //Origin
        G4ThreeVector Origin;

		// Initial neutron energy (default source) in MeV
		G4double initEnergy;

		//Heat Transfer Coefficient
		G4double htc;

		//Ambient Temperature.
		G4double T_infty;

        //Baseline flux
        G4double baselineFissionRate;

        //Conversion coefficient from flux to power.
        G4double FissionToEnergy;

		// Initial neutron start position
		G4ThreeVector initialSourcePos;

		// Shannon entropy mesh and convergence limits
		G4int numSteps[3];
		G4double limitSE;
		G4int convRuns;

		// Input and output data flags and files
		std::ofstream *logStream;		//  Logging file (results)
		G4bool logData;
		G4String logFile;
		G4int saveSources;				// Save sources to file
		G4String sourceFile;
		G4bool loadSources;				// Load initial sources from file
		G4String initialSourceFile;
		G4bool saveFissionData;			// Save fission data to file
		G4String fissionDataFile;
		G4String initialfissionDataFile;    //Initial loaded fission data file
        G4bool saveTempData;
        G4String tempFileName;
        G4bool initialFissionData;


        // Vector of data profile identifiers and file names
        std::vector<std::pair<G4String,G4String> > theDataProfiles;


		// Path to neutron data
		G4String datapath;

        // Manager for all property-material interpolation vectors
        StorkInterpManager theMPInterpMan;
        StorkMatPropChangeVector initialWorldProps;
        StorkMatPropManager *theMPMan;

        // Map for user-added worlds
        std::map<G4String,StorkMatPropChangeVector> userWorlds;
};

#endif // STORKPARSEINPUT_H
