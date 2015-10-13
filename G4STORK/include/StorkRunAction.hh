/*
StorkRunAction.hh

Created by:		Liam Russell
Date:			22-06-2011
Modified:		11-03-2013

Header for StorkRunAction class.

This class is responsible for most of the analysis of the results. At the
beginning of a run, all of the variables and containers are reinitialized.
Additionally, the primary generator's survivors and delayed lists are updated.
During a run, each event is tallied: running totals are kept of important
quantities and the survivors and delayed neutron sources are stored in buffers.
At the end of a run, the buffers are collated into single lists of neutron
sources.  Also, the results for quantities such as k_run and k_eff are
calculated and printed to the output stream (screen or file).

*/


#ifndef STORKRUNACTION_H
#define STORKRUNACTION_H

// Include G4-STORK headers
#include "StorkPrimaryGeneratorAction.hh"
#include "StorkContainers.hh"
#include "StorkEventData.hh"
#include "StorkParseInput.hh"
#include "StorkMatPropManager.hh"
#include "StorkInterpManager.hh"
#include "StorkProcessManager.hh"
#include "StorkHadronFissionProcess.hh"

// Include Geant4 headers
#include "G4Timer.hh"
#include "G4Run.hh"
#include "G4HCofThisEvent.hh"
#include "G4ThreeVector.hh"
#include "G4DataVector.hh"
#include "G4UserRunAction.hh"
#include "G4Neutron.hh"
#include "G4TransportationManager.hh"

// Include other headers
#include <typeinfo>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <dirent.h>

// Class forward declaration
class StorkPrimaryGeneratorAction;


class StorkRunAction : public G4UserRunAction
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkRunAction(StorkPrimaryGeneratorAction *genPtr,
                   const StorkParseInput *fIn);
        ~StorkRunAction();

        // Actions taken at the beginning and end of the current run
        void BeginOfRunAction(const G4Run *aRun);
        void EndOfRunAction(const G4Run *aRun);

        // Unpack and combine tally data from events
        void TallyEvent(const StorkEventData *eventData);
        void CollateNeutronSources();


        // Update the source distributions of StorkPrimaryGeneratorAction
        void UpdateSourceDistributions();
        // Update the value of the variable material-properties
        void UpdateWorldProperties(G4double *values);

        // Write the survivors and delayed neutrons of this run to a file
        void SaveSources(G4String fname, G4int runID, G4double runEnd);
        // Save fission data to a file
        G4bool WriteFissionData(G4String fname,G4int start);

        G4bool DirectoryExists( const char* pzPath );

        // Set functions
        void SetWorldDimensions(G4ThreeVector worldDims)
        {
            worldSize = worldDims;
        }
        void SaveFissionData(G4bool yesNo = true) { saveFissionData = yesNo; }

        // Get functions
        G4double* GetRunResults() { return runResults; };
        G4double GetShannonEntropy() { return shannonEntropy[0]; };
        MSHSiteVector GetCurrentFissionSites() { return CurrentfnSites; }
        DblVector GetCurrentFissionEnergy() { return CurrentfnEnergy; }


    private:
        // Private member functions

        // Calculate the Shannon entropy of the current run
        G4double CalcShannonEntropy(G4int ***sites, G4int total);
        // Calculate the meshing index of a 3D position
        Index3D SpatialIndex(G4ThreeVector position) const;
        //Calculate the neutron flux of the core
        G4double CalcNeutronFlux();
        // Erase function
        // This method is needed for the temperature changing since we only
        // want the temperature change to depend on fisson that occured this run
        void ResetCurrentFissionData();

    private:
        // Private member variables

        G4int kCalcType;

        G4double keff;              // k_eff for the current run
        G4double krun;              // k_run for the current run
        G4double runDuration;       // Duration of the run
        G4int numEvents;            // Number of events per run
        G4int totalFS;              // Total number of fission sites
        G4double avgLifetime;       // Average lifetime of lost neutrons
        G4int numNProduced;         // Number of neutrons produced
        G4int numDelayProd;         // Number of delayed neutrons produced
        G4int numNLost;             // Number of neutrons lost
        G4double runResults[8];     // Run results passed to run manager
        G4int numSites;             // Number of fission sites at the end of each run.
        G4int prevNumSites;         // Number of fission sites of the previous run.
        G4int runID;                //Current run ID.
        G4bool renormalize;         //Boolean flag if renormalization is used.
        G4int prevNumSurvivors;     //Previous number of survivors
        G4int saveInterval;         //Source and fission site/energy save interval.

        G4double tUnit;             // Time unit for output
        G4String rndmInitializer;   // File containing random seed (optional)
        G4bool saveFissionData;     // Flag for saving fission data
        G4bool RunThermalModel; // Flag to know if fission energy deposition is being kept track of
        G4int numFDCollectionRuns;  // Number of runs of collected fission data

        G4double shannonEntropy[2];     // Shannon entropy (fission sites,
                                        //                    survivors)
        G4double maxShannonEntropy;     // Maximum Shannon entropy
        G4int numSpatialSteps[3];       // Number of RS spatial steps in each
                                        // dimension (3D)
        G4ThreeVector worldSize;        // Size of world

        G4double neutronFlux;           //Neutron flux.
        G4double reactorPower;          //Reactor power.
        G4bool updatePrecursors;        //Flag to update precursors if a initial delayed file was provided.

        G4bool neutronFluxCalc;         //Flag to calculate neutron flux.

        G4double EnergyRange[2];        //Energy range of the neutron flux calc.
        G4int fn_index;                 //Current index for fission site /energy.
        G4int save_index;             //Save index for fission site/energy.
        G4ThreeVector Origin;           //Current set origin.
        G4double fluxCalcRegion[4];     //Flux region.
        G4String fluxCalcShape;         //Shape of the flux integration region.


        // Output stream (file or stdout)
        std::ostream *output;

        // Storage containers
        NeutronSources survivors;
        NeutronSources delayed;
        NeutronSources *survivorsBuffer;
        NeutronSources *delayedBuffer;
        G4int ***fSites;
        G4int ***sSites;
        DblVector fnEnergies;
        MSHSiteVector fnSites;
        MSHSiteVector CurrentfnSites;
        DblVector CurrentfnEnergy;
        G4int numRuns;
        G4int primariesPerRun;

        // Pointers to other classes
        StorkPrimaryGeneratorAction *genAction;
        const StorkParseInput *infile;

        // Timer for the run
        G4Timer runTimer;

        // Variable material-property variables
        WorldPropertyMap worldPropMap;
        G4int* nameLen;
        const StorkInterpManager *theMPInterpMan;
        G4double* variableProps;

        //Flag to save the run data (fission data and survivor data)
        G4bool saveRundata;


#ifdef G4TIMERA
        // Performance timer
        G4Timer runCalcTimer;
        G4double totalRunCalcTime;
#endif
};

#endif // STORKRUNACTION_H
