/*
StorkRunManager.hh

Created by:		Liam Russell
Date:			22-06-2011
Modified:		09-07-2012

Header for StorkRunManager class.

This class is the single processor run manager.  It is responsible for starting
the runs, generating the events, and passing information between the event,
run and primary generator actions.  It also checks to make sure the source
distribtion converges within the given limit (default is 3%).

*/

#ifndef STORKRUNMANAGER_H
#define STORKRUNMANAGER_H

// Include G4-STORK header files
#include "StorkPrimaryGeneratorAction.hh"
#include "StorkRunAction.hh"
#include "StorkEventAction.hh"
#include "StorkHeatTransfer.hh"
#include "StorkWorld.hh"
#include "StorkInterpVector.hh"
#include "StorkParseInput.hh"
#include "StorkMatPropManager.hh"
#include "StorkContainers.hh"
#include "StorkInterpManager.hh"
#include "StorkMatPropChange.hh"

// Include Geant4 header files
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4Event.hh"
#include "G4DataVector.hh"
#include "G4HCofThisEvent.hh"
#include "G4EventManager.hh"

// Include other header files
#include <fstream>
#include <sstream>
#include <list>
#include <cmath>


// Class forward declarations
class StorkPrimaryGeneratorAction;
class StorkRunAction;
class StorkEventAction;
class StorkWorld;


class StorkRunManager: public G4RunManager
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkRunManager(const StorkParseInput* infile);
        StorkRunManager();
        virtual ~StorkRunManager();

        void InitializeRunData(const StorkParseInput* infile);
        void InitializeRunData(G4double runDur, G4int numberRuns, G4int numSaveInterval, G4String saveFileName, G4bool interpStartCondition,
                                const StorkInterpManager* theMPInterpManager, G4double convergenceLim, G4int numConvRuns, G4bool saveFissionDataCond,
                                G4String fissionDataFile, std::ostream *logOutput, G4bool temperatureTracking,G4double nuclearReactorPower,
                                G4bool saveTemperature, G4String temperatureDataFile);

        // Starts the simulation
        virtual void BeamOn(G4int n_event, const char* macroFile=0,
                            G4int n_select=-1);
        // Process all events for each run
        virtual void DoEventLoop(G4int n_event, const char* macroFile=0,
                                 G4int n_select=-1);
        // Initialize the simulation for the current run
        virtual void RunInitialization();

        // Set the total number of runs
        void SetNumRuns(G4int num) { numRuns = num; }


        // Print simulation averaged results to the output stream
        void OutputResults();

        // Save the source distribution of the current run to a file
        void SaveSourceDistribution(G4String fname);

        //save the fission distribution of the current run to a file.
        void SaveFissionDistribution(G4String name);

        // Get functions
        G4double GetRunDuration() { return runDuration; }
        G4double GetRunStart() { return runStart; }
        G4double GetRunEnd() { return runEnd; }
        G4bool GetSourceConvergence() { return sourceConverged; }
        StorkWorld* GetWorld() const {return worldPointerCD;}


    protected:
        // Protected member functions

        // Add the results from the current run to the full and averaged results
		void TallyRunResults();

		// Check whether the source has converged
		G4bool UpdateCheckSourceConvergence();

		// Rebuild the world with new properties
		void UpdateWorld(StorkMatPropChangeVector theChanges);

		// Get the current value of a variable material property
		G4double GetWorldProperty(MatPropPair matProp);

        // Initialize variables at the start of the simulation
		virtual void InitializeVar(G4int n_event);

		// Average the results from all runs with source convergence
		void AverageRunResults();

	protected:
        // Protected member variables
        StorkHeatTransfer* heatTransfer;
        G4int numRuns;				// Total number of runs for the simulation

       // G4double heatTransferCoefficient;
        //G4double T_infinity;

        // Source distribution output
        G4int saveInterval;			// Interval at which source distribution is
									//    saved to file
        G4String saveFile;			// Base source distribution file name
        G4int numRunOutputWidth;	// Width of characters needed to write max
									// 	  runs

		// Fission distribution output
		G4bool saveFissionData;		// Flag to signal saving fission data
		G4String fissionFile;		// Output file for fission data

        // Run timing
        G4double runDuration;		// Duration of a run (ns)
        G4double runStart;			// Start time of the current run (ns)
        G4double runEnd;			// End time of the current run (ns)

		// Convergence variables
        G4bool sourceConverged;			// Flag indicates source has converged
        G4double convergenceLimit;		// Required convergence precision
        G4double *seSelect;				// Current selection of Shannon entropy
										// being tested for convergence

        G4int totalConv;                // Total number of SE values needed for
                                        // convergence
        G4int convergeStop;
		G4int nConv;					// Run when convergence achieved


        // Material temperature output
        G4bool saveMatTemp; // Flag to output the temperature of all materails
        G4String matTempFile; // Output file for material temperature
        G4bool RunThermalModel; // Flag to track fission and
                                                // change energy acordingly
        G4bool interp;
        G4Navigator* theNavigator; // Navigator
        G4double reactorPower; // Power output of the reactor
        G4bool fMap;                //Fission Map boolean flag.
        MSHSiteVector fnSites;      // Temporary storage for fission sites

        // Class pointers
        StorkPrimaryGeneratorAction *genAction;
        StorkRunAction *runAction;
        StorkEventAction *eventAction;
        StorkWorld *worldPointerCD;
        static G4EventManager *EventMan;
        const StorkInterpManager* theMPInterpMan;

        // Run action data
        G4double* runData[8];
        G4double avgRunData[9];

        // Interpolation of world parameters
        G4bool interpStartCond;		// Flag to start interpolation at first run
        G4bool interpStarted;		// Flag indicates interpolation started
        G4int runInterpStarted;		// Run number at start of interpolation
        G4double timeOffset;		// Time at start of interpolation
        G4double *propValues;		// Current values of variable world
                                    // properties

        // Output stream (file or stdout)
        std::ostream *output;
        G4double frac;

    private:



};

#endif // STORKRUNMANAGER_H
