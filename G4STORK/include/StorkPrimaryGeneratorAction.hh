/*
StorkPrimaryGeneratorAction.hh

Created by:		Liam Russell
Date:			22-06-2011
Modified:		11-03-2013

Header for StorkPrimaryGeneratorAction class.

This class is responsible for creating the primary neutrons from either the
initial StorkPrimaryNeutronGenerator class or from the survivors. It also
renormalizes the neutron population at the start of a run by
duplicating/deleting survivors randomly from the survivors list.  It may also
load the initial distribution from a file.

*/

#ifndef STORKPRIMARYGENERATORACTION_H
#define STORKPRIMARYGENERATORACTION_H

// Include G4-STORK headers
#include "StorkContainers.hh"
#include "StorkPrimaryData.hh"
#include "StorkPrimaryNeutronGenerator.hh"
#include "StorkRunManager.hh"
#include "StorkParseInput.hh"
#include "StorkWorld.hh"
#include "StorkRunManager.hh"
#include "StorkProcessManager.hh"
#include "StorkMaterial.hh"
#include "StorkDelayedNeutron.hh"
#include "StorkSixVector.hh"

// Include Geant4 headers
#include "G4Timer.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Event.hh"
#include "globals.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ios.hh"
#include "G4String.hh"
#include "Randomize.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4Material.hh"
#include "G4DynamicParticle.hh"
#include "G4NeutronHPFissionData.hh"
#include "G4SystemOfUnits.hh"

// Include other headers
#include <fstream>
#include <stdio.h>
#include <math.h>
#include<vector>
#include <cstring>


class StorkRunManager;

class StorkPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
	public:
        // Public member fuctions

        // Constructor and destructor
        StorkPrimaryGeneratorAction(const StorkParseInput* infile,
                                    G4bool master);
		~StorkPrimaryGeneratorAction();

        // Update the survivor and delayed neutron distributions (end of run)
		void UpdateSourceDistributions(const NeutronSources *nSource,
                                       const NeutronSources *dnSource);
        // Initialize the primaries for the current run
        void InitializeRun();
        // Generate primary neutrons for an event
        void GeneratePrimaries(G4Event*);
        // Set the primaries for the current event
        void SetPrimaries(StorkPrimaryData *pData);
        // Get the primary data for an event
        StorkPrimaryData* GetPrimaryData(G4int eventNum);

        // Get the total number of primaries/delayed neutron primaries
        G4int GetNumPrimaries();
        G4int GetNumDNPrimaries();

        // Set the number of events per run
        void SetNumEvents(G4int numE) { numEvents = numE; };
        // Create the initial neutron source
        void InitialSource();

        // Disable renormalization of the number of survivors at the start of
        // each run
		void DisableNormalization() { normalize = false; }
    
        std::vector<G4int> GetPrecursors();
    
        void AddPrecursors(MSHSiteVector fSites, DblVector fEnergy);

	private:
        // Private member funtions

        // Load neutron primary source from file
        void LoadSource(G4String fname);

        // Add the delayed neutrons that are born in the current run and were produced from precursors to the
        // survivors
        void AddCurrentDelayed();

        // Renormalize the number of survivors to initial number of primaries
        void RenormalizeSurvivors(G4int numMissing);
        // Shuffle the order of the survivors
        void ShuffleSurvivors(G4int numShuffle, NeutronSources *origSurvivors);

        // Produce uniformly distributed positions in the world volume
        void UniformPosition(StorkNeutronData* input);
    

    

	private:

        G4bool initialSource;   // Flag denotes whether initial source has been
                                // created
        G4String sourceFile;    // Filename of source file

        G4bool sourcefileDelayed;			// Flag denotes whether an initial
    
        G4bool precursorDelayed;   //Flag to indicate if precursors are used.
										// delayed distribution is to be used
        G4String delayedSourceFile; 	// Filename for initial delayed source

        G4bool normalize;       // Flag denotes whether population is
                                // renormalized at the start of a new run
        G4bool instantDelayed;	// Flag denotes whether delayed neutrons are
								// produced instantaneously

        G4double initEnergy;    // Initial energy of source neutrons
        G4ThreeVector origin;
        G4String shape; // The shape to be used for the Uniform Distribution
        G4double runEnd;        // End time of the run
        G4int numPrimaries;     // Number of primary particles per event
        G4int realNumPrimaries; // True number of primaries if normalization is
                                //not used
        G4int numDNeutrons;     // The number of delayed neutrons produced.

		G4int numEvents;        // Number of events per run
		G4int numDNPrimaries;   // Number of delayed neutron primaries in the
		                        // current run
        G4double runDuration;   //Run duration in ns.

        G4int numEntries;      //Number of fission entries.

		G4ParticleDefinition *neutron;  // Particle definition for neutron

        NeutronSources survivors;       // Survivors of last run, source
                                        // distribution of current run
        NeutronList dNeutrons;          // Remaining delayed neutrons
        NeutronSources::iterator primaryIt;
        NeutronSources currPrimaries;   // Primaries for the current event

        StorkPrimaryNeutronGenerator nGen;          // Neutron generator

        StorkRunManager *runMan;           // Pointer to the run manager
        StorkPrimaryData *primaryData;

        G4bool uniformDis;		// Uniform source distribution flag
        std::vector<G4int> Precursors; //Precursor population
    
    
        StorkDelayedNeutron* delayedNeutronGenerator; //The delayed neutron generator, uses and tracks precursors.

        StorkHadronFissionProcess* theFissionProcess;

        SiteVector fSites;
        DblVector fnEnergy;
        G4Navigator *theNav;

        G4double adjustmentFactor;



#ifdef G4TIMEPG
        G4Timer genTimer;
        G4double totalGenTime;
#endif
};

#endif // STORKPRIMARYGENERATORACTION_H
