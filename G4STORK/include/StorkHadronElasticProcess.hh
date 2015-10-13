/*
StorkHadronElasticProcess.hh

Created by:		Liam Russell
Date:			29-02-2012
Modified:

Header file for StorkHadronElasticProcess class.

This class is derived from G4HadronElasticProcess to modify the StartTracking()
function and to add a get function for theNumberOfInteractionLenghtLeft
member variable.

*/


#ifndef STORKHADRONELASTICPROCESS_H
#define STORKHADRONELASTICPROCESS_H

// Include header files
#include "G4HadronElasticProcess.hh"
#include "StorkHadronicProcess.hh"

class StorkProcessManager;

class StorkHadronElasticProcess : public G4HadronElasticProcess
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkHadronElasticProcess(const G4String& processName = "StorkHadronElastic")
        :G4HadronElasticProcess(processName)
        {
//            SetVerboseLevel(2);
        }
        virtual ~StorkHadronElasticProcess() {;}

        // Set the number of interaction lengths left from previous run
        virtual void StartTracking(G4Track *aTrack)
        {
			G4VProcess::StartTracking(aTrack);
			StartTrackingHadronicProcess(aTrack,
										 theNumberOfInteractionLengthLeft,
										 procIndex);
		}

        // Get the current number of interaction lengths left for the process
        virtual G4double GetNumberOfInteractionLengthLeft(G4double
                                                          previousStepSize)
        {
            SubtractNumberOfInteractionLengthLeft(previousStepSize);
            return theNumberOfInteractionLengthLeft;
        }

        void SetProcessIndex(G4int ind) { procIndex = ind; }

        // Declare friend function
        friend void StartTrackingHadronicProcess(G4Track *aTrack,
												 G4double &n_lambda,
												 G4int procIndex);

	private:
        // Private member variables

		G4int procIndex;    // Index of physics process (StorkProcessManager)
};


#endif // STORKHADRONELASTICPROCESS_H
