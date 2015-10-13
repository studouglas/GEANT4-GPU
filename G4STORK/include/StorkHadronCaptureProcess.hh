/*
StorkHadronCaptureProcess.hh

Created by:		Liam Russell
Date:			29-02-2012
Modified:       11-03-2013

Definition of StorkHadronCaptureProcess class.

This class is derived from G4HadronCaptureProcess to modify the StartTracking()
function and to add a get function for theNumberOfInteractionLenghtLeft
member variable.

*/


#ifndef STORKHADRONCAPTUREPROCESS_H
#define STORKHADRONCAPTUREPROCESS_H

// Include header files
#include "G4HadronCaptureProcess.hh"
#include "StorkHadronicProcess.hh"

class StorkProcessManager;

class StorkHadronCaptureProcess : public G4HadronCaptureProcess
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkHadronCaptureProcess(const G4String& processName =
                                  "StorkHadronCapture")
        :G4HadronCaptureProcess(processName)
        {
//            SetVerboseLevel(2);
        }
        virtual ~StorkHadronCaptureProcess() {;}

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


#endif // STORKHADRONCAPTUREPROCESS_H
