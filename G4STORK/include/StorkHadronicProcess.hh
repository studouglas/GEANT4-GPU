/*
StorkHadronicProcess.hh

Created by:		Liam Russell
Date:			29-02-2012
Modified:       11-03-2012

This header contains a friend function for all the hadronic/neutron processes:
    1. StorkHadronFissionProcess
    2. StorkHadronElasticProcess
    3. StorkHadronCaptureProcess
    4. StorkNeutronInelasticProcess

The friend function, StartTrackingHadronicProcess(), allows G4-STORK to set the
initial n-lambda (number of interaction lengths left) values for each process.
This is implemented as a friend function since the processes have different base
classes.

*/


#ifndef NSHADRONICPROCESS_H
#define NSHADRONICPROCESS_H

// Include header files
#include "StorkPrimaryNeutronInfo.hh"
#include "StorkTrackInfo.hh"
#include "G4PrimaryParticle.hh"
#include "G4Track.hh"


// StartTrackingHadronicProcess()
// Sets the initial currentInteractionLength and
// theNumberOfInteractionLengthLeft (eta).  Eta is set from the primary
// neutron data if there is any, or it is set to the default, -1.0.
inline void StartTrackingHadronicProcess(G4Track *aTrack, G4double &n_lambda,
										 G4int procIndex)
{
    // If this is not the first step of a track, return immediately
    if(aTrack->GetCurrentStepNumber() > 0) return;

    // Get the primary particle user information
    G4PrimaryParticle *aPrimary = aTrack->GetDynamicParticle()->
                                          GetPrimaryParticle();

	// Initialize pointers
	StorkTrackInfo *trackInfo = NULL;
	StorkPrimaryNeutronInfo *pnInfo = NULL;

    // Check that the primary neutron exists. If so, set the primary info.
    if(aPrimary)
    {
        pnInfo = dynamic_cast<StorkPrimaryNeutronInfo*>(
                                                aPrimary->GetUserInformation());
    }
    // Else check if the track has a track info object
    else if((trackInfo = dynamic_cast<StorkTrackInfo*>(
                                                aTrack->GetUserInformation())))
    {
    	pnInfo = trackInfo->GetStorkPrimaryNeutronInfo();
    }

	// Check that the primary neutron info exists
	if(pnInfo)
	{
		// Set the number of interaction lengths left
		n_lambda = pnInfo->GetEta(procIndex);
	}
}

#endif // NSHADRONICPROCESS_H
