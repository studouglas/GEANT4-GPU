/*
StorkTimeStepLimiter.cc

Created by:		Liam Russell
Date:			22-06-2011
Modified:		11-03-2013

Source code file for StorkTimeStepLimiter class.

*/


// Include header file
#include "StorkTimeStepLimiter.hh"


// Constructor
// Calls the G4StepLimiter constructor with the proper name
StorkTimeStepLimiter::StorkTimeStepLimiter(const G4String &aName)
: G4StepLimiter(aName)
{
    // Get a pointer to the run manager
    runMan = dynamic_cast<StorkRunManager*>(G4RunManager::GetRunManager());
}


// PostStepGetPhysicalInteractionLength()
// Returns the total distance the neutron can travel in the time until the
// current run ends at the current momentum of the neutron.
G4double StorkTimeStepLimiter::PostStepGetPhysicalInteractionLength(
                                            const G4Track &aTrack,
                                            G4double, // previousStepSize
                                            G4ForceCondition *condition)
{
    // Set condition to "Not Forced"
    *condition = NotForced;

    // Determine the proposed step
    G4double proposedStep = aTrack.GetVelocity() * (runMan->GetRunEnd() -
                                                     aTrack.GetGlobalTime());

    // Make sure the step length is not negaitve
    if(proposedStep < 0.0) proposedStep = 0.0;

    return proposedStep;
}
