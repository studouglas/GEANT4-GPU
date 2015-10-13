/*
StorkTimeStepLimiter.hh

Created by:		Liam Russell
Date:			29-02-2012
Modified:		2011

Header file for the StorkTimeStepLimiter class.

The StorkTimeStepLimiter class is used to ensure particles cannot move past the
end of the current run (time limited). The proposed step is based on the current
velocity of the neutron and the time left in the run.  This process is included
so that the neutrons stop at the end of the interval without undergoing an
interaction, and thus resetting the 'theNumberOfInteractionLengthLeft' counter.

This class is roughly based off of (and inherits from) the G4StepLimiter class.

*/

#ifndef NSSTEPLIMITER_H
#define NSSTEPLIMITER_H

#include "G4StepLimiter.hh"
#include "G4Track.hh"
#include "StorkRunManager.hh"


class StorkTimeStepLimiter : public G4StepLimiter
{
    public:
        // Public member functions

        // Constructors and destructors
        StorkTimeStepLimiter(const G4String& processName =
                                                    "StorkTimeStepLimiter" );
        virtual ~StorkTimeStepLimiter() {;}

        // Get the max distance neutron can travel during the current run
        virtual G4double PostStepGetPhysicalInteractionLength(
                                        const G4Track& aTrack,
                                        G4double previousStepSize,
                                        G4ForceCondition *condition);

    private:
        // Private member variables

        StorkRunManager *runMan;
};

#endif // NSSTEPLIMITER_H
