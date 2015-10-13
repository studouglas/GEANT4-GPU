/*
StorkPeriodicBCStepLimiter.hh

Created by:		Wesley Ford
Date:			22-06-2012
Modified:		11-03-2013

Header file for the StorkPeriodicBCStepLimiter class.

The boundary step limiter implements inifinite boundary conditions assuming the
world volume is symmetric about the origin.

This class is roughly based off of (and inherits from) the G4StepLimiter class.

*/


#ifndef STORKPERIODICBCLIMITER_H
#define STORKPERIODICBCLIMITER_H

// Include G4-STORK headers
#include "StorkProcessManager.hh"
#include "StorkPrimaryNeutronInfo.hh"
#include "StorkTrackInfo.hh"

// Include Geant4 headers
#include "G4StepLimiter.hh"
#include "G4Track.hh"
#include "G4Box.hh"
#include "G4VParticleChange.hh"
//#include "G4Plane.hh"
#include "G4VPhysicalVolume.hh"
#include "G4String.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"


class StorkPeriodicBCStepLimiter : public G4StepLimiter
{
    public:
		// Public member functions

        // Constructor and destructor
        StorkPeriodicBCStepLimiter(const G4String& processName =
                                   "StorkPeriodicBCStepLimiter",
                                   G4String worldPhysName="worldPhysical" );
        virtual ~StorkPeriodicBCStepLimiter() {;}

        virtual G4double PostStepGetPhysicalInteractionLength(
                                        const G4Track& aTrack,
                                        G4double previousStepSize,
                                        G4ForceCondition *condition);

        virtual G4bool IsApplicable(const G4ParticleDefinition &particle);

        virtual G4VParticleChange* PostStepDoIt(const G4Track &aTrack,
												const G4Step &aStep);


    private:
		//Private member variables

            G4String worldName;                 // Name of world physical volume
            StorkProcessManager *nsProcMan;     // Pointer to process manager
            StorkPrimaryNeutronInfo *pnInfo;
            StorkTrackInfo *trackInfo;


};

#endif // STORKPERIODICBCLIMITER_H
