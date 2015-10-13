/*
StorkUserBCStepLimiter.hh

Created by:		Wesley Ford
Date:			22-06-2012
Modified:		11-03-2013

Header file for the StorkUserBCStepLimiter class.

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
#include "StorkPeriodicBCTransform.hh"
#include "StorkReflectBCTransform.hh"

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


class StorkUserBCStepLimiter : public G4StepLimiter
{
    public:
		// Public member functions

        // Constructor and destructor
        StorkUserBCStepLimiter(std::vector<G4int>* PeriodicBC, std::vector<G4int>* ReflectBC,
                           const G4String& processName = "StorkUserBCStepLimiter",
                            G4String worldPhysName="worldPhysical" );

        virtual ~StorkUserBCStepLimiter()
        {
            if(BCTransform)
            {
                for(G4int i=0; i<6; i++)
                {
                    if(BCTransform[i])
                        delete BCTransform[i];
                }
            }
        }

        virtual G4double PostStepGetPhysicalInteractionLength(
                                        const G4Track& aTrack,
                                        G4double previousStepSize,
                                        G4ForceCondition *condition);

        virtual G4bool IsApplicable(const G4ParticleDefinition &particle);

        virtual G4VParticleChange* PostStepDoIt(const G4Track &aTrack,
												const G4Step &aStep);

        G4ThreeVector GetNormal(int side);


    private:
		//Private member variables

            G4String worldName;                 // Name of world physical volume
            StorkProcessManager *nsProcMan;     // Pointer to process manager
            StorkPrimaryNeutronInfo *pnInfo;
            StorkTrackInfo *trackInfo;
            StorkBCTransform* BCTransform[6];
            std::vector<G4int> zeroSides;
};

#endif // STORKPERIODICBCLIMITER_H
