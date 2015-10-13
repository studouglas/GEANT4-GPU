#ifndef STORKZEROBCSTEPLIMITER_HH
#define STORKZEROBCSTEPLIMITER_HH

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

class StorkZeroBCStepLimiter : public G4StepLimiter
{
    public:

        StorkZeroBCStepLimiter(std::vector<G4int>* PeriodicBC, std::vector<G4int>* ReflectBC,
                           const G4String& processName = "StorkZeroBCStepLimiter",
                            G4String worldPhysName="worldPhysical");

        virtual ~StorkZeroBCStepLimiter()
        {};

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
            std::vector<G4int> zeroSides;
};

#endif // STORKZEROBCSTEPLIMITER_HH
