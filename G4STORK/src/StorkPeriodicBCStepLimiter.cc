/*
StorkPeriodicBCStepLimiter.cc

Created by:		Wesley Ford
Date:			22-06-2012
Modified:		17-02-2012

Source code file for StorkPeriodicBCStepLimiter class.

*/


// Include header file
#include "StorkPeriodicBCStepLimiter.hh"


// Constructor()
// Calls the G4StepLimiter constructor with the proper name
StorkPeriodicBCStepLimiter::StorkPeriodicBCStepLimiter(const G4String &aName,
                                                       G4String worldPhysName)
: G4StepLimiter(aName)
{
	worldName=worldPhysName;
	nsProcMan = NULL;
	pnInfo = NULL;
}


// PostStepGetPhysicalInteractionLength()
// Returns a step size of DBL_MIN (very small) if the neutron has entered the
// world (outside) volume.  Otherwise returns DBL_MAX (very big).
G4double StorkPeriodicBCStepLimiter::PostStepGetPhysicalInteractionLength(
                                            const G4Track &aTrack,
                                            G4double,
                                            G4ForceCondition *condition)
{
    // Set condition to "Not Forced"
    *condition = NotForced;
    G4double dp = 0.;

    G4String physVolName = aTrack.GetVolume()->GetName();
    G4VSolid *solidVol = aTrack.GetVolume()->GetLogicalVolume()->GetSolid();

    // Check that the neutron is in the world volume
    if(physVolName!=worldName)
    {
        return DBL_MAX;
    }
    // Check that the neutron is ENTERING the world volume (rather than leaving)
    else
    {
        G4ThreeVector p = aTrack.GetPosition();
        G4ThreeVector v = aTrack.GetMomentumDirection();
        G4ThreeVector n = solidVol->SurfaceNormal(p);

		// Compute the dot product of direction and surface normal
		dp = v.dot(n);

        if(dp>0.)
        {
            return DBL_MIN;
        }
        else
        {
            return DBL_MAX;
        }
     }
}


// IsApplicable()
// Check that the particle is a neutron, otherwise not applicable.
G4bool
StorkPeriodicBCStepLimiter::IsApplicable(const G4ParticleDefinition &particle)
{
    G4String particleName = particle.GetParticleName();
    if(particleName=="neutron")
    {
        return true;
    }
    else
    {
        return false;
    }

}


// PostStepDoIt()
// Result of StorkPeriodicBCStepLimiter acting on a neutron. An exact copy of
// the neutron is produced at the opposite boundary (other side of world volume)
// and the original is killed.
// ASSUMES A SYMMETRIC VOLUME
G4VParticleChange*
StorkPeriodicBCStepLimiter::PostStepDoIt(const G4Track &aTrack,
                                         const G4Step &aStep)
{
    aParticleChange.Initialize(aTrack);
    G4VSolid *solidVol = aTrack.GetVolume()->GetLogicalVolume()->GetSolid();

    G4StepPoint *preStepPoint = aStep.GetPreStepPoint();

    G4ThreeVector curPos  =	preStepPoint->GetPosition ();
    G4ThreeVector curMom  =	preStepPoint->GetMomentum ();
    G4ThreeVector newPos  =	curPos;
    G4ThreeVector n = solidVol->SurfaceNormal(curPos);

    for(G4int i=0;i<3;i++)
    {
        if(n[i]==0)
        {
            newPos[i]*=1;
        }
        else
        {
            newPos[i]*=-1;
        }
    }


	// Create a new dynamic particle from the parent dynamic particle
    G4DynamicParticle *newDynamicParticle = new G4DynamicParticle();
    *newDynamicParticle = *(aTrack.GetDynamicParticle());

    // Create the primary neutron info for the primary particle
	pnInfo = new StorkPrimaryNeutronInfo();

	// Set the lifetime
	pnInfo->SetLifeTime(aTrack.GetLocalTime());

	//Get the NS process manager and use it to get the n_lambda data
	nsProcMan = StorkProcessManager::GetStorkProcessManagerPtr();
	pnInfo->SetEta(nsProcMan->GetNumberOfInteractionLengthsLeft(aStep.GetStepLength()));

	// Create a new track info object and set the primary neutron info
	trackInfo = new StorkTrackInfo();
	trackInfo->SetStorkPrimaryNeutronInfo(pnInfo);


    // Find the time of the hit (time the secondary track starts at)
    G4double timeStart = aTrack.GetGlobalTime();

    // Create a track for the secondary
    G4Track *newTrack = new G4Track(newDynamicParticle, timeStart, newPos);

    // Add track info to track
    newTrack->SetUserInformation(trackInfo);

	// Add the secondary to the particle change
    aParticleChange.AddSecondary(newTrack);

	// Kill the parent particle
    aParticleChange.ProposeTrackStatus(fStopAndKill);

    return &aParticleChange;
}

