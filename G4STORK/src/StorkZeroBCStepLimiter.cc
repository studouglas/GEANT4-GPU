#include "StorkZeroBCStepLimiter.hh"

// Constructor()
// Calls the G4StepLimiter constructor with the proper name
StorkZeroBCStepLimiter::StorkZeroBCStepLimiter(std::vector<G4int>* PeriodicBC, std::vector<G4int>* ReflectBC,
                                        const G4String &aName, G4String worldPhysName)
: G4StepLimiter(aName)
{
	worldName=worldPhysName;
	nsProcMan = NULL;
	pnInfo = NULL;
	G4int offset[6]={0,0,0,0,0,0};

	for(G4int i=0; i<6; i++)
	{
        zeroSides.push_back(i);
	}

	for(G4int i=0; i<G4int((*ReflectBC).size()); i++)
	{
        zeroSides.erase(zeroSides.begin()+(*ReflectBC)[i]-offset[G4int((*ReflectBC)[i])]);
        for(G4int j=(*ReflectBC)[i]; j<6; j++)
        {
            offset[j]=offset[j]+1;
        }
	}
	for(G4int i=0; i<G4int((*PeriodicBC).size()); i=i+2)
	{
        zeroSides.erase(zeroSides.begin()+(*PeriodicBC)[i]-offset[G4int((*PeriodicBC)[i])]);
        for(G4int j=(*PeriodicBC)[i]; j<6; j++)
        {
            offset[j]=offset[j]+1;
        }
        zeroSides.erase(zeroSides.begin()+(*PeriodicBC)[i+1]-offset[G4int((*PeriodicBC)[i+1])]);
        for(G4int j=(*PeriodicBC)[i+1]; j<6; j++)
        {
            offset[j]=offset[j]+1;
        }
	}
}


// PostStepGetPhysicalInteractionLength()
// Returns a step size of DBL_MIN (very small) if the neutron has entered the
// world (outside) volume.  Otherwise returns DBL_MAX (very big).
G4double StorkZeroBCStepLimiter::PostStepGetPhysicalInteractionLength(
                                            const G4Track &aTrack,
                                            G4double,
                                            G4ForceCondition *condition)
{
    // Set condition to "Not Forced"
    *condition = NotForced;

    G4String physVolName = aTrack.GetVolume()->GetName();
    G4VSolid *solidVol = aTrack.GetVolume()->GetLogicalVolume()->GetSolid();

    // Check that the neutron is in the world volume
    if(physVolName!=worldName)
    {
        return DBL_MAX;
    }

    // Check that the neutron is ENTERING the world volume (rather than leaving)
    G4ThreeVector curPos  =	aTrack.GetPosition ();
    G4ThreeVector curMomDir  =	aTrack.GetMomentumDirection ();
    G4ThreeVector n = solidVol->SurfaceNormal(curPos);

    int side = 0;

    for(G4int i=0; i<3; i++)
    {
        if(n[i]==1)
        {
            side=2*i;
        }
        else if(n[i]==-1)
        {
            side=2*i+1;
        }
    }

    for(G4int i=0; i<G4int(zeroSides.size()); i++)
    {
        if(side==zeroSides[i])
        {
            return DBL_MIN;
        }
    }

    return DBL_MAX;
}


// IsApplicable()
// Check that the particle is a neutron, otherwise not applicable.
G4bool
StorkZeroBCStepLimiter::IsApplicable(const G4ParticleDefinition &particle)
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
// Result of StorkZeroBCStepLimiter acting on a neutron. An exact copy of
// the neutron is produced at the opposite boundary (other side of world volume)
// and the original is killed.
// ASSUMES A SYMMETRIC VOLUME
G4VParticleChange*
StorkZeroBCStepLimiter::PostStepDoIt(const G4Track &aTrack,
                                         const G4Step& /*aStep*/)
{
    aParticleChange.Initialize(aTrack);
    //aParticleChange.ProposeTrackStatus(fStopAndKill);
    return &aParticleChange;
}

