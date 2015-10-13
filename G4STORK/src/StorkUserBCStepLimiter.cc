/*
StorkUserBCStepLimiter.cc

Created by:		Wesley Ford
Date:			22-06-2012
Modified:		17-02-2012

Source code file for StorkUserBCStepLimiter class.

*/


// Include header file
#include "StorkUserBCStepLimiter.hh"


// Constructor()
// Calls the G4StepLimiter constructor with the proper name
StorkUserBCStepLimiter::StorkUserBCStepLimiter(std::vector<G4int>* PeriodicBC, std::vector<G4int>* ReflectBC,
                                        const G4String &aName, G4String worldPhysName)
: G4StepLimiter(aName)
{
	worldName=worldPhysName;
	nsProcMan = NULL;
	pnInfo = NULL;
	G4int offset[6]={0,0,0,0,0,0};
	G4ThreeVector n1, n2;

	for(G4int i=0; i<6; i++)
	{
        zeroSides.push_back(i);
        BCTransform[i] = NULL;
	}

	//BCTransform = new StorkBCTransform *[6];

	for(G4int i=0; i<G4int((*ReflectBC).size()); i++)
	{
        n1=GetNormal((*ReflectBC)[i]);
        BCTransform[(*ReflectBC)[i]]= new StorkReflectBCTransform(n1);
        zeroSides.erase(zeroSides.begin()+(*ReflectBC)[i]-offset[G4int((*ReflectBC)[i])]);
        for(G4int j=(*ReflectBC)[i]; j<6; j++)
        {
            offset[j]=offset[j]+1;
        }
	}
	for(G4int i=0; i<G4int((*PeriodicBC).size()); i=i+2)
	{
        n1=GetNormal((*PeriodicBC)[i]);
        n2=GetNormal((*PeriodicBC)[i+1]);
        BCTransform[(*PeriodicBC)[i]]= new StorkPeriodicBCTransform(n1,n2);
        BCTransform[(*PeriodicBC)[i+1]]= new StorkPeriodicBCTransform(n2,n1);
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
G4double StorkUserBCStepLimiter::PostStepGetPhysicalInteractionLength(
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

    G4bool check=true;
    for(G4int i=0; i<G4int(zeroSides.size()); i++)
    {
        if(side==zeroSides[i])
        {
            check=false;
        }
    }

    if((n.dot(curMomDir)>=0.)&&(check))
    {
        return DBL_MIN;
    }
    else
    {
        return DBL_MAX;
    }
}


// IsApplicable()
// Check that the particle is a neutron, otherwise not applicable.
G4bool
StorkUserBCStepLimiter::IsApplicable(const G4ParticleDefinition &particle)
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
// Result of StorkUserBCStepLimiter acting on a neutron. An exact copy of
// the neutron is produced at the opposite boundary (other side of world volume)
// and the original is killed.
// ASSUMES A SYMMETRIC VOLUME
G4VParticleChange*
StorkUserBCStepLimiter::PostStepDoIt(const G4Track &aTrack,
                                         const G4Step &aStep)
{
    aParticleChange.Initialize(aTrack);
    G4VSolid *solidVol = aTrack.GetVolume()->GetLogicalVolume()->GetSolid();

    G4StepPoint *preStepPoint = aStep.GetPreStepPoint();

    G4ThreeVector newPos  =	preStepPoint->GetPosition ();
    G4ThreeVector newMomDir  =	preStepPoint->GetMomentumDirection ();
    G4ThreeVector n = solidVol->SurfaceNormal(newPos);
    G4int side=-1;

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

    (BCTransform[side])->Transform(newPos, newMomDir);

    if(newMomDir==G4ThreeVector(0.,0.,0.))
    {
        G4cout << "Bad Transform" << G4endl;
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

    // Change Momentum Direction
    newTrack->SetMomentumDirection(newMomDir);

    // Add track info to track
    newTrack->SetUserInformation(trackInfo);

	// Add the secondary to the particle change
    aParticleChange.AddSecondary(newTrack);

	// Kill the parent particle
    aParticleChange.ProposeTrackStatus(fStopAndKill);

    return &aParticleChange;
}

G4ThreeVector StorkUserBCStepLimiter::GetNormal(int side)
{
    G4ThreeVector n = G4ThreeVector(0.,0.,0.);
    if(floor(double(side)/2)==ceil(double(side)/2))
    {
        n[int(side/2)]=1;
    }
    else
    {
        n[int(side/2)]=-1;
    }

    return n;
}

