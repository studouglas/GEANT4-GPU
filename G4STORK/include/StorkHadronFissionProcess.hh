/*
StorkHadronFissionProcess.hh

Created by:		Liam Russell
Date:			29-02-2012
Modified:       11-03-2012

Header file for StorkHadronFissionProcess class.

This class is derived from G4HadronFissionProcess to modify the StartTracking()
function and to add a get function for theNumberOfInteractionLenghtLeft
member variable.

*/


#ifndef STORKHADRONFISSIONPROCESS_H
#define STORKHADRONFISSIONPROCESS_H

#include "G4HadronFissionProcess.hh"
#include "StorkHadronicProcess.hh"
#include "StorkHadProjectile.hh"
#include "StorkNeutronData.hh"
#include "StorkHadProjectile.hh"
#include "G4TransportationManager.hh"
#include "StorkNeutronHPFission.hh"

#include "G4DynamicParticle.hh"
#include "G4Material.hh"
#include "G4Nucleus.hh"


class StorkHadronFissionProcess : public G4HadronFissionProcess
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkHadronFissionProcess(const G4String& processName = "StorkHadronFission")
        :G4HadronFissionProcess(processName)
        {
            theFissModel = NULL;
            theNavi = NULL;
            theDataStore = NULL;

//            SetVerboseLevel(2);
        }
        virtual ~StorkHadronFissionProcess() {
            if(theFissModel) delete theFissModel;
        };

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

        // Produce the characteristics of a delayed neutron
		inline StorkNeutronData GetADelayedNeutron(G4DynamicParticle *aNeutron,
												G4Material *theMat);
        inline StorkNeutronData GetADelayedNeutron(G4DynamicParticle *aNeutron,
                                               G4Material *theMat, G4Nucleus theTarget);

        inline G4Nucleus GetFissionNucleus(G4StepPoint* aStep);
        inline G4Nucleus GetFissionNucleus(G4double fnEnergy, G4ThreeVector fnSite);


	private:

		G4int procIndex;
        StorkNeutronHPFission *theFissModel;
        G4Navigator* theNavi;
        G4CrossSectionDataStore *theDataStore;

};

StorkNeutronData StorkHadronFissionProcess::GetADelayedNeutron(
                                                               G4DynamicParticle *aNeutron,
                                                               G4Material *theMat, G4Nucleus theTarget)
{
	// Local Variables
	G4HadFinalState* theResult = NULL;
    StorkHadProjectile projectile(*aNeutron,theMat);
	G4HadSecondary *aSec = NULL;
	G4DynamicParticle *aDelayedN = NULL;
	G4bool dnFlag = false;


	// Keep getting a result until a delayed neutron is produced
	do
	{
		// Simulate a fission
		try
		{
            G4HadProjectile* Temp = reinterpret_cast<G4HadProjectile*>(&projectile);
			//theResult = theFissionModel->ApplyYourself(*Temp, theTarget);
            theResult = theFissModel->ApplyYourself(*Temp,theTarget);

		}
		catch(G4HadronicException aR)
		{
			G4Exception("G4HadronicProcess", "007", FatalException,
						"The fission model failed to produce a result.");

		}

		// Check to see if the result contains a delayed neutron
		for(G4int i=0; i<theResult->GetNumberOfSecondaries(); i++)
		{
			aSec = theResult->GetSecondary(i);

			// Check if a delayed neutron was created
			if(aSec->GetParticle()->GetParticleDefinition() ==
               G4Neutron::Neutron() &&
               aSec->GetTime() > aNeutron->GetProperTime())
			{
				dnFlag = true;
				aDelayedN = aSec->GetParticle();

				break;
			}
		}
	}
	while(!dnFlag);


	// Create the neutron data container
	StorkNeutronData dnData(aSec->GetTime(),
                            aSec->GetTime()-aNeutron->GetProperTime(),
						    G4ThreeVector(0.,0.,0.), aDelayedN->GetMomentum());

	return dnData;
}


// GetADelayedNeutron()
// Produce the characteristics (momentum, time of birth, etc) of a delayed
// neutron by simulating a fission with a given incoming neutron.
// NOTE: the fission is virtual and DOES NOT OCCUR
StorkNeutronData StorkHadronFissionProcess::GetADelayedNeutron(
													G4DynamicParticle *aNeutron,
													G4Material *theMat)
{
	// Local Variables
	G4Nucleus theTarget;
	G4HadFinalState* theResult = NULL;
	StorkHadProjectile theProjectile(*aNeutron, theMat);
	G4HadronicInteraction *theFissionModel = NULL;
	G4HadSecondary *aSec = NULL;
	G4DynamicParticle *aDelayedN = NULL;
	G4bool dnFlag = false;
	theDataStore = GetCrossSectionDataStore();


	// Find the target element
	G4Element *anElement = NULL;
	try
	{
		anElement = theDataStore->SampleZandA(aNeutron, theMat, theTarget);
	}
	catch(G4HadronicException & aR)
	{
//		DumpState(aTrack,"SampleZandA");
		G4Exception("G4HadronicProcess", "007", FatalException,
					"GetADelayedNeutron failed on element selection.");
	}


	// Find the appropriate interaction
	try
	{
		theFissionModel = ChooseHadronicInteraction(
												aNeutron->GetKineticEnergy(),
												theMat, anElement);
	}
	catch(G4HadronicException & aE)
	{
		G4cout << "Target element "<< anElement->GetName() << G4endl;
//		DumpState(aTrack,"ChooseHadronicInteraction");
		G4Exception("G4HadronicProcess", "007", FatalException,
					"ChooseHadronicInteraction failed.");
	}

	// Keep getting a result until a delayed neutron is produced
	do
	{
		// Simulate a fission
		try
		{
			G4HadProjectile* Temp = reinterpret_cast<G4HadProjectile*>(&theProjectile);
			theResult = theFissionModel->ApplyYourself((*Temp), theTarget);
		}
		catch(G4HadronicException aR)
		{
			G4Exception("G4HadronicProcess", "007", FatalException,
						"The fission model failed to produce a result.");

		}


		// Check to see if the result contains a delayed neutron
		for(G4int i=0; i<theResult->GetNumberOfSecondaries(); i++)
		{
			aSec = theResult->GetSecondary(i);

			// Check if a delayed neutron was created
			if(aSec->GetParticle()->GetParticleDefinition() ==
               G4Neutron::Neutron() &&
               aSec->GetTime() > aNeutron->GetProperTime())
			{
				dnFlag = true;
				aDelayedN = aSec->GetParticle();

				break;
			}
		}
	}
	while(!dnFlag);


	// Create the neutron data container
	StorkNeutronData dnData(aSec->GetTime(),
                            aSec->GetTime()-aNeutron->GetProperTime(),
						    G4ThreeVector(0.,0.,0.), aDelayedN->GetMomentum());

	return dnData;
}

G4Nucleus StorkHadronFissionProcess::GetFissionNucleus(G4StepPoint* aStep){

    G4DynamicParticle *aNeutron;
    if(!theNavi) theNavi = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();

    aNeutron = new G4DynamicParticle(G4Neutron::NeutronDefinition(),
                                     aStep->GetMomentumDirection(), aStep->GetKineticEnergy());

    G4ThreeVector Location = aStep->GetPosition();


    G4Material* Material = theNavi->LocateGlobalPointAndSetup(Location)->GetLogicalVolume()->GetMaterial();
    G4Nucleus aNuc;

    if(!theDataStore) theDataStore = GetCrossSectionDataStore();

    try
	{
		theDataStore->SampleZandA(aNeutron, Material, aNuc);
	}
	catch(G4HadronicException & aR)
	{
		G4Exception("G4HadronicProcess", "007", FatalException,
					"GetNeutronFissionIso failed on element selection.");
	}

    if(aNeutron) delete aNeutron;
    return aNuc;


}

G4Nucleus StorkHadronFissionProcess::GetFissionNucleus(G4double fnEnergy, G4ThreeVector fnSite){

    G4DynamicParticle *aNeutron;
    G4ThreeVector MomentumDir;
    if(!theNavi) theNavi = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();

    MomentumDir.setRThetaPhi(1.0, G4UniformRand()*CLHEP::pi,
                          G4UniformRand()*2.0*CLHEP::pi);

    aNeutron = new G4DynamicParticle(G4Neutron::NeutronDefinition(),
                                    MomentumDir, fnEnergy);

    theNavi->ResetStackAndState();
    G4Material* Material = theNavi->LocateGlobalPointAndSetup(fnSite)->GetLogicalVolume()->GetMaterial();

    G4Nucleus aNuc;

    if(!theDataStore) theDataStore = GetCrossSectionDataStore();

    try
	{
		theDataStore->SampleZandA(aNeutron, Material, aNuc);
	}
	catch(G4HadronicException & aR)
	{
		G4Exception("G4HadronicProcess", "007", FatalException,
					"GetFissionNucleus failed on element selection.");
	}

    if(aNeutron) delete aNeutron;
    return aNuc;


}

#endif // NSHADRONFISSIONPROCESS_H

