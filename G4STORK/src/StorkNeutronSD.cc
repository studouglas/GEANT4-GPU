/*
StorkNeutronSD.cc

Created by:		Liam Russell
Date:			22-06-2011
Modified:		11-03-2013

Source code file for StorkNeutronSD class.

*/


// Include header file

#include "StorkNeutronSD.hh"


// Constructor

StorkNeutronSD::StorkNeutronSD(G4String name, G4int KCalcType, G4bool instD, G4bool sourcefileD)
:G4VSensitiveDetector(name), kCalcType(KCalcType), instantDelayed(instD), sourcefileDelayed(sourcefileD)
{
    // Set collection name and initialize ID
    collectionName.insert("Tally");
    HCID1 = -1;

    // Get a pointer to the run manager and process manager
    runMan = dynamic_cast<StorkRunManager*>(G4RunManager::GetRunManager());

    // Set process manager pointer to NULL
    procMan = NULL;
}


// Initialize()
// Initialize the sensitive detector member variables for the current event.
void StorkNeutronSD::Initialize(G4HCofThisEvent *HCE)
{
#ifdef G4TIMESD
    sdTimer.Start();
#endif

    // Make a new tally hit collection for this detector
    tallyHitColl = new TallyHC(SensitiveDetectorName, collectionName[0]);

    if(HCID1 < 0)
        HCID1 = G4SDManager::GetSDMpointer()->GetCollectionID(tallyHitColl);


    // Add the losses to the hit collection for this event
    HCE->AddHitsCollection(HCID1,tallyHitColl);

    // Get the new run end time
    runEnd = runMan->GetRunEnd();

    // Initialize and clear the member variables
    nLoss = nProd = dProd = 0;
    totalLifetime = 0.0;
    fSites.clear();
    fnEnergy.clear();
    survivors.clear();
	delayed.clear();
	prevTrackID = -1;

#ifdef STORK_EXPLICIT_LOSS
	nCap = nFiss = nEsc = nInel = 0;
#endif

#ifdef G4TIMESD
    cycles = 0;
    cycleTime = 0.0;
    sdTimer.Stop();
    totalSDTime = sdTimer.GetRealElapsed();
#endif

	// Get pointer to StorkProcessManager
	procMan = StorkProcessManager::GetStorkProcessManagerPtr();
}


// ProcessHits()
// Determine which reaction type of each hit.  Tally neutrons produced and lost,
// and lifetime of lost neutrons.  Also record fission sites/energy and save
// survivors and delayed neutrons.
// Kill any non-neutron particles produced.
G4bool StorkNeutronSD::ProcessHits(G4Step *aStep, G4TouchableHistory*)
{
#ifdef G4TIMESD
    phTimer.Start();
#endif

	// Get the track for the current particle
    G4Track *aTrack = aStep->GetTrack();

    // Particle type involved in hit
    G4ParticleDefinition* hitDefinition = aTrack->GetDefinition();

    // Only track hits by neutrons
    // Kill any particles that are not neutrons and return
    if(hitDefinition != G4Neutron::NeutronDefinition())
    {
		aTrack->SetTrackStatus(fKillTrackAndSecondaries);

#ifdef G4TIMESD
		phTimer.Stop();
		cycleTime += phTimer.GetRealElapsed();
		cycles++;
#endif

    	return true;
    }

	// Find data necessary to record the hit

	// The pre and post step inside the SD
    G4StepPoint *preStepPoint = aStep->GetPreStepPoint();
    G4StepPoint *postStepPoint = aStep->GetPostStepPoint();

    // Find the time of the hit
    G4double hitTime = postStepPoint->GetGlobalTime();
    // Find the lifetime of the track
    G4double lifetime = postStepPoint->GetLocalTime();

    // Set up other variables to be used later
    G4TrackVector *trackVector;
    std::vector<G4Track*>::iterator itr;


    // If the neutron is a primary neutron, on its first step, fix the lifetime
	// Also add the track info (fission generation)
	if(!aTrack->GetParentID()  && aTrack->GetCurrentStepNumber() == 1)
	{
		// Get the primary particle user information
		StorkPrimaryNeutronInfo *pnInfo = dynamic_cast<StorkPrimaryNeutronInfo*>(
									 aTrack->GetDynamicParticle()->
									 GetPrimaryParticle()->
									 GetUserInformation());

		// Set the lifetime of the primary track
		lifetime += pnInfo->GetLifeTime();
		aTrack->SetLocalTime(lifetime);
		postStepPoint->AddLocalTime(pnInfo->GetLifeTime());

		// Set the weight of the particle
		aTrack->SetUserInformation(new StorkTrackInfo(pnInfo->GetWeight()));
	}

    // Find the process used in the hit (if it is defined)
    hitProcess = "";
    if(postStepPoint->GetProcessDefinedStep() != 0)
    {
        hitProcess = postStepPoint->GetProcessDefinedStep()->GetProcessName();
    }

	// If the time is beyond the simulation time, the kill the track and
	// any secondaries (stops simulation from running forever)
	// Save the neutron as a survivor
	if(hitProcess == "StorkTimeStepLimiter")
	{
		SaveSurvivors(aTrack);

		aTrack->SetTrackStatus(fKillTrackAndSecondaries);

#ifdef G4TIMESD
		phTimer.Stop();
		cycleTime += phTimer.GetRealElapsed();
		cycles++;
#endif

		return true;
	}

    if(kCalcType != 2)
    {
        if(hitTime > runEnd)
        {
            G4cout << "*** ERROR particle exists beyond run end: "
                   << hitTime << " > " << runEnd << G4endl;
        }
	}


	// Record the site of the fission and the lifetime of the incident neutron.
	// Also update the neutron production and loss totals, and save any delayed
	// neutron daughters.
	if(hitProcess == "StorkHadronFission")
	{
        trackVector = const_cast<G4TrackVector*>(aStep->GetSecondary());
        itr = trackVector->begin();

        // Record number of daughter neutrons
        for( ; itr != trackVector->end(); itr++)
        {
            // Check if secondary is a neutron
            if((*itr)->GetDefinition() == G4Neutron::NeutronDefinition())
            {
                // Check if the neutron is a delayed neutron
                if((*itr)->GetGlobalTime() > hitTime)
                {
                    // Correct the lifetime of the delayed neutron
                    (*itr)->SetLocalTime((*itr)->GetGlobalTime() - hitTime);

                    // Correct global time to produce instantaneously
                    if(instantDelayed)
                    {
                        (*itr)->SetGlobalTime(hitTime);
                        if(kCalcType == 2)
                        {
                            SaveSurvivors((*itr));
                            (*itr)->SetTrackStatus(fKillTrackAndSecondaries);
                        }
                    }
                    // Otherwise save (and kill) the delayed neutron for later
                    else
                    {
                        SaveDelayed(*itr);
                        (*itr)->SetTrackStatus(fKillTrackAndSecondaries);
                    }

                    // Increment delayed neutron production counter
                    dProd++;
                }
                else if(kCalcType == 2)
                {
                    SaveSurvivors((*itr));
                    (*itr)->SetTrackStatus(fKillTrackAndSecondaries);
                }

                // Increment neutron production counter
                nProd++;
            }
            // Stop and kill particle if it is not a neutron
            else
            {
                (*itr)->SetTrackStatus(fKillTrackAndSecondaries);
            }
        }

        // Record tally info
        nLoss++;
        totalLifetime += lifetime;
        fSites.push_back(postStepPoint->GetPosition());

        // changed fnenergy to collect the energy from the poststeppoint instead of the presteppoint
        //fnEnergy.push_back(postStepPoint->GetKineticEnergy());

        fnEnergy.push_back(preStepPoint->GetKineticEnergy());

        #ifdef STORK_EXPLICIT_LOSS
            nFiss++;
        #endif

        if(kCalcType == 2)
        {
            #ifdef G4TIMESD
                phTimer.Stop();
                cycleTime += phTimer.GetRealElapsed();
                cycles++;
            #endif

            return true;
        }
	}
	// If the neutron is captured, update the loss counter and record the
	// lifetime.
	else if(hitProcess == "StorkHadronCapture")
	{
		nLoss++;
		totalLifetime += lifetime;

		// Kill any non-neutron (all) secondaries
		trackVector = const_cast<G4TrackVector*>(aStep->GetSecondary());
		itr = trackVector->begin();

		for( ; itr != trackVector->end(); itr++)
		{
			(*itr)->SetTrackStatus(fKillTrackAndSecondaries);
		}

#ifdef STORK_EXPLICIT_LOSS
		nCap++;
#endif
	}

	// If an inelastic collision occurs, set the first daughter to be the
	// incident neutron, and then any others (n,2n; etc.) are simply
	// daughter neutrons. Update the production and loss totals.
	else if(hitProcess == "StorkNeutronInelastic")
	{
		G4bool nMulti = false;

		trackVector = const_cast<G4TrackVector*>(aStep->GetSecondary());
		itr = trackVector->begin();

		for( ; itr != trackVector->end(); itr++)
		{
			// Set the neutrons lifetime to that of the current neutron
			if((*itr)->GetDefinition() == G4Neutron::NeutronDefinition())
			{
				if(nMulti)
				{
					nProd++;
				}
				else
				{
					nMulti = true;
					(*itr)->SetLocalTime(lifetime);
				}
			}
			// Kill secondary particle if it is not a neutron
			else
			{
				(*itr)->SetTrackStatus(fKillTrackAndSecondaries);
			}
		}

		// Check for inelastic collisions where neutron is killed
		if(!nMulti)
		{
			nLoss++;
			totalLifetime += lifetime;

#ifdef STORK_EXPLICIT_LOSS
			nInel++;
#endif
		}
	}
	// If an elastic collision occurs kill any secondaries (uranium atom, etc.)
	else if(hitProcess == "StorkHadronElastic")
	{
		trackVector = const_cast<G4TrackVector*>(aStep->GetSecondary());
		itr = trackVector->begin();

		for( ; itr != trackVector->end(); itr++)
		{
			if((*itr)->GetDefinition() != G4Neutron::NeutronDefinition())
			{
				(*itr)->SetTrackStatus(fKillTrackAndSecondaries);
			}
            else
            {
                nProd++;
            }
		}
	}
	// If the neutron leaves the simulation world, update the loss counter and
	// lifetime total.
	else if(hitProcess == "StorkZeroBCStepLimiter")
	{
		nLoss++;
		totalLifetime += lifetime;

		// Kill the neutron
		aTrack->SetTrackStatus(fKillTrackAndSecondaries);

        #ifdef STORK_EXPLICIT_LOSS
                nEsc++;
        #endif
	}
	else
	{
		if(hitProcess != "Transportation" &&
		   hitProcess != "StorkUserBCStepLimiter")
			G4cerr << "***WARNING: Untracked process is" << hitProcess << G4endl;
	}

#ifdef G4TIMESD
    phTimer.Stop();
    cycleTime += phTimer.GetRealElapsed();
    cycles++;
#endif

    return true;
}


// SaveSurvivors()
// Finds the position, lifetime and momentum of the particle at the run
// threshold (end of the run)
// Assumes that velocity doesn't change during flight (shouldn't because no
// along step processes active)
void StorkNeutronSD::SaveSurvivors(const G4Track *aTrack)
{
    // Get size of last step
    if(aTrack->GetStep())
    {
        G4double previousStepSize = aTrack->GetStep()->GetStepLength();

        // Get the number of interaction length left data
        G4double *n_lambda = procMan->
                                GetNumberOfInteractionLengthsLeft(previousStepSize);

        // Create the survivor record
        StorkNeutronData aSurvivor(runEnd, aTrack->GetLocalTime(),
                           aTrack->GetPosition(), aTrack->GetMomentum(),
                           n_lambda[0], n_lambda[1], n_lambda[2], n_lambda[3], 1.0);

        // Add the survivor to the list
        survivors.push_back(aSurvivor);
    }
    else
    {
        // Create the survivor record
        StorkNeutronData aSurvivor(runEnd, aTrack->GetLocalTime(),
                           aTrack->GetPosition(), aTrack->GetMomentum(),
                           -1.0, -1.0, -1.0, -1.0, 1.0);

        // Add the survivor to the list
        survivors.push_back(aSurvivor);
    }
}


// SaveDelayed()
// Adds a delayed neutron entry to the Delayed neutron list.
// Corrects the lifetime to be time since fission that "created it".
void StorkNeutronSD::SaveDelayed(const G4Track *aTrack)
{
	// Create a neutron data container for the delayed neutron and put it in
	// the delayed list.
	// lifetime is time of neutron birth is time of creating fission.
	StorkNeutronData aDelayed(aTrack->GetGlobalTime(), aTrack->GetLocalTime(),
						      aTrack->GetPosition(), aTrack->GetMomentum(),
                              -1.0, -1.0, -1.0, -1.0,
                              -1.0*runMan->GetCurrentRun()->GetRunID());

	delayed.push_back(aDelayed);
}


// EndOfEvent()
// Create the tally hit and add it to the collection
void StorkNeutronSD::EndOfEvent(G4HCofThisEvent*)
{
#ifdef G4TIMESD
    sdTimer.Start();
#endif

    StorkTallyHit *tHit = new StorkTallyHit();

    tHit->SetTotalLifetime(totalLifetime);
    tHit->SetNLost(nLoss);
    tHit->SetNProd(nProd);
    tHit->SetDProd(dProd);
    tHit->SetFissionSites(fSites);
    tHit->SetFissionEnergies(fnEnergy);
    tHit->SetSurvivors(survivors);
    tHit->SetDelayed(delayed);

    tallyHitColl->insert(tHit);

#ifdef G4TIMESD
    sdTimer.Stop();

    totalSDTime += sdTimer.GetRealElapsed() + cycleTime;

    // Output timing data for SD
    G4cout << "Sensitive Detector Timing:" << G4endl
           << "Total time taken for SD in this event = " << totalSDTime << "s"
           << G4endl
           << "Total time take to process hits = " << cycleTime << "s" << G4endl
           << "Average time for each hit = " << cycleTime/G4double(cycles)
           << "s" << G4endl;
#endif

#ifdef STORK_EXPLICIT_LOSS
    PrintCounterTotals();
#endif

}


// PrintCounterTotals()
// Diagnostic function to get the specific loss/production totals
void StorkNeutronSD::PrintCounterTotals() const
{
	G4cout << "Event totals:" << G4endl
		   << "Production:  " << nProd << G4endl
		   << "Total Loss:  " << nLoss << G4endl
		   << "Escapes:  " << nEsc << G4endl
		   << "Fissions:  " << nFiss << G4endl
		   << "Captures:  " << nCap << G4endl
		   << "Inelastic Loss:  " << nInel << G4endl;
}
