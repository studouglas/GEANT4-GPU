/*
StorkEventAction.cc

Created by:		Liam Russell
Date:		    22-06-2011
Modified:		11-03-2013

Source code for StorkEventAction class.

*/


// Include header file
#include "StorkEventAction.hh"


// Constructor
StorkEventAction::StorkEventAction(/*G4ThreeVector res,*/ G4String name)
{
    eventData = NULL;
    SDname = name;
}


// Destructor
StorkEventAction::~StorkEventAction()
{
    if(eventData) delete eventData;
    eventData = NULL;
}


// BeginOfEventAction()
// Actions taken before each event begins.  Clear data containers and find the
// identifier for the hit data of the event.
void StorkEventAction::BeginOfEventAction(const G4Event*)
{
#ifdef G4TIMEEA
    // Start the event timer
    eventTimer.Start();
#endif

    G4SDManager *sDMan = G4SDManager::GetSDMpointer();

    // Find hit collection ID's for the NeutronSD sensitive detector
    tfHCID = sDMan->GetCollectionID(SDname + "/Tally");

    // Clear containers
    if(eventData) delete eventData;
    eventData = NULL;
}


// EndOfEventAction()
// Actions taken at the end of each event.  Convert the tally hit collection
// data to a StorkEventData container.
void StorkEventAction::EndOfEventAction(const G4Event *anEvent)
{
#ifdef G4TIMEEA
    G4Timer calcTimer;
    calcTimer.Start();
#endif

    // Local variables
    G4int numElements = 0;          // Number of elements in a hit collection
    const SiteVector *fSiteVector = NULL;
    const DblVector *fnEnergy = NULL;
    const NeutronSources *survivors = NULL;
    const NeutronSources *delayed = NULL;
    StorkTallyHit *aTally = NULL;

    // Create a new event data container
    if(eventData) delete eventData;
    eventData = NULL;
    eventData = new StorkEventData();
    eventData->eventNum = anEvent->GetEventID();

    // Get the hit collection for this event
    G4HCofThisEvent *HCE = anEvent->GetHCofThisEvent();

    // Get the individual hit collections
    TallyHC *tFuelHC = (TallyHC*)(HCE->GetHC(tfHCID));
//    TallyHC *tModHC = (TallyHC*)(HCE->GetHC(tmHCID));

	// Unpack the tally hit
    if(tFuelHC)
    {
    	// Set the tally hit pointer to the first (and only) hit in the
    	// collection
    	aTally = (*tFuelHC)[0];

    	// Get pointers to the survivors and delayed neutrons
    	survivors = aTally->GetSurvivors();
    	delayed = aTally->GetDelayed();
    	fnEnergy = aTally->GetFissionEnergies();

    	// Insert these into the event data structure
    	eventData->survivors->insert(eventData->survivors->begin(),
									 survivors->begin(),survivors->end());
		eventData->delayed->insert(eventData->delayed->begin(),
								   delayed->begin(),delayed->end());
		eventData->fnEnergy->insert(eventData->fnEnergy->begin(),
								   fnEnergy->begin(),fnEnergy->end());

        // Get a pointer to the site vector and find the number of sites
        fSiteVector = (*tFuelHC)[0]->GetFissionSites();
        numElements = fSiteVector->size();

        // Copy the contents of the site vector into the event data container
        eventData->fSites->reserve(numElements);

        for(G4int i=0; i<numElements; i++)
        {
            eventData->fSites->push_back(StorkTripleFloat((*fSiteVector)[i]));
        }

        eventData->numNProd += (*tFuelHC)[0]->GetNProd();
        eventData->numNLost += (*tFuelHC)[0]->GetNLost();
        eventData->numDProd += (*tFuelHC)[0]->GetDProd();

        eventData->totalLifetime += (*tFuelHC)[0]->GetTotalLifetime();
    }

#ifdef G4TIMEEA
    calcTimer.Stop();
    eventTimer.Stop();

    G4cout << "Event Timing:" << G4endl
           << "Total time:  " << eventTimer << G4endl
           << "Calculation time: " << calcTimer << G4endl;
#endif
}
