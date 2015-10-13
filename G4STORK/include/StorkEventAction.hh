/*
StorkEventAction.hh

Created by:		Liam Russell
Date:			22-06-2011
Modified:		11-03-2013

Header for StorkEventAction class.

This class takes the tally hits created by the sensitive detectors, and converts
them into an StorkEventData container that can be marshalled.

*/

#ifndef NSEVENTACTION_H
#define NSEVENTACTION_H

// Include G4-STORK header files
#include "StorkTallyHit.hh"
#include "StorkEventData.hh"
#include "StorkTrackInfo.hh"

// Include Geant4 header files
#include "G4Timer.hh"
#include "G4UserEventAction.hh"
#include "G4RunManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4PrimaryParticle.hh"
#include "G4UnitsTable.hh"
#include "G4ios.hh"
#include "globals.hh"


class StorkEventAction: public G4UserEventAction
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkEventAction(G4String name = "Reactor");
        ~StorkEventAction();

        void BeginOfEventAction(const G4Event *anEvent);
        void EndOfEventAction(const G4Event *anEvent);

        StorkEventData* GetEventData() const { return eventData; };


    private:
        // Private member data

        G4int tfHCID, tmHCID;
        G4String SDname;
        StorkEventData *eventData;


#ifdef G4TIMEEA
        G4Timer eventTimer;
#endif
};

#endif // NSEVENTACTION_H
