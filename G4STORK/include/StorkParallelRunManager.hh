/*
StorkParallelRunManager.hh

Created by:		Liam Russell
Date:			07-08-2011
Modified:		11-03-2013

Header for the StorkParallelRunManager class.

This is the parallel run manager and it is derived from the StorkRunManager
class. The TOPC toolkit is used to communicate between the master and slave
processes. The simulation starts with the beamOn function where it sets up the
intial pointers and run counters. Then it initializes the run (begin of run
action), and goes into the do event loop.

The three main steps of the parallel do event loop are:

    1. GenerateEventInput - Generate primaries, random seed and event number.
                                This primary data is sent to a slave process.
    2. DoEvent - Initialize event with primary data and excute. Return the
                    results to the master process as event data.
    3. CheckEventResults - Send the event data to the run action class (only
                                on the master). The run tallies and stores
                                the event data.

After all events finish, the run ends (end of run action) and the run action
performs analysis and updates the primary generator action.

*/

#ifdef G4USE_TOPC

#ifndef STORKPARALLELRUNMANAGER_H
#define STORKPARALLELRUNMANAGER_H

#include "topc.h"
#include "StorkRunManager.hh"
#include "MarshaledStorkEventData.h"
#include "MarshaledStorkPrimaryData.h"
#include "MarshaledObj.h"
#include "G4StateManager.hh"


class StorkParallelRunManager : public StorkRunManager
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkParallelRunManager(const StorkParseInput* infile);
        virtual ~StorkParallelRunManager();

        // Start the simulation
        virtual void BeamOn(G4int n_event, const char* macroFile=0,
                            G4int n_select=-1);
        // Process all of the events for a run
        virtual void DoEventLoop(G4int n_event, const char* macroFile=0,
                                 G4int n_select=-1);
        // Initialize the run
        virtual void RunInitialization();


    protected:
        // Protected member functions

        // TOP-C static callback functions (locatable by TOP-C code)
		static TOPC_BUF myGenerateEventInput();
		static TOPC_BUF myDoEvent(void *input);
		static TOPC_ACTION myCheckEventResult(void *input, void *output);

        // Implementations of TOP-C functions (contain the actual instructions)
        TOPC_BUF GenerateEventInput();
        TOPC_BUF DoEvent(void *input);
        TOPC_ACTION CheckEventResult(void *input, void *output);

     protected:
        // Protected member variables

        // Static pointer to singleton run manager
		static StorkParallelRunManager *myRunManager;

        G4int i_event;          // Current event id (number)
        G4int numEvents;        // Total number of events
        G4long *slaveSeeds;     // Random number seeds for the slaves

        // Marshalled data classes (sending and recieving)
        MarshaledStorkPrimaryData *sendMPD;
        MarshaledStorkEventData *sendMED;
        MarshaledStorkPrimaryData *recMPD;
        MarshaledStorkEventData *recMED;
};

#endif // STORKPARALLELRUNMANAGER_H

#endif // G4USE_TOPC
