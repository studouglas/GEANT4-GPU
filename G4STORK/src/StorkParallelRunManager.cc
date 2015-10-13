/*
StorkParallelRunManager.cc

Created by:		Liam Russell
Date:			07-08-2011
Modified:		11-03-2013

Source code for the StorkParallelRunManager class.

*/

#ifdef G4USE_TOPC


// Include header file

#include "StorkParallelRunManager.hh"


// Deal with static variables and functions
StorkParallelRunManager* StorkParallelRunManager::myRunManager = 0;

// Static callback functions
TOPC_BUF StorkParallelRunManager::myGenerateEventInput()
{
	return myRunManager->GenerateEventInput();
}

TOPC_BUF StorkParallelRunManager::myDoEvent(void *input)
{
	return myRunManager->DoEvent(input);
}

TOPC_ACTION StorkParallelRunManager::myCheckEventResult(void *input,
                                                        void *outputData)
{
	return myRunManager->CheckEventResult(input,outputData);
}


// Constructor
// Calls the constructor of the base class (StorkRunManager) to set member data
StorkParallelRunManager::StorkParallelRunManager(const StorkParseInput* infile)
:StorkRunManager(infile)
{
    i_event = numEvents = 0;

    // Set the marshaling object pointers to NULL
    recMED = sendMED = NULL;
    recMPD = sendMPD = NULL;

    // Set seeds pointer to null
    slaveSeeds = NULL;
}


// Destructor
// Deletes any dynamically allocated data
StorkParallelRunManager::~StorkParallelRunManager()
{
	// Delete the marshaling objects
	if(recMED) delete recMED;
    if(recMPD) delete recMPD;
	if(sendMED) delete sendMED;
	if(sendMPD) delete sendMPD;

	// Delete the seeds
	if(slaveSeeds) delete [] slaveSeeds;

	// Set the pointers to NULL
    recMED = sendMED = NULL;
    recMPD = sendMPD = NULL;
    slaveSeeds = NULL;
}


// BeamOn()
// Begins the simulation.
void StorkParallelRunManager::BeamOn(G4int n_event, const char* macroFile,
                                     G4int n_select)
{
    G4bool cond = ConfirmBeamOnCondition();
    if(cond)
    {
        InitializeVar(n_event);
        EventMan = G4EventManager::GetEventManager();

        // Set the number of events in the primary generator action
        genAction->SetNumEvents(n_event);

        if(n_event>0)
        {
            while(runIDCounter < numRuns)
            {
                // Only initialize run now on the master, the slaves must
                // initialize AFTER any geometry changes proposed by the master
                if(TOPC_is_master()) RunInitialization();

                // Process events and terminate run
                DoEventLoop(n_event,macroFile,n_select);
                RunTermination();

                // Update the source distributions of the primary generator on
                // the master, and tally the run results
                if(TOPC_is_master())
                {
                    TallyRunResults();

                    if (sourceConverged)
                        G4cout << G4endl << "#### Souce Has Converged #####" << G4endl;
                    else
                        G4cout << G4endl << "#### Souce Has Not Converged #####" << G4endl;

                    runAction->UpdateSourceDistributions();
                }

                runStart += runDuration;
                runEnd += runDuration;

                // Save the source distribution if the given interval of runs
                // has passed
                if(TOPC_is_master() && saveInterval > 0
                   && !(runIDCounter%saveInterval)){
                    SaveSourceDistribution(saveFile);
                    if(saveFissionData) SaveFissionDistribution(fissionFile);
                }
            }

            // Save the final source distribution if the save interval is not
            // zero and it has not been just saved
            if(TOPC_is_master() && saveInterval && (runIDCounter%saveInterval ||
                                                    saveInterval == 1)){
                SaveSourceDistribution(saveFile);
                if(saveFissionData) SaveFissionDistribution(fissionFile);
            }
        }
   }
}


// RunInitialization()
// Initialize the master using the StorkRunManager function. The slaves only
// need the basic G4RunManager initialization.
void StorkParallelRunManager::RunInitialization()
{
    if(TOPC_is_master())
    {
        StorkRunManager::RunInitialization();
    }
    else
    {
        G4RunManager::RunInitialization();
    }
}


// DoEventLoop()
// Override G4RunManager::DoEventLoop()
// Generate seeds for the slaves and start parallel processing.
void StorkParallelRunManager::DoEventLoop(G4int n_event, const char*, G4int)
{
    // Make the run manager availble to the callback functions
    myRunManager = this;

	// Event loop
    i_event = 0;
    numEvents = n_event;

    // Initialize the current run
    if(TOPC_is_master())
    {
    	genAction->InitializeRun();
    	if(slaveSeeds) delete [] slaveSeeds;
    	slaveSeeds = new G4long[numEvents];

    	// Generate new random seeds for each event
    	for(G4int i=0; i<numEvents; i++)
    	{
    		slaveSeeds[i] = (G4long) (100000000L * G4UniformRand());
    	}
    }

    // Send events to the slaves and collect responses
    // Simulation pauses here until all events are simulated
    TOPC_master_slave(myGenerateEventInput, myDoEvent, myCheckEventResult,NULL);
}


// GenerateEventInput()
// Generate and marshal the primary data for the current event on the master
// The resultant marshaled data is passed to the slaves
TOPC_BUF StorkParallelRunManager::GenerateEventInput()
{
    if(i_event >= numEvents) return NOTASK;

    // Marshal the primary data (delete any data that already exists)
    if(sendMPD) delete sendMPD;
    StorkPrimaryData *primaryData = genAction->GetPrimaryData(i_event);

    // If no primary data received, do not send task
    if(!primaryData) return NOTASK;

    // Clear any previous property changes, and create a local property change
    primaryData->propChanges->clear();
    StorkMatPropChange change;

	// If interpolation has started, find the updated values for each property
    if(interpStarted)
    {
        // the updated material properties are stored in the primarydata class
        for(G4int i=0; i < theMPInterpMan->GetNumberOfInterpVectors(); i++)
        {
            // Set the property change (material, property,
            // current (updated) world condition
            change.Set(((*theMPInterpMan)[i])->second,
                        propValues[i]);

            primaryData->propChanges->push_back(change);
        }
    }


    // Set primary seed and event number
    primaryData->eventSeed = slaveSeeds[i_event];
    primaryData->eventNum = i_event;
    sendMPD = new MarshaledStorkPrimaryData(primaryData);

    // Increment the event counter
    i_event++;

    return TOPC_MSG(sendMPD->getBuffer(), sendMPD->getBufferSize());
}


// DoEvent()
// Use the marshaled input to simulate the event on the slave
// Marshal the event data and return it to the master
TOPC_BUF StorkParallelRunManager::DoEvent(void *input)
{
    // Unmarshal the primary data buffer (delete any data that exists)
    if(recMPD) delete recMPD;
    recMPD = new MarshaledStorkPrimaryData(input);

    // Unmarshal the buffer and set the primaries in the generator action
    StorkPrimaryData *primaryData = recMPD->unmarshal();
	genAction->SetPrimaries(primaryData);

    // Set the random seed
    CLHEP::HepRandom::setTheSeed(primaryData->eventSeed);

    // Set the event number given by the master
    G4int eventID = primaryData->eventNum;

    // Update world and initialize run for slaves (on the first event of the
	// run!)
    if(!TOPC_is_master() &&
		G4StateManager::GetStateManager()->GetCurrentState() == G4State_Idle)
	{
		// Exctracts data from the primary data class and sends it to the world
		// to modify the world properties
		if(G4int(primaryData->propChanges->size()) > 0)
		{
			UpdateWorld(*(primaryData->propChanges));
		}

		RunInitialization();
	}

	//newly added
	if(primaryData)
        delete primaryData;

    // Run the current event
    currentEvent = GenerateEvent(eventID);
    eventManager->ProcessOneEvent(currentEvent);
    AnalyzeEvent(currentEvent);

    StackPreviousEvent(currentEvent);
    currentEvent = 0;

	// Delete any event data stored in the sending event data pointer
	if(sendMED) delete sendMED;

	// Marshal the data
	sendMED = new MarshaledStorkEventData(eventAction->GetEventData());

    return TOPC_MSG(sendMED->getBuffer(), sendMED->getBufferSize());
}


// CheckEventResult()
// Updates the run with the results of the event.
// Executes on the master.
TOPC_ACTION StorkParallelRunManager::CheckEventResult(void *input, void *outputData)
{
    if(input == NULL) return NO_ACTION;
    else if(outputData == NULL)
    {
	G4cerr << "Master Check: No output for event" << G4endl;
	return NO_ACTION;
    }

    // Delete any event data stored in the receiving event data pointer
	if(recMED) delete recMED;

    // Recreate the marshalled event data
	recMED = new MarshaledStorkEventData(outputData);

    // Tally the event data in the run action
    //newly added
    StorkEventData *eventData = recMED->unmarshal();
    runAction->TallyEvent(eventData);
    //newly added
    delete eventData;

    return NO_ACTION;
}

#endif // G4USE_TOPC
