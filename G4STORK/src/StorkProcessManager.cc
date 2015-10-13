/*
StorkProcessManager.cc

Created by:		Liam Russell
Date:			24-05-2012
Modified:       11-03-2013

Source code file for StorkProcessManager class.

*/

#include "StorkProcessManager.hh"


// Initialize static pointer
StorkProcessManager* StorkProcessManager::theManager = NULL;


// GetStorkProcessManagerPtr()
// Static function to get pointer to StorkProcessManager singleton. This function
// will create the singleton if it does not already exist.
StorkProcessManager* StorkProcessManager::GetStorkProcessManagerPtr()
{
	if(!theManager)
	{
		theManager = new StorkProcessManager();
	}

	return theManager;
}


// GetStorkProcessManagerPtrIfExists()
// Static function to get pointer to StorkProcessManager singleton. This function
// will NOT create the singleton if it does not exist.
StorkProcessManager* StorkProcessManager::GetStorkProcessManagerPtrIfExists()
{
	return theManager;
}


// Constructor
// Initialize pointers and then set them to the appropriate processes.
StorkProcessManager::StorkProcessManager()
{
	// Initialize the pointers
	theElasticProcess = NULL;
    theInelasticProcess = NULL;
    theFissionProcess = NULL;
    theCaptureProcess = NULL;

    // Initialize data arrays
    numProcesses = 4;
    n_lambda = new G4double[numProcesses];
    procOrder = new G4String[numProcesses]();

	// Get pointers to the hadronic processes
    G4ProcessManager *procMan = G4Neutron::Neutron()->GetProcessManager();
    G4ProcessVector *procVector = procMan->GetPostStepProcessVector();
    G4String procName;

    for(G4int i=0; i<procVector->entries(); i++)
    {
        procName = (*procVector)[i]->GetProcessName();

        if(procName == "StorkHadronElastic")
        {
            theElasticProcess = dynamic_cast<StorkHadronElasticProcess*>((*procVector)[i]);
            procOrder[0] = procName;
            theElasticProcess->SetProcessIndex(0);
        }
        else if(procName == "StorkNeutronInelastic")
        {
            theInelasticProcess = dynamic_cast<StorkNeutronInelasticProcess*>((*procVector)[i]);
            procOrder[1] = procName;
            theInelasticProcess->SetProcessIndex(1);
        }
        else if(procName == "StorkHadronFission")
        {
            theFissionProcess = dynamic_cast<StorkHadronFissionProcess*>((*procVector)[i]);
            procOrder[2] = procName;
            theFissionProcess->SetProcessIndex(2);
        }
        else if(procName == "StorkHadronCapture")
        {
            theCaptureProcess = dynamic_cast<StorkHadronCaptureProcess*>((*procVector)[i]);
            procOrder[3] = procName;
            theCaptureProcess->SetProcessIndex(3);
        }
    }
}


// Destructor
StorkProcessManager::~StorkProcessManager()
{
	// Clear dynamic memory
	delete [] procOrder;
	delete [] n_lambda;
}


// GetNumberOfInteractionLengthsLeft()
// Returns the number of interaction lengths left for each process.  Returns
// -1.0 if the process does not exist.
G4double* StorkProcessManager::GetNumberOfInteractionLengthsLeft(const G4double previousStepSize) const
{
	// Initialize the n_lambda array
	for(G4int i=0; i<numProcesses; i++)
		n_lambda[i] = -1.0;

	// Set the data in the n_lambda array. Check to make sure process exists
	if(theElasticProcess)
		n_lambda[0] = theElasticProcess->GetNumberOfInteractionLengthLeft(previousStepSize);

	if(theInelasticProcess)
		n_lambda[1] = theInelasticProcess->GetNumberOfInteractionLengthLeft(previousStepSize);

	if(theFissionProcess)
		n_lambda[2] = theFissionProcess->GetNumberOfInteractionLengthLeft(previousStepSize);

	if(theCaptureProcess)
		n_lambda[3] = theCaptureProcess->GetNumberOfInteractionLengthLeft(previousStepSize);


	return n_lambda;
}


// GetProcessIndex()
// Find the index for the process by name
G4int StorkProcessManager::GetProcessIndex(const G4String procName) const
{
	// Search the process name array for the index (i)
	for(G4int i=0; i<numProcesses; i++)
	{
		if(procName == procOrder[i])
			return i;
	}

	// Return -1 if process not found
	return -1;
}


// GetProcess()
// Get the process by name
G4VProcess* StorkProcessManager::GetProcess(const G4String procName) const
{
	// Search the process name array for the index (i)
	for(G4int i=0; i<numProcesses; i++)
	{
		if(procName == procOrder[i])
		{
			switch(i)
			{
				case 0:
					return theElasticProcess;
				case 1:
					return theInelasticProcess;
				case 2:
					return theFissionProcess;
				case 3:
					return theCaptureProcess;
			}
		}
	}

	// Return NULL if process not found
	return NULL;
}




