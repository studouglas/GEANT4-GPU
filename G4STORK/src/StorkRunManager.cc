/*
StorkRunManager.cc

Created by:		Liam Russell
Date:			22-06-2011
Modified:		09-07-2012

Source code for StorkRunManager class.

*/


// Include header file
#include "StorkRunManager.hh"


// Static pointer to the event manager
G4EventManager* StorkRunManager::EventMan=0;


// Constructor
StorkRunManager::StorkRunManager()
:G4RunManager()
{
    runStart = 0.0;
	frac = 25.*perCent;

	// Initialize flags and values
    interpStarted = false;
    sourceConverged = false;
    convergeStop=0;
    seSelect = NULL;
    nConv = 0;
    heatTransfer = NULL;

}

// Constructor with input file
StorkRunManager::StorkRunManager(const StorkParseInput* infile)
:G4RunManager()
{
    // maybe able to run default Constructor instead
    runStart = 0.0;
	frac = 25.*perCent;

	// Initialize flags and values
    interpStarted = false;
    sourceConverged = false;
    convergeStop=0;
    seSelect = NULL;
    nConv = 0;

    // Set default values
    runDuration = infile->GetRunDuration();
    runEnd = runStart + runDuration;
    numRuns = infile->GetNumberOfRuns();
    saveInterval = infile->SaveSourcesInterval();
    saveFile = infile->GetSourceFile();
	theMPInterpMan = infile->GetNSInterpolationManager();
	interpStartCond = infile->GetInterpStartCond();

	// Fission Energy Deposition Flags and files
	interp = infile->GetInterp();
    RunThermalModel = infile->GetRunThermalModel();
	reactorPower = infile->GetReactorPower();
	saveMatTemp = infile->SaveTemperature();
	matTempFile = infile->GetTemperatureDataFile();

	if(RunThermalModel)
    {
        heatTransfer = new StorkHeatTransfer(infile);
    }

	// Initialize flags and values
    convergenceLimit = infile->GetConvergenceLimit();
    totalConv = infile->GetNumberOfConvergenceRuns();
    runInterpStarted = numRuns;
    propValues = new G4double[theMPInterpMan->GetNumberOfInterpVectors()];
    saveFissionData = infile->SaveFissionData();
    fissionFile = infile->GetFissionDataFile();

	// Set up the run data array and initialize all values to zero
	for(G4int i=0; i < 8; i++)
    {
        runData[i] = new G4double[numRuns];

        for(G4int j=0; j < numRuns; j++)
        {
        	runData[i][j] = 0.0;
        }

        // Initialize the average run data to 0
        avgRunData[i+1] = 0.;
    }

    // Get output stream
	output = infile->GetLogOutputStream();

    // Find the number of characters needed to write maximum run number
    numRunOutputWidth = std::ceil(std::log(numRuns)/std::log(10));
}


// Destructor
StorkRunManager::~StorkRunManager()
{
	// Delete the run data arrays
	for(G4int i=0; i < 8; i++)
    {
        delete [] runData[i];
    }

    // Delete Heat Transfer Class
    if(heatTransfer)
        delete heatTransfer;

    // Delete Shannon entropy array
    if(seSelect) delete [] seSelect;

    // Delete variable property values array
    if(propValues) delete [] propValues;
}

// Data Initializer in case no input file was used during construction
void StorkRunManager::InitializeRunData(const StorkParseInput* infile)
{
    // Set default values
    runDuration = infile->GetRunDuration();
    runEnd = runStart + runDuration;
    numRuns = infile->GetNumberOfRuns();
    saveInterval = infile->SaveSourcesInterval();
    saveFile = infile->GetSourceFile();
	theMPInterpMan = infile->GetNSInterpolationManager();
	interpStartCond = infile->GetInterpStartCond();

	// Fission Energy Deposition Flags and files
	interp = infile->GetInterp();
    RunThermalModel = infile->GetRunThermalModel();
	reactorPower = infile->GetReactorPower();
	saveMatTemp = infile->SaveTemperature();
	matTempFile = infile->GetTemperatureDataFile();

	// Initialize flags and values
    convergenceLimit = infile->GetConvergenceLimit();
    totalConv = infile->GetNumberOfConvergenceRuns();
    runInterpStarted = numRuns;
    propValues = new G4double[theMPInterpMan->GetNumberOfInterpVectors()];
    saveFissionData = infile->SaveFissionData();
    fissionFile = infile->GetFissionDataFile();

    // Set up the run data array and initialize all values to zero
	for(G4int i=0; i < 8; i++)
    {
        runData[i] = new G4double[numRuns];

        for(G4int j=0; j < numRuns; j++)
        {
        	runData[i][j] = 0.0;
        }

        // Initialize the average run data to 0
        avgRunData[i+1] = 0.;
    }

    // Get output stream
	output = infile->GetLogOutputStream();

    // Find the number of characters needed to write maximum run number
    numRunOutputWidth = std::ceil(std::log(numRuns)/std::log(10));
}

void StorkRunManager::InitializeRunData(G4double runDur, G4int numberRuns, G4int numSaveInterval, G4String saveFileName, G4bool interpStartCondition,
                                        const StorkInterpManager* theMPInterpManager, G4double convergenceLim, G4int numConvRuns, G4bool saveFissionDataCond,
                                        G4String fissionDataFile, std::ostream *logOutput, G4bool temperatureTracking,G4double nuclearReactorPower,
                                        G4bool saveTemperature, G4String temperatureDataFile)
{
    // Set default values
    runDuration = runDur;
    runEnd = runStart + runDuration;
    numRuns = numberRuns;
    saveInterval = numSaveInterval;
    saveFile = saveFileName;
	theMPInterpMan = theMPInterpManager;
	interpStartCond = interpStartCondition;

	// Fission Energy Deposition Flags and files
    RunThermalModel = temperatureTracking;
	reactorPower = nuclearReactorPower;
	saveMatTemp = saveTemperature;
	matTempFile = temperatureDataFile;

	// Initialize flags and values
    convergenceLimit = convergenceLim;
    totalConv = numConvRuns;
    runInterpStarted = numRuns;
    propValues = new G4double[theMPInterpMan->GetNumberOfInterpVectors()];
    saveFissionData = saveFissionDataCond;
    fissionFile = fissionDataFile;

    // Set up the run data array and initialize all values to zero
	for(G4int i=0; i < 8; i++)
    {
        runData[i] = new G4double[numRuns];

        for(G4int j=0; j < numRuns; j++)
        {
        	runData[i][j] = 0.0;
        }

        // Initialize the average run data to 0
        avgRunData[i+1] = 0.;
    }

    // Get output stream
	output = logOutput;

    // Find the number of characters needed to write maximum run number
    numRunOutputWidth = std::ceil(std::log(numRuns)/std::log(10));
}

// BeamOn()
// Start the simulation.  Process each run and increment the simulation time.
void StorkRunManager::BeamOn(G4int n_event, const char* macroFile,
                             G4int n_select)
{
    //G4cout << " made it to the beginning of the StorkRunManager::BeamOn" << G4endl;
    G4bool cond = ConfirmBeamOnCondition();
    if(cond)
    {

        InitializeVar(n_event);
        //G4cout << " made it to the past InitializeVar(n_event) in StorkRunManager::BeamOn" << G4endl;
        // Set the number of events in the primary generator action
        genAction->SetNumEvents(n_event);

        if(n_event>0)
        {
            while(runIDCounter < numRuns)
            {
                // Process the run
                RunInitialization();
                DoEventLoop(n_event,macroFile,n_select);
                RunTermination();

                // Record the important results of the run
                TallyRunResults();
                if (sourceConverged)
                    G4cout << G4endl << "#### Souce Has Converged #####" << G4endl;
                else
                    G4cout << G4endl << "#### Souce Has Not Converged #####" << G4endl;
                // Update the source distributions of the primary generator
                runAction->UpdateSourceDistributions();

                runStart += runDuration;
                runEnd += runDuration;


                //Run thermal calculation
                if(RunThermalModel)
                    heatTransfer->RunThermalCalculation(runAction->GetCurrentFissionSites());

                // Save the source distribution if the given interval of runs
                // has passed
                if(saveInterval && !(runIDCounter%saveInterval)){
                    SaveSourceDistribution(saveFile);
                    if(saveFissionData) SaveFissionDistribution(fissionFile);


                }
            }

            // Save the final source distribution if the save interval is not
            // zero and it has not been just saved
            if(saveInterval && runIDCounter%saveInterval){
                SaveSourceDistribution(saveFile);
                if(saveFissionData) SaveFissionDistribution(fissionFile);
            }
        }
   }
}


// DoEventLoop()
// Override G4RunManager::DoEventLoop()
void StorkRunManager::DoEventLoop(G4int n_event, const char* macroFile,
                               G4int n_select)
{
    timer->Start();
    //if(verboseLevel>0)
    //{ timer->Start(); }

    G4String msg;
    if(macroFile!=0)
    {
    if(n_select<0) n_select = n_event;
    msg = "/control/execute ";
    msg += macroFile;
    }
    else
    { n_select = -1; }

    // Initialize the current run
    genAction->InitializeRun();

    // Event loop
    G4int i_event;
    for( i_event=0; i_event<n_event; i_event++ )
    {
        // Set the primaries for the current event
        genAction->SetPrimaries(genAction->GetPrimaryData(i_event));

        currentEvent = GenerateEvent(i_event);
        eventManager->ProcessOneEvent(currentEvent);
        AnalyzeEvent(currentEvent);
        UpdateScoring();

        // Update the run action tallies
        runAction->TallyEvent(eventAction->GetEventData());

        if(i_event<n_select) G4UImanager::GetUIpointer()->ApplyCommand(msg);

        StackPreviousEvent(currentEvent);
        currentEvent = 0;

        if(runAborted) break;
    }

    if(verboseLevel>0)
    {
        timer->Stop();
        G4cout << "Run terminated." << G4endl;
        G4cout << "Run Summary" << G4endl;

        if(runAborted)
        {
            G4cout << "  Run Aborted after " << i_event
                   << " events processed." << G4endl;
        }
        else
        {
            G4cout << "  Number of events processed : " << n_event << G4endl;
        }

        G4cout << "  "  << *timer << G4endl;
    }

    return;
}


// RunInitialization()
// Initialize the current run.  Make any necessary changes to the world and
// save the fission data if necessary.
void StorkRunManager::RunInitialization()
{
	// Update properties only if source has converged or interpolation at
	// start flag is true
	if(sourceConverged || interpStartCond)
	{
		if(!interpStarted)
		{
			timeOffset = runStart;
			runInterpStarted = runIDCounter;
			interpStarted = true;

            /*
			// Since interpolation just started create header and record
			// pre interpolation temperatures
			if(saveMatTemp)
			{
			    worldPointerCD->SaveMaterialTemperatureHeader(matTempFile);
                worldPointerCD->SaveMaterialTemperatures(matTempFile, G4int(runStart/runDuration));
			}
             */
		}

        /*
		// Update the material temperature based on fission sites if
        // RunThermalModel is on
        if(RunThermalModel)
        {
            //MapFissionSitesToMaterial();
            heatTransfer->RunThermalCalculation(runAction->GetCurrentFissionSites());

            // Save the new temperatures only if asked to do so
            if(saveMatTemp)
            {
                worldPointerCD->SaveMaterialTemperatures(matTempFile, G4int(runEnd/runDuration));
            }
        }*/


        // Update the world properties
        UpdateWorld(theMPInterpMan->GetStorkMatPropChanges(runStart -timeOffset));

	}

    // For each variable property, set it in the run action
    for (G4int i=0; i < theMPInterpMan->GetNumberOfInterpVectors(); i++)
    {
    	// Get each property value from the world
    	propValues[i] = worldPointerCD->GetWorldProperty(
									(*theMPInterpMan)[i]->second);

		// Send a pointer to the values to the run action
		runAction->UpdateWorldProperties(propValues);
    }

    // Start collecting fission data if source has converged
    if(saveFissionData && sourceConverged && nConv == runIDCounter)
    {
    	runAction->SaveFissionData(true);
    }

    // Do standard Geant4 initialization tasks
    G4RunManager::RunInitialization();

    return;
}


// UpdateWorld()
// Rebuild the world with new properties.
void StorkRunManager::UpdateWorld(StorkMatPropChangeVector theChanges)
{
    // Create new world volume
    DefineWorldVolume(worldPointerCD->UpdateWorld(theChanges));
    if(worldPointerCD->HasPhysChanged())
    {
        // Inform kernel of change
        PhysicsHasBeenModified();
    }

}


// SaveSources()
// Calls the run actions save sources function.
// Saves survivors and delayed to the given file.
void StorkRunManager::SaveSourceDistribution(G4String fname)
{
    // Find the end of the output file name minus ".txt"
    std::stringstream nameCount;
    G4int pos = fname.find(".txt");

    // Set the fill character to '0'
    nameCount.fill('0');

    nameCount << fname.substr(0,pos) << "-" << std::setw(numRunOutputWidth)
              << runIDCounter << ".txt";

    runAction->SaveSources(nameCount.str(), runIDCounter, runEnd-runDuration);
}

void StorkRunManager::SaveFissionDistribution(G4String name)
{
    // Find the end of the output file name minus ".txt"
    std::stringstream nameCount;
    G4int pos = name.find(".txt");

    // Set the fill character to '0'
    nameCount.fill('0');

    nameCount << name.substr(0,pos) << "-" << std::setw(numRunOutputWidth)
              << runIDCounter << ".txt";

    runAction->WriteFissionData(nameCount.str(), runIDCounter);
}


// TallyRunResults()
// Totals the results of each run after the source has converged
void StorkRunManager::TallyRunResults()
{
	// Get run results from run action
	G4double *currentRunData = runAction->GetRunResults();

	// Copy run results to runData arrays
	for(G4int i=0; i < 8; i++)
	{
		runData[i][runIDCounter-1] = currentRunData[i];
	}

	// Check source convergence
	sourceConverged = UpdateCheckSourceConvergence();
}


// UpdateCheckSourceConvergence()
// Check convergence of the last "totalConv" runs in terms of Shannon entropy.
// For simplicity, only check at intervals of "totalConv" (25 runs default).
// The default convergence limit is 1%.
G4bool StorkRunManager::UpdateCheckSourceConvergence()
{
	// If the source has already converged, do nothing
	if(sourceConverged) return true;
	//changed elseif so that it checks convergence when there is no known discontuinity in the shannon entropy instead of every totalConv runs has passed
	else if((runIDCounter < totalConv) || (convergeStop > (runIDCounter-totalConv)) ) return false;

	// Local variables
	G4int i=0;
	G4double seMean=0.;

	// Clear the seSelect array if full
	if(seSelect) delete [] seSelect;

	// Create a new array
	seSelect = new G4double[totalConv];

	// Add Shannon entropies to the array
	for(i=0; i < totalConv; i++)
	{
		seSelect[i] = runData[6][runIDCounter - totalConv + i];
	}

	// Find the mean of the selected shannon entropy
	for(i=0; i < totalConv; i++)
	{
		seMean += seSelect[i];
	}

	// Divide mean by total
	seMean /= G4double(totalConv);

	// Check whether se values are within the convergence limit of the mean
	for(i=0; i < totalConv; i++)
	{
		if(convergenceLimit < std::abs(seSelect[i] - seMean))
		{
            G4cout << "\nRun " << i << " has a Shannon Entropy of " << seSelect[i] << " which differed from the mean of " << seMean << " beyond the limit of " << convergenceLimit << G4endl;
            convergeStop = runIDCounter - totalConv + i;
			return false;
		}
	}

	// Convergence has been achieved, set convergence flag
	//sourceConverged = true;
	nConv = runIDCounter;
	return true;
}


// OutputConvergence()
// Output the results of the run including convergence to an output stream
void StorkRunManager::OutputResults()
{
    if(sourceConverged)
    {
    	// Find the average run data
		AverageRunResults();

        // Output run results
        output->precision(6);
        output->fill(' ');
        *output << G4endl
                << std::right
                << "# Avg (last " << std::setw(5) << G4int(avgRunData[0])
                << " runs):"
                << std::setw(16) << std::setprecision(4) << std::setw(12)
                << avgRunData[1] << " "
                << std::setw(12) << G4int(avgRunData[2]) << " "
				<< std::setw(12) << G4int(avgRunData[3]) << " "
                << std::setw(12) << std::setprecision(6) << std::fixed
                << avgRunData[4] << " "
                << std::setw(12) << std::setprecision(6) << avgRunData[5] << " "
                << std::setw(12) << std::setprecision(4) << avgRunData[6] << " "
                << std::setw(12) << std::setprecision(4) << avgRunData[7] << " "
                << std::setw(12) << std::setprecision(2)
                << std::resetiosflags(std::ios_base::floatfield)
                << avgRunData[8]
                << G4endl;
    }

	// Output start of interpolation (if used)
	if(interpStarted)
	{
		*output << "# Interpolation started at run " << runInterpStarted
				<< G4endl;
	}

	// Output the convergence limit in %
	*output << G4endl
			<< "# Source convergence limit = " << convergenceLimit << "%"
	        << G4endl;

	// Report the results of convergence and the number of runs taken to reach
	// this result.
	if(!sourceConverged)
	{
		*output << "# Source convergence not achieved after " << numRuns
		        << " runs." << G4endl;
	}
	else
	{
		*output << "# Source converged after " << nConv << " runs." << G4endl;
	}

	/*// Write fission data to file if necessary
	if(saveFissionData)
		runAction->WriteFissionData(fissionFile, nConv);*/
}


// AverageRunResults()
// Averages a fraction of the total run results.
void StorkRunManager::AverageRunResults()
{
	// Find the number of runs to average over
	avgRunData[0] = std::floor(frac * G4double(numRuns-nConv));
	G4int i = numRuns - G4int(avgRunData[0]);

	// Sum the run properties over these runs
	for( ; i < numRuns; i++)
	{
		for(G4int j=0; j < 8; j++)
		{
			avgRunData[j+1] += runData[j][i];
		}
	}

	// Divide by number of runs
	for(G4int j=0; j < 8; j++)
	{
		avgRunData[j+1] /= avgRunData[0];
	}
}


//GetWorldProperty()
//returns the material property requested in the input
G4double StorkRunManager::GetWorldProperty(MatPropPair matProp)
{
    G4double val = worldPointerCD->GetWorldProperty(matProp);

    return val;
}


// InitializeVar()
// Initializes variables at the start of the beam on function
void StorkRunManager::InitializeVar(G4int n_event)
{
    // Reset the current run counter and run start
    runIDCounter = 0;
    runStart = 0.;
    timeOffset = 0.;
    runEnd = runStart + runDuration;

    // Set the pointers to the NS versions of the user action classes
    genAction = dynamic_cast<StorkPrimaryGeneratorAction*>
                            (userPrimaryGeneratorAction);
    runAction = dynamic_cast<StorkRunAction*>(userRunAction);
    eventAction = dynamic_cast<StorkEventAction*>(userEventAction);
    worldPointerCD = dynamic_cast<StorkWorld*>(userDetector);

    if(RunThermalModel)
        heatTransfer->SetWorld(worldPointerCD);

    numberOfEventToBeProcessed = n_event;
}
