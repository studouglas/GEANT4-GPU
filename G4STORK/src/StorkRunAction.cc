/*
StorkRunAction.cc

Created by:		Liam Russell
Date:			22-06-2011
Modified:		11-03-2013

Source code for StorkRunAction class.

*/


// Include header file
#include "StorkRunAction.hh"


// Constructor
StorkRunAction::StorkRunAction(StorkPrimaryGeneratorAction *genPtr,
                        const StorkParseInput *fIn)
:genAction(genPtr)
{
    // Set default and user input values
    infile = fIn;
    runDuration = fIn->GetRunDuration();
    kCalcType = fIn->GetKCalcType();
    RunThermalModel = fIn->GetRunThermalModel();
    worldSize = G4ThreeVector(0.,0.,0.);
    survivorsBuffer = NULL;
    delayedBuffer = NULL;
    rndmInitializer = fIn->GetInitialSourceFile();
    maxShannonEntropy = 1.0;
    theMPInterpMan = fIn->GetNSInterpolationManager();
    nameLen = new G4int[theMPInterpMan->GetNumberOfInterpVectors()];
    variableProps = NULL;
    saveFissionData = false;
    numDelayProd = 0;
    numFDCollectionRuns = 0;
    numRuns = fIn->GetNumberOfRuns();
    reactorPower = fIn->GetReactorPower();
    updatePrecursors = fIn->GetPrecursorDelayed();
    neutronFluxCalc = fIn->GetNeutronFluxCalc();
    renormalize = fIn->GetRenormalizeAfterRun();
    saveInterval = fIn->SaveSourcesInterval();
    primariesPerRun = fIn->GetNumberOfEvents()*fIn->GetNumberOfPrimariesPerEvent();
    saveRundata = false;

    fn_index = save_index = 0;
    Origin = fIn->GetOrigin();

    //Set up the neutron flux calcution
    if(neutronFluxCalc){
        fluxCalcShape = fIn->GetFluxCalcShape();

        fluxCalcRegion[0] = (fIn->GetFluxRegion())[0];
        fluxCalcRegion[1] = (fIn->GetFluxRegion())[1];
        fluxCalcRegion[2] = (fIn->GetFluxRegion())[2];
        fluxCalcRegion[3] = (fIn->GetFluxRegion())[3];

        EnergyRange[0]=(fIn->GetEnergyRange())[0];
        EnergyRange[1]=(fIn->GetEnergyRange())[1];
    }


	// Set the shannon entropy mesh
    for(G4int i=0; i<3; i++)
    {
    	numSpatialSteps[i] = (fIn->GetMeshSteps())[i];
    	maxShannonEntropy *= numSpatialSteps[i];
    }

    // Calculate maximum Shannon entropy
    maxShannonEntropy = -1.0 * std::log(1.0/maxShannonEntropy) / std::log(2);

    // Create the fSites and sSites arrays
    fSites = new G4int **[numSpatialSteps[0]];
    sSites = new G4int **[numSpatialSteps[0]];

    for(G4int i=0; i<numSpatialSteps[0]; i++)
    {
        fSites[i] = new G4int *[numSpatialSteps[1]];
        sSites[i] = new G4int *[numSpatialSteps[1]];

        for(G4int j=0; j<numSpatialSteps[1]; j++)
        {
            fSites[i][j] = new G4int [numSpatialSteps[2]];
            sSites[i][j] = new G4int [numSpatialSteps[2]];
        }
    }

    // Set up the output stream
	output = infile->GetLogOutputStream();
}


// Destructor
StorkRunAction::~StorkRunAction()
{
    // Delete all elements of the fSites and sSites arrays
    for(G4int i=0; i<numSpatialSteps[0]; i++)
    {
        for(G4int j=0; j<numSpatialSteps[1]; j++)
        {
            delete [] fSites[i][j];
            delete [] sSites[i][j];
        }

        delete [] fSites[i];
        delete [] sSites[i];
    }

    delete [] fSites;
    delete [] sSites;
    fSites = NULL;
    sSites = NULL;

    // Delete all the elements of the survivorsBuffer and delayedBuffer
    if(survivorsBuffer)
    {
    	delete [] survivorsBuffer;
		survivorsBuffer = NULL;
    }
    if(delayedBuffer)
    {
    	delete [] delayedBuffer;
		delayedBuffer = NULL;
    }

    // Delete array of variable property name lengths
    delete [] nameLen;
    nameLen = NULL;

}


// BeginOfRunAction()
// Initialize (clear and reset) containers and set random number generator seed
// if starting from saved source.  On the first run, print the headers for the
// output data.
void StorkRunAction::BeginOfRunAction(const G4Run *aRun)
{
    G4int underline;
    // Start the timer
    runTimer.Start();

#ifdef G4TIMERA
    runCalcTimer.Start();
#endif

	// Initialize random number generator from file if necessary
	if(!rndmInitializer.empty() && aRun->GetRunID() == 0 &&
		!infile->OverrideInitRandomSeed())
	{
		// Restore engine from file
		CLHEP::HepRandom::restoreEngineStatus(rndmInitializer);
	}

    // Get the number of events and initialize the run
    numEvents = aRun->GetNumberOfEventToBeProcessed();

    // Clear the container vectors
    survivors.clear();
    delayed.clear();

    // Delete and recreate the survivors buffer
    if(survivorsBuffer)
        delete [] survivorsBuffer;
    survivorsBuffer = new NeutronSources[numEvents];

    // Delete and recreate the delayed buffer
    if(delayedBuffer)
        delete [] delayedBuffer;
    delayedBuffer = new NeutronSources[numEvents];

    // Reset the entries of the fsites array
    for(G4int i=0; i<numSpatialSteps[0]; i++)
    {
        for(G4int j=0; j<numSpatialSteps[1]; j++)
        {
            for(G4int k=0; k<numSpatialSteps[2]; k++)
            {
            	fSites[i][j][k] = 0;
                sSites[i][j][k] = 0;
            }

        }
    }

    // Initialize variables/counters
    totalFS = 0;
    avgLifetime = 0.0;
    numNProduced = 0;
    numNLost = 0;

    runID = aRun->GetRunID();


    if(runID == 0)
    {
        // Set up the appropriate unit for output
        tUnit = 1.*ns;
        G4String unit = "(ns)";

        if(runDuration >= 1e9)
        {
            tUnit = 1.*s;
            unit = "(s) ";
        }
        else if(runDuration >= 1e6)
        {
            tUnit = 1.*ms;
            unit = "(ms)";
        }
        else if(runDuration >= 1e3)
        {
            tUnit = 1.*microsecond;
            unit = "(us)";
        }


        // Write header lines
        *output << "# Number of Primaries per Run = "
                << genAction->GetNumPrimaries() << G4endl
                << "# Number of Events per Run = "
                << numEvents << G4endl
                << "# " << G4endl << "# " << G4endl;


        *output << "# Run #      "
                << " Start " << unit << "  "
                << "Lifetime " << unit
                << " Production  "
                << "    Loss     "
                << "    krun     "
                << "    keff     "
                << "FS Shannon H "
                << " S Shannon H "
                << "Duration (s) ";
        if(neutronFluxCalc){
            *output << "Flux (n cm^-2 s^-1) ";
            underline = 150;

        }
		else{
        	underline = 129;
		}
        G4String interpName;

		// Output the material properties that are being interpolated
        for(G4int i=0; i<theMPInterpMan->GetNumberOfInterpVectors(); i++)
        {
            interpName = (*theMPInterpMan)[i]->first;

            *output << interpName << " ";

            // Set length of each name
            nameLen[i] = interpName.size() + 1;
            underline += nameLen[i];
        }

        *output << G4endl;
        *output << "#";

        output->width(underline);
        output->fill('-');
        *output << "-" << G4endl;

    }

    //Set the saving index for fission site/energies.
    if(saveInterval && !(runID%saveInterval)){
        save_index = fnSites.size();
        saveRundata = true;
    }
    else
        saveRundata = false;

#ifdef G4TIMERA
    runCalcTimer.Stop();
    totalRunCalcTime = runCalcTimer.GetRealElapsed();
#endif
}


// UpdateSourceDistributions()
// Update the survivor and delayed neutron distributions of the
// StorkPrimaryGeneratorAction class.
void StorkRunAction::UpdateSourceDistributions()
{
    // Pass suvivors and delayed neutrons to primary generator
    genAction->UpdateSourceDistributions(&survivors,&delayed);
}


// EndOfRunAction()
// Calculate the results of the run (k_eff, k_run, Shannon entropy, etc).
// Output results to the screen and to the StorkRunManager.
void StorkRunAction::EndOfRunAction(const G4Run *aRun)
{
#ifdef G4TIMERA
    runCalcTimer.Start();
#endif

    // Collate the survivors and delayed neutrons
    CollateNeutronSources();

    //G4int numPrimaries = genAction->GetNumPrimaries();

    // Find the Shannon Entropy
    shannonEntropy[0] = CalcShannonEntropy(fSites,totalFS);
    shannonEntropy[1] = CalcShannonEntropy(sSites,G4int(survivors.size()));

    //Find neutron flux of specified material.
    if(neutronFluxCalc)
        neutronFlux = CalcNeutronFlux();

    //Find krun
    krun = G4double(survivors.size()/G4double(primariesPerRun));

    // Find the average neutron lifetime
    avgLifetime /= G4double(numNLost);

    // Correct the production total for the delayed primary neutrons
    // the delay neutrons are already added on to the total production in StorkNeutronSD
    //numNProduced += genAction->GetNumDNPrimaries();

    // Find keff

    // Dynamic Criticality Method
    if(kCalcType == 0)
    {
        keff = G4double(numNProduced) / G4double(numNLost);
    }
    // Time Absorption Method
    else if(kCalcType == 1)
    {
        G4double alpha = (std::log(krun))/runDuration;
        keff = std::exp(alpha*avgLifetime);
    }
    // Generational Method
    else if(kCalcType == 2)
    {
        keff = G4double(numNProduced) / G4double(numNLost);
    }
    else
    {
        G4cout << "Error: Invalid Keff calculation method selected!" << G4endl;
        keff=0.;
    }

    //Reset the current fission sites.
    ResetCurrentFissionData();

    //Add precursors if flag is set.
    if(updatePrecursors)
    {
        genAction->AddPrecursors(GetCurrentFissionSites(),GetCurrentFissionEnergy());
    }

    // Find the run time
    runTimer.Stop();


    // Output data to the output stream (csv format)
    output->fill(' ');
    *output << std::setw(6) << std::right << aRun->GetRunID()+1 << " "
            << std::right << std::resetiosflags(std::ios_base::floatfield)
            << std::setw(16) << (aRun->GetRunID()) * runDuration / tUnit << " "
            << std::setw(12) << std::setprecision(4)
            << avgLifetime / tUnit << " "
            << std::setw(12) << numNProduced << " "
            << std::setw(12) << numNLost << " "
            << std::setw(12) << std::setprecision(4)
            << std::fixed << krun << " "
            << std::setw(12) << std::fixed  << keff << " "
            << std::setw(12) << std::setprecision(4) << std::fixed
            << shannonEntropy[0] << " "
            << std::setw(12) << std::setprecision(4) << std::fixed
            << shannonEntropy[1] << " "
            << std::resetiosflags(std::ios_base::floatfield)
            << std::setw(12) << runTimer.GetRealElapsed();
    if(neutronFluxCalc){
        *output << std::setw(12) << std::setprecision(4)
                << neutronFlux;
    }

	// Output the current value of the interpolated material properties
    for(G4int i=0; i<theMPInterpMan->GetNumberOfInterpVectors(); i++)
    {
        *output << std::setw(nameLen[i]) << variableProps[i];
    }

    *output << G4endl;

    // Assign run results
    runResults[0] = avgLifetime / tUnit;
    runResults[1] = G4double(numNProduced);
    runResults[2] = G4double(numNLost);
    runResults[3] = krun;
    runResults[4] = keff;
    runResults[5] = shannonEntropy[0];
    runResults[6] = shannonEntropy[1];
    runResults[7] = runTimer.GetRealElapsed();

    // Increment fission data collection counter
    if(saveFissionData)
		numFDCollectionRuns++;

#ifdef G4TIMERA
    runCalcTimer.Stop();
    totalRunCalcTime += runCalcTimer.GetRealElapsed();


    G4cout << "Run Timing:" << G4endl
           << "Total run time:  " << runTimer << G4endl
           << "Total run calculation time = " << totalRunCalcTime << "s"
           << G4endl;
#endif
}


// CalcShannonEntropy()
// Calculate the Shannon entropy of the given sites.
G4double StorkRunAction::CalcShannonEntropy(G4int ***sites, G4int total)
{
    G4double Pijk;
    G4double entropy = 0.0;

    // Find the Shannon entropy
    for(G4int i=0; i<numSpatialSteps[0]; i++)
    {
        for(G4int j=0; j<numSpatialSteps[1]; j++)
        {
            for(G4int k=0; k<numSpatialSteps[2]; k++)
            {
                // Find the probability of the ijk-th site having a fission
                Pijk = G4double(sites[i][j][k]) / G4double(total);

                if(Pijk)
                {
                    // Add the entropy of the ijk-th site
                    entropy += Pijk * std::log(Pijk) / std::log(2);
                }
            }
        }
    }

    // Convert the entropy to a percentage of the maximum entropy
    entropy *= -100.0/maxShannonEntropy;

    return entropy;
}


// TallyEvent()
// Add the event data to the running tallies and containers.
void StorkRunAction::TallyEvent(const StorkEventData *eventData)
{
#ifdef G4TIMERA
    // Start the calculation timer
    runCalcTimer.Start();
#endif

    // Local variable
    Index3D fInd;

    // Put the survivors and delayed into the appropriate slot of their buffer
    survivorsBuffer[eventData->eventNum] = *(eventData->survivors);
    delayedBuffer[eventData->eventNum] = *(eventData->delayed);

    // Add the lifetime to the total (average later)
    avgLifetime += eventData->totalLifetime;

    // Add the neutron production and loss
    numNProduced += eventData->numNProd;
    numNLost += eventData->numNLost;

    // Add the fission sites to the 3D array
    numSites = eventData->fSites->size();
    totalFS += numSites;

    // Discretize fission sites
    for(G4int i=0; i<numSites; i++)
    {
        fInd = SpatialIndex(((*(eventData->fSites))[i]).data);
        fSites[fInd.first][fInd.second][fInd.third]++;
    }

    // Discretize survivor positions
    for(G4int i=0; i<G4int(eventData->survivors->size()); i++)
    {
    	fInd = SpatialIndex(((*(eventData->survivors))[i]).third);
    	sSites[fInd.first][fInd.second][fInd.third]++;
    }

    // Save fission data if necessary
    if(saveFissionData || updatePrecursors)
    {
    	if(saveFissionData) numDelayProd += eventData->numDProd;

    	fnEnergies.insert(fnEnergies.end(),eventData->fnEnergy->begin(),
						  eventData->fnEnergy->end());
        fnSites.insert(fnSites.end(),eventData->fSites->begin(),
					   eventData->fSites->end());
    }

#ifdef G4TIMERA
    // Add the tally event time to the calulation total
    runCalcTimer.Stop();
    totalRunCalcTime += runCalcTimer.GetRealElapsed();
#endif
}


// CollateNeutronSources()
// Combine the neutron source buffers into a single list for each buffer
void StorkRunAction::CollateNeutronSources()
{
    // Add the sources in order to their respective containers
    for(G4int i=0; i < numEvents; i++)
    {
        // Add survivors and delayed for ith event
        survivors.insert(survivors.end(),(survivorsBuffer[i]).begin(),
                         (survivorsBuffer[i]).end());
        delayed.insert(delayed.end(),(delayedBuffer[i]).begin(),
                       (delayedBuffer[i]).end());
    }

}


// SpatialIndex()
// Find the spatial index of a fission site based on the given world
// size and number of steps in the mesh used to divide the world
Index3D StorkRunAction::SpatialIndex(G4ThreeVector position) const
{
    Index3D site;

    site.first = G4int(numSpatialSteps[0] * (position[0]/worldSize[0] + 0.5));
    site.second = G4int(numSpatialSteps[1] * (position[1]/worldSize[1] + 0.5));
    site.third = G4int(numSpatialSteps[2] * (position[2]/worldSize[2] + 0.5));

    // Check to make sure index is in range
    if(site.first < 0 || site.first >= numSpatialSteps[0] ||
		site.second < 0 || site.second >= numSpatialSteps[1] ||
		site.third < 0 || site.third >= numSpatialSteps[2])
	{
		G4cerr << "*** ERROR:  Improper indexing in Shannon entropy mesh."
		       << G4endl << "World box dimensions = " << worldSize << G4endl
		       << "\t Current position = " << position << G4endl
		       << "\t Calculated index = (" << site.first << ","
		       << site.second << "," << site.third << ")  ***" << G4endl;
	}

    return site;
}


// SetWorldProperty()
// Sets the requested material property in the worldPropMap to the desired value
void StorkRunAction::UpdateWorldProperties(G4double *values)
{
    variableProps = values;
}


// SaveSources()
// Saves the survivors and delayed to a text file with the following format
// FILE FORMAT: each on a new line(s)
// a) header lines (start with #)
// b) current time of records
// c) number of survivors
// d) survivors (global time, lifetime, position (x,y,z), momentum (px,py,pz))
// e) number of delayed
// f) delayed (global time, lifetime, position (x,y,z), momentum (px,py,pz))
void StorkRunAction::SaveSources(G4String fname, G4int curRunID, G4double runEnd)
{
#ifdef G4TIMERA
    // Set up timer for writing sources to file
    G4Timer sourceTimer;
    sourceTimer.Start();
#endif

    runID = curRunID;
	// Print the current state of the random engine
	CLHEP::HepRandom::saveEngineStatus(fname);

	// User variables
    G4int numEntries;
    StorkNeutronData *record;

	// Open stream to output file (append)
	G4String FDir = fname.substr(0, fname.find_last_of('/'));
	if(!(DirectoryExists(FDir.data())))
    {
        system( ("mkdir -p -m=666 "+FDir).c_str());
        if(DirectoryExists(FDir.data()))
        {
            G4cout << "created directory " << FDir << "\n" << G4endl;
        }
        else
        {
            G4cout << "\nError: could not create directory " << FDir << "\n" << G4endl;
            return;
        }
    }
    std::ofstream outFile(fname.data(),std::ios_base::app);

    if(!outFile.is_open())
    {
        *output << G4endl << "Error:  Could not write source to file."
                << "Improper file name " << fname << G4endl;
        return;
    }

    // Write header lines
    outFile << "# Source file for following input:" << G4endl << "#" << G4endl;

    infile->PrintInput(&outFile);

    outFile << "#" << G4endl;

    // Write time of records and number of runs
    outFile << "# Source distribution after " << runID  << " runs" << G4endl
            << "#" << G4endl << runEnd << G4endl;

    // Write number of survivors then each survivors
    numEntries = survivors.size();
    outFile << numEntries << G4endl;

    outFile.fill(' ');

    for(G4int i=0; i<numEntries; i++)
    {
        record = &(survivors[i]);
        outFile << std::resetiosflags(std::ios_base::floatfield)
                << record->first << " "
                << std::right << std::setprecision(16) << std::scientific
                << std::setw(25) << record->second << " "
                << std::setw(25) << record->third[0] << " "
                << std::setw(25) << record->third[1] << " "
                << std::setw(25) << record->third[2] << " "
                << std::setw(25) << record->fourth[0] << " "
                << std::setw(25) << record->fourth[1] << " "
                << std::setw(25) << record->fourth[2] << " "
                << std::setw(25) << record->fifth << " "
                << std::setw(25) << record->sixth << " "
                << std::setw(25) << record->seventh << " "
                << std::setw(25) << record->eigth << " "
                << std::setw(25) << record->ninth << G4endl;
    }


    // Write the number of delayed and then each delayed neutron record
    numEntries = delayed.size();
    outFile << numEntries << G4endl;

    for(G4int i=0; i<numEntries; i++)
    {
        record = &(delayed[i]);
        outFile << std::resetiosflags(std::ios_base::floatfield)
                << std::scientific << std::setprecision(16)
                << std::setw(25) << record->first << " "
                << std::right << std::setprecision(16)
                << std::setw(25) << record->second << " "
                << std::setw(25) << record->third[0] << " "
                << std::setw(25) << record->third[1] << " "
                << std::setw(25) << record->third[2] << " "
                << std::setw(25) << record->fourth[0] << " "
                << std::setw(25) << record->fourth[1] << " "
                << std::setw(25) << record->fourth[2] << " "
                << std::setw(25) << record->fifth << " "
                << std::setw(25) << record->sixth << " "
                << std::setw(25) << record->seventh << " "
                << std::setw(25) << record->eigth << " "
                << std::setw(25) << record->ninth << G4endl;
    }

    outFile.close();

#ifdef G4TIMERA
    sourceTimer.Stop();
    G4cout << G4endl << "Source writing time:  " << sourceTimer << G4endl;
#endif
}


// WriteFissionData()
// Write fission sites and energies to file
G4bool StorkRunAction::WriteFissionData(G4String fname,
									 G4int start)
{
	// Declare and open file stream
	G4String FDir = fname.substr(0, fname.find_last_of('/'));
	if(!(DirectoryExists(FDir.data())))
    {
        system( ("mkdir -p -m=666 "+FDir).c_str());
        if(DirectoryExists(FDir.data()))
        {
            G4cout << "created directory " << FDir << "\n" << G4endl;
        }
        else
        {
            G4cout << "\nError: could not create directory " << FDir << "\n" << G4endl;
            return false;
        }
    }
	std::ofstream outFile(fname.c_str(),std::ifstream::out);

	// Check that stream is ready for use
	if(!outFile.good())
	{
		G4cerr << G4endl << "ERROR:  Could not write source to file. "
			   << "Improper file name: " << fname << G4endl;

		return false;
	}

	// Check if fission data vectors are the same size
	if(G4int(fnEnergies.size()) != G4int(fnSites.size()))
	{
		G4cerr << G4endl << "ERROR:  Different numbers of fission sites and "
		       << "fission energies: " << G4int(fnSites.size()) << "/"
		       << G4int(fnEnergies.size()) << G4endl;

		return false;
	}


	// Write header lines
    outFile << "# Fission data file for following input:" << G4endl
            << "#" << G4endl;

    infile->PrintInput(&outFile);

    // Write time of records and number of runs
    outFile << "#" << G4endl << "# Fission data collection started at run "
            << start << G4endl << "#" << G4endl;

    //Get the collection interval.
    G4int collectionInt = numRuns-start;
    if(saveInterval)
        collectionInt = saveInterval;

    if(updatePrecursors)
    {
        //Get the precursor numbers
        std::vector<G4int> precursors = genAction->GetPrecursors();

        //Write the precursor data
        outFile << "# Precursor Groups: " << G4endl;

        for(G4int i = 0; i<6; i++){
            outFile << std::setw(12) << precursors[i] << G4endl;
        }
    }
    // Write number of fission data points, the run duration, the current run and primaries simulated per run.
    G4int numEntries = G4int(fnEnergies.size());
    outFile << numEntries << G4endl << runDuration << G4endl << collectionInt << G4endl
            << primariesPerRun << G4endl;

    outFile.fill(' ');
    for(G4int i = save_index; i<numEntries; i++)
    {

        outFile << std::resetiosflags(std::ios_base::floatfield) << std::right
                << std::setprecision(5) << std::scientific
                << std::setw(12) << fnSites[i][0] << " "
                << std::setw(12) << fnSites[i][1] << " "
                << std::setw(12) << fnSites[i][2] << " "
                << std::setw(12) << fnEnergies[i] << G4endl;
    }


    outFile.close();


    return true;
}

G4bool StorkRunAction::DirectoryExists( const char* pzPath )
{
    if ( pzPath == NULL) return false;

    DIR *pDir;
    G4bool bExists = false;

    pDir = opendir (pzPath);

    if (pDir != NULL)
    {
        bExists = true;
        closedir (pDir);
    }

    return bExists;
}

G4double StorkRunAction::CalcNeutronFlux(){

    //Initialize known variables;
    G4int NumEntries = survivors.size();
    G4double Mass = G4Neutron::Neutron()->GetPDGMass();

    //In units of c (speed of light)
    G4double lowVelocity_cutoff = sqrt(2*EnergyRange[0]/Mass);
    G4double highVelocity_cutoff = sqrt(2*EnergyRange[1]/Mass);

    //Initialize variables for neutron flux calculation.
    G4double volume;
    G4double speed;
    G4double total = 0.0;
    G4int tally;
    G4int CalcType;
    G4double x, y, z;
    G4bool inRegion;
    G4double innerR=-1, outerR=-1, minX=-1, maxX=-1, minY=-1, maxY=-1, minZ=-1, maxZ=-1;

    //Get region of interest and volume. This is relative to current set origin (default = 0,0,0).
    if(fluxCalcShape == "Cylinder"){
        innerR = fluxCalcRegion[0];
        outerR = fluxCalcRegion[1];
        minZ = fluxCalcRegion[2];
        maxZ = fluxCalcRegion[3];

        volume = CLHEP::pi*(maxZ-minZ)*( pow(outerR,2) - pow(innerR,2) );
        CalcType = 0;


    }
    else if(fluxCalcShape == "Cube"){
        minX = Origin[0]-fluxCalcRegion[0]/2;
        maxX = Origin[0]+fluxCalcRegion[0]/2;
        minY = Origin[1]-fluxCalcRegion[1]/2;
        maxY = Origin[1]+fluxCalcRegion[1]/2;
        minZ = Origin[2]-fluxCalcRegion[2]/2;
        maxZ = Origin[2]+fluxCalcRegion[2]/2;

        volume = fluxCalcRegion[0]*fluxCalcRegion[1]*fluxCalcRegion[2];
        CalcType = 1;
    }
    else if(fluxCalcShape == "Sphere"){
        innerR = fluxCalcRegion[0];
        outerR = fluxCalcRegion[1];

        volume = (4/3)*CLHEP::pi*(pow(outerR,3) - pow(innerR,3) );
        CalcType = 2;
    }
    else{
        G4cerr << "*** ERROR: Shape not properly specified for flux calculation." << G4endl;
        return 0;
    }





    //Calculate the simulated power in W.
    G4double simulatedPower = numSites*198/(6.24150934*pow(10,12)*runDuration*pow(10,-9));

    //Run through all survivors, check if they are in the material of interest and calculate fluence.
    for(G4int i=0; i<NumEntries; i++){

            //Reorient positions according to set origin.
        x = survivors[i].third[0]-Origin[0];
        y = survivors[i].third[1]-Origin[1];
        z = survivors[i].third[2]-Origin[2];

        inRegion = false;

        if(CalcType==0){
            G4double radius = pow( (pow(x,2) + pow(y,2)) ,0.5 );
            if( radius>innerR && radius<outerR && z>minZ && z<maxZ)
                inRegion = true;
        }
        else if(CalcType==1){
            if(x>minX && x<maxX && y>minY && y<maxY && z>minZ && z<maxZ)
                inRegion = true;
        }
        else if(CalcType==2){
            G4double radius = pow( (pow(x,2) + pow(y,2) + pow(z,2) ) ,0.5);
            if( radius>innerR && radius<outerR)
                inRegion = true;
        }

        //Get the speed from the momentum of the neutron.
        speed = sqrt(survivors[i].fourth[0]*survivors[i].fourth[0] + survivors[i].fourth[1]*survivors[i].fourth[1] +
            survivors[i].fourth[2]*survivors[i].fourth[2])/Mass;

        //Collect the sum if within the specified energy range and in the correct region.
        if( speed > lowVelocity_cutoff && speed < highVelocity_cutoff && inRegion){
            total += speed;
            tally++;
        }
    }

    G4double simulatedFlux = 299.792458*total/(volume);
    G4double currentPower = reactorPower;

    //Calculate actual flux, convert to cm^(-2)s^(-1) from mm^(-2)ns^(-1)
    G4double actualFlux = pow(10,11)*simulatedFlux*currentPower/simulatedPower;

    return actualFlux;

}

void StorkRunAction::ResetCurrentFissionData(){

    CurrentfnSites.clear();
    CurrentfnEnergy.clear();

    CurrentfnSites.insert(CurrentfnSites.end(),fnSites.begin()+fn_index,fnSites.end());
    CurrentfnEnergy.insert(CurrentfnEnergy.end(),fnEnergies.begin()+fn_index,fnEnergies.end());

    fn_index = fnSites.size();

    return;
}
