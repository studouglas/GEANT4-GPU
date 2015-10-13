/*
StorkDelayedNeutron Class

Written By: Andrew Tan
Date: August 25, 2015

Last Edit:
*/

#include "StorkDelayedNeutron.hh"

StorkDelayedNeutron::StorkDelayedNeutron(G4String dnFilename,G4double runduration, G4int numPrimaries)
{
    //Set the fission file name.
    delayedSourceFile = dnFilename;

    //Get the run Duration
    runDuration = runduration;

    //Get neutron particle object
    G4ParticleTable* pTable = G4ParticleTable::GetParticleTable();
    neutron = pTable->FindParticle("neutron");

    //Setup initial precursor population
    GetInitialPrecursors(numPrimaries);


}

StorkDelayedNeutron::~StorkDelayedNeutron()
{

}

G4bool StorkDelayedNeutron::GetInitialPrecursors(G4int numPrimaries)
{
	// Local variables
	char line[256];
	G4int numRuns;
    G4int numEntries;
	G4double primariesPerRun;
    G4double runTime;
    G4double adjustmentFactor;
    G4bool createPrecursors = true;
    Precursors.resize(6);


	// Load data from file
	std::ifstream dnfile(delayedSourceFile);

	// Check if file opened properly
	if(!dnfile.good())
	{
		G4cerr << "*** WARNING:  Unable to open initial delayed neutron file:"
        << delayedSourceFile << G4endl;
		return false;
	}

	// Skip header lines
	while(dnfile.peek() == '#')
		dnfile.getline(line,256);

    //Read in precursor data if nonzero.
    for(G4int i = 0; i<6; i++){
        dnfile >> Precursors[i];
        if(Precursors[i]!=0) createPrecursors = false;
    }


	// Read in delayed neutron distribution parameters
	// Number of entries, runtime, number of runs collected over, primaries per run.
	dnfile >> numEntries >> runTime >> numRuns >> primariesPerRun ;


    // Resize the fission vectors.
    fSites.resize(numEntries);
    fEnergy.resize(numEntries);


	// Read in fission data.
	for(G4int i=0; i<numEntries && dnfile.good(); i++)
	{

		dnfile >> fSites[i][0] >> fSites[i][1] >> fSites[i][2] >> fEnergy[i];

	}

    //Create the precursors if needed.
    if(createPrecursors){
        //Total time of previous simulation.
        G4double totalTime = pow(10,-9)*numRuns*runTime;
        //Iterate through fission sites and energies to create precursors.
        // Will need to change this method. Precursor populations should be directly calculated.

        G4int totalPrecursors = G4int(numEntries*TotPrecursorConstants/totalTime);

        for(G4int i=0; i<totalPrecursors; i++){

            // For now assume that U235 is the only fuel, uncomment or edit if you want to factor in more types of fuel.
            //Can also edit the StorkDelayedData.hh to change delay constants and fission yields.

            //index = fissionIndex(fEnergy[i],fSites[i]);

            G4double r = G4UniformRand()*TotPrecursorConstants;

            G4double temp = 0.0;

            for(G4int j=0;j<6;j++){

                temp += PrecursorConstants[j];

                if( r < temp){
                    Precursors[j]++;
                    break;
                }
            }
        }

    }



    //Adjustment factor for differing number of primaries.
    adjustmentFactor = numPrimaries/G4double(primariesPerRun);
    //Rescale the groups.
    for(G4int i=0; i<6;i++){
        Precursors[i] = G4int(adjustmentFactor*Precursors[i]);
    }

    return true;
}

void StorkDelayedNeutron::SetFissionSource(MSHSiteVector fissionSites, DblVector fissionEnergies)
{

    fSites.clear();
    fEnergy.clear();

    fSites.insert(fSites.end(),fissionSites.begin(),fissionSites.end());
    fEnergy.insert(fEnergy.end(),fissionEnergies.begin(),fissionEnergies.end());

    return;
}

void StorkDelayedNeutron::AddPrecursors()
{
    G4int entries = fSites.size();
    G4int index;
    G4double totalYield;

    for(G4int i = 0; i < entries; i++){

        // For now assume that U235 is the only fuel, uncomment or edit if you want to factor in more types of fissionable isotopes.
        //Can also edit the StorkDelayedData.hh to change delay constants and fission yields.

        //index = fissionIndex(fEnergy[i],fSites[i]);

        index = 0;


        //Get the total yield.
        totalYield = TotalYields[index];

        //Roll the dice to determine if a precursor is created.
        if(G4UniformRand() < totalYield){

            //Initialize temporary variable (cumulative yield) to determine which precursor.
            G4double temp = 0.0;
            //Random number for precursor group.
            G4double rand = G4UniformRand()*totalYield;

            //Iterate through the precursor yields.
            for(G4int j = 0; j<6; j++){
                temp = temp + FissionYields[index][j];
                if(rand < temp){
                    Precursors[j]++;
                    break;
                }

            }

        }

    }

    return;
}


NeutronSources StorkDelayedNeutron::GetDelayedNeutrons(G4double runEnd)
{
    //Initialize variables.

    G4double mass = neutron->GetPDGMass();
    G4ThreeVector theMomDir;
    G4double nMom;
    NeutronSources dNeutrons;
    StorkNeutronData theDelayed;


    for(G4int i=0; i<6; i++){

        for(G4int j = 0; j< Precursors[i] ; j++){

            //Roulette to find the time of decay.
            G4double r = G4UniformRand();
            G4double TimeOfDecay = (-log(r)/DecayConstants[0][i])*pow(10,9);

            //Check if within upcoming run if so decay a precursor and create a delayed neutron.
            if (TimeOfDecay<runDuration){

                //Get a random indice for a fission site.
                G4int R_ind = (G4int) std::floor(G4UniformRand()*fSites.size() + 0.5);
                G4ThreeVector site(fSites[R_ind].data);

                //Remove a precursor.
                Precursors[i]--;

                //Gaussian sample for a momentum.
                nMom = G4RandGauss::shoot(sqrt(2*mass*EnergyYields[0][i]),0.05);

                // Create the incident neutron
                theMomDir.setRThetaPhi(nMom, G4UniformRand()*CLHEP::pi,
                                       G4UniformRand()*2.0*CLHEP::pi);
                //Get global time of decay.
                G4double GlobalDecayTime = (runEnd-runDuration)+TimeOfDecay;

                // Set delayed neutron data.
                theDelayed = StorkNeutronData(GlobalDecayTime ,0.,site,theMomDir);

                // Add to delayed neutron list
                dNeutrons.push_back(theDelayed);

            }

        }
    }
    //Return the list of created delayed neutrons.
    return dNeutrons;
}



/*
//Finds the isotope involved in the fission process.
G4int StorkDelayedNeutron::fissionIndex(G4double fEnergy, G4ThreeVector fSite)
{
    //Get the isotope by re-simulating the interacting of the material with the neutron.
    G4String iso = theFissionProcess->GetFissionNucleus(fEnergy,fSite).GetIsotope()->GetName();
    G4int index;

    //Select the correct data (using the index).
    if(iso == "U235"){
        if(fEnergy<0.0000005) index = 0;
        else index = 3;

    }
    else if(iso == "U238"){
        index = 6;

    }
    else if(iso == "U233"){
        if(fEnergy<0.0000005) index = 2;
        else index = 5;

    }
    else if(iso == "Pu239"){
        if(fEnergy<0.0000005) index = 1;
        else index = 4;
    }
    else if(iso == "Pu240"){
        index = 7;
    }
    else if(iso == "Th232"){
        index = 8;
    }
    else {
        index = 0;
    }
    return index;
}
 */


