#ifndef STORKDELAYEDNEUTRON_H
#define STORKDELAYEDNEUTRON_H

#include "StorkContainers.hh"
#include "StorkDelayedNeutronData.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

class StorkDelayedNeutron
{
    //Public methods
    public:
    //Constructor
    StorkDelayedNeutron(G4String dnFilename,G4double runDuration,G4int numPrimaries);
    
    //Destructor
    virtual ~StorkDelayedNeutron();
    
    //Method to add precursors between runs.
    void AddPrecursors();
    
    //Get and set functions
    void SetPrecursors(std::vector<G4int> p) {Precursors = p;}
    std::vector<G4int> GetPrecursors() {return Precursors;}
    
    //Generate delayed neutrons through roulette of precursors.
    NeutronSources GetDelayedNeutrons(G4double runEnd);
    
    //Set the fission sites and energies.
    void SetFissionSource(MSHSiteVector fissionSites, DblVector fissionEnergies);

    //Private methods
    private:
    
    // Produce initial precursor groups
    G4bool GetInitialPrecursors(G4int numPrimaries);
    
    
    
    
    //Samples an isotope using incident fission energy and site, returns an index.
   // G4int fissionIndex(G4double fEnergy, G4ThreeVector fSite);
    
    
    //Private variables
    private:
    //Initial fission file name.
    G4String delayedSourceFile;
    
    //Fission source data.
    MSHSiteVector fSites;
    DblVector fEnergy;
    std::vector<G4int> Precursors;
    
    G4ParticleDefinition *neutron;
    
    //Run duration
    G4double runDuration;
    
    
    
};

#endif