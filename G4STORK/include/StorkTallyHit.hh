/*
StorkTallyHit.hh

Created by:		Liam Russell
Date:			18-07-2011
Modified:		11-03-2013

Header for StorkTallyHit class.

Hit used for each SD to track the number of neutrons lost (capture, loss
or fission), the number of neutrons produced in n2n, and the lifetime of the
lost neutrons.

*/

#ifndef STORKTALLYHIT_H
#define STORKTALLYHIT_H

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "StorkContainers.hh"


class StorkTallyHit : public G4VHit
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkTallyHit();
        ~StorkTallyHit() {;}

        // Print hit to screen
        void Print();

        // Get and set functions for member variables
        void SetTotalLifetime(G4double life) { totalLifetime = life; }
        G4double GetTotalLifetime() { return totalLifetime; }
        void SetNLost(G4int numLost) { nLoss = numLost; }
        G4int GetNLost() { return nLoss; }
        void SetNProd(G4int numProd) { nProd = numProd; }
        G4int GetNProd() { return nProd; }
        void SetDProd(G4int numDelayProd) { dProd = numDelayProd; }
        G4int GetDProd() { return dProd; }

        void SetFissionSites(SiteVector sites) { fSites = sites; }
        const SiteVector* GetFissionSites() { return &fSites; }
        void SetFissionEnergies(DblVector ens) { fnEnergy = ens; }
        const DblVector* GetFissionEnergies() { return &fnEnergy; }
        void SetSurvivors(NeutronSources sList) { survivors = sList; }
        const NeutronSources* GetSurvivors() { return &survivors; }
        void SetDelayed(NeutronSources dList) { delayed = dList; }
        const NeutronSources* GetDelayed() { return &delayed; }


    private:
        // Private member variables

        G4double totalLifetime;         // Total lifetime of lost neutrons
        G4int nLoss;                    // Number of neutrons lost
        G4int nProd;                    // Number of neutrons produced
        G4int dProd;                    // Number of delayed neutrons produced
        DblVector fnEnergy;             // Fission incident neutron energies
        SiteVector fSites;              // Positions of fissions
        NeutronSources survivors;       // Survivors vector
        NeutronSources delayed;         // Delayed neutrons vector
};

typedef G4THitsCollection<StorkTallyHit> TallyHC;

#endif // STORKTALLYHIT_H
