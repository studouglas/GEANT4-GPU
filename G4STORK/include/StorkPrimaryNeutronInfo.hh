/*
StorkPrimaryNeutronInfo.hh

Created by:		Liam Russell
Date:			22-06-2011
Modified:		11-03-2012

Definition of StorkPrimaryNeutronInfo class.

Used to store the lifetime (duration) of the neutron.  For both delayed and
prompt neutrons this will be the time since the fission that created the
neutrons (relative to the current global time).  This is saved in the
local time information of the track (using a User Tracking Action).

*/

#ifndef STORKPRIMARYNEUTRONINFO_H
#define STORKPRIMARYNEUTRONINFO_H

#include "G4VUserPrimaryParticleInformation.hh"
#include "globals.hh"


class StorkPrimaryNeutronInfo : public G4VUserPrimaryParticleInformation
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkPrimaryNeutronInfo(G4int numProcess = 4)
        {
            numHadProcess = numProcess;
            eta = new G4double[numProcess];
            lifetime = 0.0;
            weight = 1.0;
        }
        ~StorkPrimaryNeutronInfo()
        {
            delete [] eta;
        }

        void Print() const {;}

        // Set and get lifetime
        void SetLifeTime(G4double lTime) { lifetime = lTime; };
        G4double GetLifeTime() { return lifetime; };

        // Set and get eta values (number of interaction lengths left)
        void SetEta(G4double *nArray)
        {
            for(G4int i=0; i<numHadProcess; i++)
            {
                eta[i] = nArray[i];
            }
        }
        G4double GetEta(G4int ind) const { return eta[ind]; };

        // Set and get weight
        void SetWeight(G4double pWeight) { weight = pWeight; };
        G4double GetWeight() { return weight; };


    private:
        // Private member variables

        G4double lifetime;
        G4double *eta;
        G4int numHadProcess;
        G4double weight;
};

#endif // STORKPRIMARYNEUTRONINFO_H
