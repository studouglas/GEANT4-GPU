/*
StorkNeutronPhysicsList.hh

Created by:		Liam Russell
Date:			23-02-2012
Modified:       11-03-2013

Header file for StorkNeutronPhysicsList class.

This class creates the physics processes for the simulation through process
builder classes.  It also sets the energy range for each builder and the
available particles.

*/

#ifndef STORKHPNEUTRONPHYSICSLIST_H
#define STORKHPNEUTRONPHYSICSLIST_H

#include "G4VUserPhysicsList.hh"
#include "StorkHPNeutronBuilder.hh"
#include "StorkNeutronProcessBuilder.hh"
#include "G4NeutronBuilder.hh"
#include "G4BertiniNeutronBuilder.hh"
//#include "G4LEPNeutronBuilder.hh"
#include "StorkParseInput.hh"

#ifdef G4USE_TOPC
    #include "topc.h"
    #include "G4HadronicProcessStore.hh"
#endif


class StorkNeutronPhysicsList : public G4VUserPhysicsList
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkNeutronPhysicsList(const StorkParseInput* infile);
        ~StorkNeutronPhysicsList();


    protected:
        //Protected member functions

        void ConstructParticle();       // Set available particles
        void ConstructProcess();        // Set available processes
        void SetCuts();


    private:
        // Private member variables

        // User inputs
        G4String csDirName;
        G4String fsDirName;
        G4int kCalcType;
        std::vector<G4int>* periodicBCVec;
        std::vector<G4int>* reflectBCVec;

        // Neutron physics builders
        StorkNeutronProcessBuilder *theNeutrons;
        StorkHPNeutronBuilder *theHPNeutron;
        G4BertiniNeutronBuilder *theBertiniNeutron;
        //G4LEPNeutronBuilder *theLEPNeutron;
};

#endif // STORKHPNEUTRONPHYSICSLIST_H
