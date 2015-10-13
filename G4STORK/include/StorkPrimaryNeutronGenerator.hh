/*
StorkPrimaryNeutronGenerator.hh

Created by:		Liam Russell
Date:			06-22-2011
Modified:		11-03-2013

Header for StorkPrimaryNeutronGenerator class.

This class generators primary vertices for the neutrons based on a list of
neutron sources. The generator creates a vertex for each primary neutron, even
duplicates.

*/

#ifndef STORKNEUTRONGENERATOR_H
#define STORKNEUTRONGENERATOR_H

#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "StorkPrimaryNeutronInfo.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Event.hh"
#include "StorkContainers.hh"

class StorkPrimaryNeutronGenerator
{
    public:
        // Public member functions

        // Constructor and Destructor
        StorkPrimaryNeutronGenerator() {;}
        ~StorkPrimaryNeutronGenerator() {;}

        // Generate the primaries for the event
        void GeneratePrimaryVertices(G4Event *evt, NeutronSources *nSource);

        // Set the particle type to neutrons
        inline void SetParticleType(G4ParticleDefinition *nType)
        {
            neutron = nType;
        }


    private:
        // Private member variable

        G4ParticleDefinition *neutron;

};

#endif // STORKNEUTRONGENERATOR_H
