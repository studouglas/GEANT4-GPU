/*
StorkPrimaryNeutronGenerator.cc

Created by:		Liam Russell
Date:			22-06-2011
Modified:		11-03-2013

Source code for StorkPrimaryNeutronGenerator class.

*/


// Include header file
#include "StorkPrimaryNeutronGenerator.hh"


// GeneratePrimaryVertices()
// Create a primary neutron in the event for each of the NeutronSources entries.
void StorkPrimaryNeutronGenerator::GeneratePrimaryVertices(G4Event *evt,
                                               NeutronSources *nSource)
{
    G4PrimaryVertex *vertex;
    G4PrimaryParticle *particle;
    StorkNeutronData *source;
    StorkPrimaryNeutronInfo *pnInfo;

    G4double mass = neutron->GetPDGMass();
    G4ThreeVector pol = G4ThreeVector(0.,0.,0.);
    G4int numPrimaries = G4int(nSource->size());

    // Create a primary neutron for each entry in nSource
    for(G4int i=0; i<numPrimaries; i++)
    {
        source = &(*nSource)[i];

        // Set the position and global time of the vertex
        vertex = new G4PrimaryVertex(source->third, source->first);

        particle = new G4PrimaryParticle(neutron, source->fourth[0],
                                         source->fourth[1], source->fourth[2]);
        particle->SetMass(mass);
        particle->SetCharge(0.0);
        particle->SetPolarization(pol[0],pol[1],pol[2]);

        // Create the primary particle user information
        pnInfo = new StorkPrimaryNeutronInfo();
        // Set the lifetime
        pnInfo->SetLifeTime(source->second);
        // Set the eta values
        G4double eta[4] = {source->fifth,source->sixth,source->seventh,
                            source->eigth};
        pnInfo->SetEta(eta);
        // Set the weight and primary neutron info
        pnInfo->SetWeight(source->ninth);
        particle->SetUserInformation(pnInfo);

        vertex->SetPrimary(particle);

        // Add the primary to the event
        evt->AddPrimaryVertex(vertex);
    }
}
