/*
StorkTallyHit.cc

Created by:		Liam Russell
Date:			18-07-2011
Modified:		11-03-2013

Source code for StorkTallyHit class.

*/


// Include header file

#include "StorkTallyHit.hh"


// Constructor
StorkTallyHit::StorkTallyHit()
{
    totalLifetime =  0.0;
    nLoss = nProd = dProd = 0;
}


// Print()
// Outputs all of the tally information to the G4cout.
void StorkTallyHit::Print()
{
    G4cout << "Following quantities are tallied: " << G4endl
           << "Neutrons Lost = " << nLoss << G4endl
           << "Neutrons Produced = " << nProd << G4endl
           << "Total lifetime of lost neutrons = " << totalLifetime << G4endl
           << "Number of survivors = " << survivors.size() << G4endl
           << "Number of delayed = " << delayed.size() << G4endl
           << "Number of fission sites = " << fSites.size() << G4endl;
}
