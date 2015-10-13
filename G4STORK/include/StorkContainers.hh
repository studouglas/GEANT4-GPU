/*
StorkContainers.hh

Created by:		Liam Russell
Date:			22-06-2011
Modified:		10-03-2013

Header file with defintions (typedefs) for various containers classes. All
typedef container classes used in multiple files are defined here.

*/



#ifndef CONTAINERS_HH_INCLUDED
#define CONTAINERS_HH_INCLUDED

// Include G4-STORK headers
#include "StorkTriple.hh"
#include "StorkQuad.hh"
#include "StorkNeutronData.hh"
#include "StorkTripleFloat.hh"
#include "StorkInterpVector.hh"

// Include Geant4 headers
#include "G4String.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"

// Include other headers
#include <vector>
#include <list>
#include <map>


// Containers for neutron data (survivors and delayed)
typedef std::vector<StorkNeutronData> NeutronSources;
typedef std::list<StorkNeutronData> NeutronList;


// List of site indices for the fission source distribution
// This is used in the Shannon entropy calculation
typedef StorkTriple<G4int,G4int,G4int> Index3D;
typedef std::vector<Index3D> SiteList;
typedef std::vector<G4double> DblVector;
typedef std::vector<G4ThreeVector> SiteVector;
typedef std::vector<StorkTripleFloat> MSHSiteVector;

// Three dimensional integer array
// This is used for tallying the fission sites and calculating the
// Shannon entropy
typedef std::vector<std::vector<std::vector<G4int> > > IntArray3D;


// Material-Property containers
typedef std::pair<MatEnum,PropEnum> MatPropPair;

#include "StorkMatPropChange.hh"
typedef std::vector<StorkMatPropChange> StorkMatPropChangeVector;
typedef StorkQuad<G4String,MatPropPair,
                  StorkInterpVector,G4double> MatPropInterpolator;


// Property and material maps
typedef std::map<MatPropPair,G4double> InitialPropertyMap;
typedef std::map<MatPropPair,G4double*> WorldPropertyMap;
typedef std::map<G4String,G4Material*> StorkMaterialMap;
typedef std::map<G4String,G4Material*>::iterator StorkMaterialMapItr;

#endif // CONTAINERS_HH_INCLUDED
