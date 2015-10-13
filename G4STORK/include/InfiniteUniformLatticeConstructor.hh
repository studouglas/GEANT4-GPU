/*
InfiniteUniformLatticeConstructor.hh

Created by:		Liam Russell
Date:			23-05-2012
Modified:       11-03-2013

Header for InfiniteUniformLatticeConstructor class.

Builds a world composed of an infinite lattice of homogeneous cubic cells.
The available materials are a mixture of natural uranium and heavy water or
a cell of solid uranium with different (U235) enrichment levels.

*/

#ifndef CUBECONSTRUCTOR_H
#define CUBECONSTRUCTOR_H

// Include header files
#include "StorkVWorldConstructor.hh"
#include "StorkNeutronSD.hh"
#include "G4Box.hh"


class InfiniteUniformLatticeConstructor : public StorkVWorldConstructor
{
    public:
        // Public member functions

		// Constructor and destructor
        InfiniteUniformLatticeConstructor();
        virtual ~InfiniteUniformLatticeConstructor();

        virtual G4VPhysicalVolume* ConstructNewWorld(const
                                                     StorkParseInput* infile);


    protected:
        // Protected member functions

        virtual G4VPhysicalVolume* ConstructWorld();
        virtual void ConstructMaterials();

    protected:
        // Protected member functions

        // Logical Volumes
        G4LogicalVolume *reactorLogical;

        // Visualization attributes
        G4VisAttributes *reactorVisAtt;

		// Stored variables from infile
		G4String materialID;
		G4double matTemp;
		G4double matDensity;
		G4double u235Conc;
		G4double hwConc;
		G4double latticePitch;
};

#endif // CUBECONSTRUCTOR_H
