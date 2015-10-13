/*
C6latticeConstructor.hh

Created by:		Liam Russell
Date:			23-05-2012
Modified:       11-03-2013

Header for C6LatticeConstructor class.

This class creates the simulation geometry for a CANDU 6 lattice cell.  The
material and geometric composition of the lattice cell were taken from the
DRAGON manual[1].

Based on the "C6World" class created by Wesley Ford 10-05-2012.

[1] G. Marleau, A. Hebert, and R. Roy, "A User Guide for DRAGON 3.06".  Ecole
Polytechnique de Montreal, 2012, pp. 148-152. IGE-174 Rev. 10.

*/

#ifndef C6LATTICECONSTRUCTOR_H
#define C6LATTICECONSTRUCTOR_H

// Include header files
#include "StorkVWorldConstructor.hh"
#include "StorkNeutronSD.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include <sstream>


class C6LatticeConstructor: public StorkVWorldConstructor
{
	public:
		// Public member functions

		C6LatticeConstructor();
		virtual ~C6LatticeConstructor();


    protected:
        // Protected member functions

		virtual G4VPhysicalVolume* ConstructWorld();
        virtual void ConstructMaterials();

    protected:
        // Protected member variables

        // Logical Volumes
        G4LogicalVolume *cellLogical;
        G4LogicalVolume *cTubeLogical;
        G4LogicalVolume *gasAnnLogical;
        G4LogicalVolume *pressTubeLogical;
        G4LogicalVolume *coolantLogical;
        G4LogicalVolume *sheatheLogical;
        G4LogicalVolume *fuelLogical;

        // Visualization attributes
        G4VisAttributes *modVisAtt;
        G4VisAttributes *cTubeVisAtt;
        G4VisAttributes *gasAnnVisAtt;
        G4VisAttributes *pressTubeVisAtt;
        G4VisAttributes *coolantVisAtt;
        G4VisAttributes *sheatheVisAtt;
        G4VisAttributes *fuelVisAtt;

        // Stored variables from infile
        G4double latticePitch;
        G4double fuelTemp;
        G4double fuelDensity;
        G4double coolantTemp;
        G4double coolantDensity;
        G4double moderatorTemp;
        G4double moderatorDensity;
};

#endif // C6LATTICECONSTRUCTOR_H
