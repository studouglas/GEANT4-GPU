/*
BareSphereConstructor.hh

Created by:		Liam Russell
Date:			23-05-2012
Modified:       10-03-2013

Header for BareSphereConstructor class.

This class creates the simulation geometry and materials based on the input
file. The geometry is a bare sphere of various diameters and the materials are
either solid U235, a homogeneous mixture of natural uranium and heavy water, or
the Godiva sphere (critical sphere of HEU).

*/

#ifndef BARESPHERE_H
#define BARESPHERE_H

// Include header files
#include "StorkNeutronSD.hh"
#include "StorkVWorldConstructor.hh"
#include "G4Orb.hh"


class BareSphereConstructor : public StorkVWorldConstructor
{
	public:
        // Public memeber functions

		// Constructor and destructor
		BareSphereConstructor();
		virtual ~BareSphereConstructor();
		virtual G4VPhysicalVolume* ConstructNewWorld(const StorkParseInput* infile);


    protected:
        // Protected member functions

		// Material name enumerator
		enum WorldMats
		{ e_U235=0, e_Godiva, e_UHW, NUM_MATERIALS };

        // Private member functions
        virtual G4VPhysicalVolume* ConstructWorld();
        virtual void ConstructMaterials();

    protected:
        // Protected member variables

        // Logical Volumes
        G4LogicalVolume *reactorLogical;

        // Visualization attributes
        G4VisAttributes *reactorVisAtt;

        // Stored variables from infile
        G4String materialID;
        G4double matTemp;
        G4double matDensity[NUM_MATERIALS];
        G4double reactorRadius;
};

#endif // BARESPHERE_H
