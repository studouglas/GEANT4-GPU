/*
SCWRJasonConstructor.hh

Created by:		Wesley Ford
Date:			14-05-2014
Modified:       NA

Header for SCWRJasonConstructor class.

This class creates the simulation geometry for a SCWR.  The
material and geometric composition of the lattice cell were taken from the
DRAGON manual[1].

Based on the "C6World" class created by Wesley Ford 10-05-2012.

[1] G. Marleau, A. Hebert, and R. Roy, "A User Guide for DRAGON 3.06".  Ecole
Polytechnique de Montreal, 2012, pp. 148-152. IGE-174 Rev. 10.

*/

#ifndef SCWRJasonConstructor_H
#define SCWRJasonConstructor_H

// Include header files
#include "StorkVWorldConstructor.hh"
#include "StorkNeutronSD.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include <sstream>


class SCWRJasonConstructor: public StorkVWorldConstructor
{
	public:
		// Public member functions

		SCWRJasonConstructor();
		virtual ~SCWRJasonConstructor();


    protected:
        // Protected member functions

		virtual G4VPhysicalVolume* ConstructWorld();
        virtual void ConstructMaterials();

    protected:
        // Protected member variables

        // Logical Volumes
        G4LogicalVolume *cellLogical;
        G4LogicalVolume *pressTubeLogical;
        G4LogicalVolume *outLinerLogical;
        G4LogicalVolume *insulatorLogical;
        G4LogicalVolume *linerLogical;
        G4LogicalVolume *coolantLogical;
        G4LogicalVolume *outSheatheLogical;
        G4LogicalVolume *outSheatheLogicalH1;
        G4LogicalVolume *outSheatheLogicalH2;
        G4LogicalVolume *inSheatheLogical;
        G4LogicalVolume *inSheatheLogicalH1;
        G4LogicalVolume *inSheatheLogicalH2;
        G4LogicalVolume *outFuelLogical;
        G4LogicalVolume *outFuelLogicalH1;
        G4LogicalVolume *outFuelLogicalH2;
        G4LogicalVolume *inFuelLogical;
        G4LogicalVolume *inFuelLogicalH1;
        G4LogicalVolume *inFuelLogicalH2;
        G4LogicalVolume *flowTubeLogical;
        G4LogicalVolume *centralCoolantLogical;


        // Visualization attributes
        G4VisAttributes *cellVisAtt;
        G4VisAttributes *pressTubeVisAtt;
        G4VisAttributes *outLinerVisAtt;
        G4VisAttributes *insulatorVisAtt;
        G4VisAttributes *linerVisAtt;
        G4VisAttributes *coolantVisAtt;
        G4VisAttributes *outSheatheVisAtt;
        G4VisAttributes *inSheatheVisAtt;
        G4VisAttributes *outFuelVisAtt;
        G4VisAttributes *inFuelVisAtt;
//        G4VisAttributes *outFlowTubeVisAtt;
        G4VisAttributes *flowTubeVisAtt;
//        G4VisAttributes *inFlowTubeVisAtt;
        G4VisAttributes *centralCoolantVisAtt;

        // Stored variables from infile
        G4double latticePitch;

        G4double moderatorTemp;
        G4double moderatorDensity;

        G4double pressTubeTemp;
        G4double pressTubeDensity;

        G4double outLinerTemp;
        G4double outLinerDensity;

        G4double insulatorTemp;
        G4double insulatorDensity;

        G4double linerTemp;
        G4double linerDensity;

        G4double coolantTemp;
        G4double coolantDensity;

        G4double inSheatheTemp;
        G4double inSheatheDensity;

        G4double outSheatheTemp;
        G4double outSheatheDensity;

        G4double innerFuelTemp;
        G4double innerFuelDensity;

        G4double outerFuelTemp;
        G4double outerFuelDensity;

        G4double flowTubeTemp;
        G4double flowTubeDensity;

        G4double centralCoolantTemp;
        G4double centralCoolantDensity;
};

#endif // SCWRJasonConstructor_H
