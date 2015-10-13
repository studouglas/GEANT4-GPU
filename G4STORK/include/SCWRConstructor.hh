/*
SCWRConstructor.hh

Created by:		Wesley Ford
Date:			14-05-2014
Modified:       NA

Header for SCWRConstructor class.

This class creates the simulation geometry for a SCWR.

Based on the "C6World" class created by Wesley Ford 10-05-2012.

*/

#ifndef SCWRCONSTRUCTOR_H
#define SCWRCONSTRUCTOR_H

// Include header files
#include "StorkVWorldConstructor.hh"
#include "StorkNeutronSD.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include <sstream>


class SCWRConstructor: public StorkVWorldConstructor
{
	public:
		// Public member functions

		SCWRConstructor();
		virtual ~SCWRConstructor();


    protected:
        // Protected member functions

		virtual G4VPhysicalVolume* ConstructWorld();
        virtual void ConstructMaterials();

    protected:
        // Protected member variables

        // Logical Volumes
        G4LogicalVolume *cellLogical;
        G4LogicalVolume *pressTubeLogical1;
        G4LogicalVolume *pressTubeLogical2;
        G4LogicalVolume *pressTubeLogical3;
        G4LogicalVolume *pressTubeLogical4;
        G4LogicalVolume *outLinerLogical;
        G4LogicalVolume *insulatorLogical1;
        G4LogicalVolume *insulatorLogical2;
        G4LogicalVolume *insulatorLogical3;
        G4LogicalVolume *insulatorLogical4;
        G4LogicalVolume *linerLogical;
        G4LogicalVolume *coolantLogical;
        G4LogicalVolume *outSheatheLogical;
        G4LogicalVolume *inSheatheLogical;
        G4LogicalVolume *outFuelLogical1;
        G4LogicalVolume *outFuelLogical2;
        G4LogicalVolume *outFuelLogical3;
        G4LogicalVolume *outFuelLogical4;
        G4LogicalVolume *inFuelLogical1;
        G4LogicalVolume *inFuelLogical2;
        G4LogicalVolume *inFuelLogical3;
        G4LogicalVolume *inFuelLogical4;
//        G4LogicalVolume *outFlowTubeLogical;
//        G4LogicalVolume *flowTubeLogical1;
//        G4LogicalVolume *flowTubeLogical2;
//        G4LogicalVolume *inFlowTubeLogical;
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

        G4double pressTubeTemp[4];
        G4double pressTubeDensity;

        G4double outLinerTemp;
        G4double outLinerDensity;

        G4double insulatorTemp[4];
        G4double insulatorDensity;

        G4double linerTemp;
        G4double linerDensity;

        G4double coolantTemp;
        G4double coolantDensity;

        G4double inSheatheTemp;
        G4double inSheatheDensity;

        G4double outSheatheTemp;
        G4double outSheatheDensity;

        G4double innerFuelTemp[4];
        G4double innerFuelDensity;

        G4double outerFuelTemp[4];
        G4double outerFuelDensity;

        G4double flowTubeTemp;
        G4double flowTubeDensity;

        G4double centralCoolantTemp;
        G4double centralCoolantDensity;
};

#endif // SCWRCONSTRUCTOR_H
