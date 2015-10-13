
#ifndef SLOWPOKECONSTRUCTOR_H
#define SLOWPOKECONSTRUCTOR_H


// Include header files
#include "StorkVWorldConstructor.hh"
#include "StorkNeutronSD.hh"
#include "G4VSolid.hh"
#include "G4UnionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4IntersectionSolid.hh"

#include "G4Para.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Orb.hh"
#include "StorkMaterialHT.hh"
#include <sstream>
#include <cmath>
#include "StorkUnion.hh"

class SLOWPOKEConstructor: public StorkVWorldConstructor
{
    public:
        SLOWPOKEConstructor();
        virtual ~SLOWPOKEConstructor();

    protected:
        // Protected member functions

		virtual G4VPhysicalVolume* ConstructWorld();

        virtual void ConstructMaterials();

    protected:
    // Protected member variables

    // Logical Volumes
    G4LogicalVolume *ZirconiumLogical1;
    G4LogicalVolume *ZirconiumLogical2;
    G4LogicalVolume *ZirconiumLogical3;
    G4LogicalVolume *AirGapLogical;
    G4LogicalVolume *AirGapLogical2;
    G4LogicalVolume *FuelRodLogical;
    G4LogicalVolume *FuelRodLogical2;
    G4LogicalVolume *ReflectorLogical;
    G4LogicalVolume *D2OContainerLogical;
    G4LogicalVolume *D2OLogical;
    G4LogicalVolume *contRodZirLogical;
    G4LogicalVolume *contRodAlumLogical;
    G4LogicalVolume *contRodCadLogical;
    G4LogicalVolume *contRodCentLogical;
    G4LogicalVolume *insAlumLogical;
    G4LogicalVolume *insBeamLogical;
    G4LogicalVolume *outSmallAlumLogical;
    G4LogicalVolume *outLargeAlumLogical;
    G4LogicalVolume *cadLinLogical;
    G4LogicalVolume *outSmallBeamLogical;
    G4LogicalVolume *outLargeBeamLogical;
    G4LogicalVolume *alumShellLogical;
    G4LogicalVolume *cellLogical;

    // Visualization attributes
    G4VisAttributes *ZirconiumAtt1;
    G4VisAttributes *ZirconiumAtt2;
    G4VisAttributes *ZirconiumAtt3;
    G4VisAttributes *AirGapAtt;
    G4VisAttributes *FuelRodAtt;
    G4VisAttributes *ReflectorAtt;
    G4VisAttributes *D2OContainerAtt;
    G4VisAttributes *D2OAtt;
    G4VisAttributes *contRodZirVisAtt;
    G4VisAttributes *contRodAlumVisAtt;
    G4VisAttributes *contRodCadVisAtt;
    G4VisAttributes *contRodCentVisAtt;
    G4VisAttributes *insAlumVisAtt;
    G4VisAttributes *insBeamVisAtt;
    G4VisAttributes *outSmallAlumVisAtt;
    G4VisAttributes *outLargeAlumVisAtt;
    G4VisAttributes *cadLinTubeVisAtt;
    G4VisAttributes *outSmallBeamVisAtt;
    G4VisAttributes *outLargeBeamVisAtt;
    G4VisAttributes *alumShellVisAtt;
    G4VisAttributes *cellVisAtt;

    // Stored variables from infile
    G4double ControlRodPosition;
    G4double FuelTemp;
    G4double moderatorTemp;
    G4double FuelDensity;
    G4double FuelRadius;


};

#endif // SLOWPOKECONSTRUCTOR_H
