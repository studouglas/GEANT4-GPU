#ifndef GUILLAUMECONSTRUCTOR_H
#define GUILLAUMECONSTRUCTOR_H


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
#include <sstream>
#include <cmath>

class GuillaumeConstructor: public StorkVWorldConstructor
{
    public:
        GuillaumeConstructor();
        virtual ~GuillaumeConstructor();

    protected:
        // Protected member functions

		virtual G4VPhysicalVolume* ConstructWorld();
        virtual void ConstructMaterials();

    protected:
        // Protected member variables

        // Logical Volumes
        G4LogicalVolume *ZirconiumLogical;
        G4LogicalVolume *WaterHolesLowerLogical;
        G4LogicalVolume *WaterHolesUpperLogical;
        G4LogicalVolume *AirGapLogical;
        G4LogicalVolume *FuelRodLogical;
        G4LogicalVolume *LowerPinLogical;
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
        G4VisAttributes *ZirconiumAtt;
        G4VisAttributes *WaterHolesLowerAtt;
        G4VisAttributes *WaterHolesUpperAtt;
        G4VisAttributes *AirGapAtt;
        G4VisAttributes *FuelRodAtt;
        G4VisAttributes *LowerPinAtt;
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

        // Union Solid
        G4UnionSolid *Zirconium;

        // Stored variables from infile
        G4double latticePitch;
        G4double fuelTemp;
        G4double fuelDensity;
        G4double coolantTemp;
        G4double coolantDensity;
        G4double moderatorTemp;
        G4double moderatorDensity;
};

#endif // GUILLAUMECONSTRUCTOR_H
