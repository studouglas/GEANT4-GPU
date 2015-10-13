#ifndef DEBUGCONSTRUCTOR_H
#define DEBUGCONSTRUCTOR_H

// Include header files
#include "StorkVWorldConstructor.hh"
#include "StorkNeutronSD.hh"
#include "UnionBinaryTree.hh"
#include "G4SubtractionSolid.hh"
#include "G4IntersectionSolid.hh"
#include "G4ReflectionFactory.hh"
#include "G4ReflectedSolid.hh"
#include "G4Transform3D.hh"
#include "G4DisplacedSolid.hh"
#include "G4RotationMatrix.hh"
#include "G4AffineTransform.hh"
#include "G4PVReplica.hh"
#include "G4Para.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include <sstream>
#include "G4NistManager.hh"

class DebugConstructor: public StorkVWorldConstructor
{
    public:
        DebugConstructor();
        virtual ~DebugConstructor();

    protected:
        // Protected member functions

		virtual G4VPhysicalVolume* ConstructWorld();
        virtual void ConstructMaterials();

    protected:
        // Protected member variables
        G4ThreeVector originTubeSec[6];
        // Logical Volumes
        G4LogicalVolume *cellLogical;
        G4LogicalVolume *alumShellLogical;
        G4LogicalVolume *alumContLogical;
        G4LogicalVolume *D2OContLogical;
        G4LogicalVolume *reflectorLogical;
        G4LogicalVolume *insAlumLogical;
        G4LogicalVolume *insBeamLogical;
        G4LogicalVolume *outSmallAlumLogical;
        G4LogicalVolume *outLargeAlumLogical;
        G4LogicalVolume *cadLinLogical;
        G4LogicalVolume *outSmallBeamLogical;
        G4LogicalVolume *outLargeBeamLogical;
        G4LogicalVolume *coreWaterLogical;
        G4LogicalVolume *coreWaterSliceLogical;
        G4LogicalVolume *zircGridLogical;
        G4LogicalVolume *airGapsLatLogical;
//        G4LogicalVolume *airGapsLatHLogical;
        G4LogicalVolume *airGapsLatHRLogical;
        G4LogicalVolume *airGapsLatHR2Logical;
        G4LogicalVolume *fuelLatLogical;
//        G4LogicalVolume *fuelLatHLogical;
        G4LogicalVolume *fuelLatHRLogical;
        G4LogicalVolume *fuelLatHR2Logical;
        G4LogicalVolume *contRodZirLogical;
        G4LogicalVolume *contRodAlumLogical;
        G4LogicalVolume *contRodCadLogical;
        G4LogicalVolume *contRodCentLogical;


//        // Visualization attributes
        G4VisAttributes *cellVisAtt;
        G4VisAttributes *alumShellVisAtt;
        G4VisAttributes *alumContVisAtt;
        G4VisAttributes *D2OContVisAtt;
        G4VisAttributes *reflectorVisAtt;
        G4VisAttributes *insAlumVisAtt;
        G4VisAttributes *insBeamVisAtt;
        G4VisAttributes *outSmallAlumVisAtt;
        G4VisAttributes *outLargeAlumVisAtt;
        G4VisAttributes *cadLinTubeVisAtt;
        G4VisAttributes *outSmallBeamVisAtt;
        G4VisAttributes *outLargeBeamVisAtt;
        G4VisAttributes *coreWaterVisAtt;
        G4VisAttributes *coreWaterSliceVisAtt;
        G4VisAttributes *zircGridVisAtt;
        G4VisAttributes *airGapsLatVisAtt;
        G4VisAttributes *airGapsLatHVisAtt;
        G4VisAttributes *fuelLatVisAtt;
        G4VisAttributes *fuelLatHVisAtt;
        G4VisAttributes *contRodZirVisAtt;
        G4VisAttributes *contRodAlumVisAtt;
        G4VisAttributes *contRodCadVisAtt;
        G4VisAttributes *contRodCentVisAtt;


        // Stored variables from infile
        G4double contRodH;
};

#endif // SLOWPOKECONSTRUCTOR_H
