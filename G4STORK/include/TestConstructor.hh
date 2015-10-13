#ifndef TestConstructor_H
#define TestConstructor_H

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

class TestConstructor: public StorkVWorldConstructor
{
    public:
        TestConstructor();
        virtual ~TestConstructor();

    protected:
        // Protected member functions

		virtual G4VPhysicalVolume* ConstructWorld();
        virtual void ConstructMaterials();

    protected:
        // Protected member variables
        solidPos sheatheTubeLatPair;
        G4VSolid* gridSlice;
        G4VSolid* test;
        G4ThreeVector originTubeSec[6];
        // Logical Volumes
        G4LogicalVolume *cellLogical;

        G4LogicalVolume *zircGridLogical;

//        // Visualization attributes
        G4VisAttributes *cellVisAtt;

        G4VisAttributes *zircGridVisAtt;


};

#endif // TestConstructor_H

