//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// $Id$
//
//
// class StorkUnionSolid
//
// Class description:
//
// Class for description of union of two CSG solids.

// History:
//
// 12.09.98 V.Grichine: created
//
// --------------------------------------------------------------------
#ifndef STORKUNIONSOLID_HH
#define STORKUNIONSOLID_HH

#include "G4BooleanSolid.hh"
#include "G4VSolid.hh"

#include "StorkSixVector.hh"
#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4AffineTransform.hh"
#include <CLHEP/Units/PhysicalConstants.h>

enum ShapeEnum {cylUnit, cubicUnit, sphUnit};
enum DirEnum {right, left, up, down, above, below};

class StorkUnionSolid : public G4BooleanSolid
{
  public:  // with description

    StorkUnionSolid( const G4String& pName, G4VSolid* pSolidA , G4VSolid* pSolidB, ShapeEnum shape, StorkSixVector<G4double> regDim, G4ThreeVector offset=G4ThreeVector(0.,0.,0.) ) ;

    StorkUnionSolid( const G4String& pName, G4VSolid* pSolidA, G4VSolid* pSolidB, G4RotationMatrix* rotMatrix, const G4ThreeVector& transVector,
                        ShapeEnum shape, StorkSixVector<G4double> regDim, G4ThreeVector offset=G4ThreeVector(0.,0.,0.) ) ;

    StorkUnionSolid( const G4String& pName, G4VSolid* pSolidA , G4VSolid* pSolidB , const G4Transform3D& transform,
                        ShapeEnum shape, StorkSixVector<G4double> regDim, G4ThreeVector offset=G4ThreeVector(0.,0.,0.) ) ;

//    StorkUnionSolid( StorkUnionSolid* solid, G4double addRegion[], ShapeEnum shape, DirEnum dir) ;

    virtual ~StorkUnionSolid() ;

    G4GeometryType  GetEntityType() const ;

    G4VSolid* Clone() const;

  public:  // without description

    StorkUnionSolid(__void__&);
      // Fake default constructor for usage restricted to direct object
      // persistency for clients requiring preallocation of memory for
      // persistifiable objects.

    StorkUnionSolid(const StorkUnionSolid& rhs);
    StorkUnionSolid& operator=(const StorkUnionSolid& rhs);
      // Copy constructor and assignment operator.

    G4bool CalculateExtent( const EAxis pAxis,
                            const G4VoxelLimits& pVoxelLimit,
                            const G4AffineTransform& pTransform,
                                  G4double& pMin, G4double& pMax ) const ;

    EInside Inside( const G4ThreeVector& p ) const ;

    G4ThreeVector SurfaceNormal( const G4ThreeVector& p ) const ;

    G4double DistanceToIn( const G4ThreeVector& p,
                           const G4ThreeVector& v  ) const ;

    G4double DistanceToIn( const G4ThreeVector& p ) const ;

    G4double DistanceToIn( const G4ThreeVector& p, G4double minDist) const;

    G4double DistanceToOut( const G4ThreeVector& p,
                            const G4ThreeVector& v,
                            const G4bool calcNorm=false,
                                  G4bool *validNorm=0,
                                  G4ThreeVector *n=0 ) const ;

    G4double DistanceToOut( const G4ThreeVector& p ) const ;


    G4bool InsideRegion( const G4ThreeVector& p ) const ;

    G4bool DistInRegion( const G4ThreeVector& p,
                           const G4ThreeVector& v  ) const ;

    G4double DistInRegion( const G4ThreeVector& q) const ;

    void AddRegionToMe( DirEnum dir, StorkSixVector<G4double> regionDim );

    void ComputeDimensions(       G4VPVParameterisation* p,
                            const G4int n,
                            const G4VPhysicalVolume* pRep ) ;

    void DescribeYourselfTo ( G4VGraphicsScene& scene ) const ;
    G4Polyhedron* CreatePolyhedron () const ;

    ShapeEnum GetRegionShape() const
    {
        return regShape;
    }
    void SetRegionShape(ShapeEnum shape)
    {
        regShape=shape;
    }
    StorkSixVector<G4double> GetRegionDim() const
    {
        return regDim;
    }
    void SetRegionDim(StorkSixVector<G4double> regionDim)
    {
        regDim=regionDim;
    }
    G4ThreeVector const GetRegionOffSet() const
    {
        return regOffSet;
    }
    void SetRegionOffSet(G4ThreeVector offSet)
    {
        regOffSet=offSet;
    }
    //G4NURBS*      CreateNURBS      () const ;

    private:

    ShapeEnum regShape;
    StorkSixVector<G4double> regDim;
    G4ThreeVector regOffSet;


};

#endif
