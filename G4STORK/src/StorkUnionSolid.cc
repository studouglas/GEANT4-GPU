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
// Implementation of methods for the class G4IntersectionSolid
//
// History:
//
// 12.09.98 V.Grichine: first implementation
// 28.11.98 V.Grichine: fix while loops in DistToIn/Out
// 27.07.99 V.Grichine: modifications in DistToOut(p,v,...), while -> do-while
// 16.03.01 V.Grichine: modifications in CalculateExtent()
//
// --------------------------------------------------------------------

#include <sstream>

#include "StorkUnionSolid.hh"

#include "G4SystemOfUnits.hh"
#include "G4VoxelLimits.hh"
#include "G4VPVParameterisation.hh"
#include "G4GeometryTolerance.hh"

#include "G4VGraphicsScene.hh"
#include "G4Polyhedron.hh"
#include "HepPolyhedronProcessor.h"
//#include "G4NURBS.hh"
// #include "G4NURBSbox.hh"

///////////////////////////////////////////////////////////////////
//
// Transfer all data members to G4BooleanSolid which is responsible
// for them. pName will be in turn sent to G4VSolid
//### this function has been checked ###
StorkUnionSolid:: StorkUnionSolid( const G4String& pName,
                                   G4VSolid* pSolidA ,
                                   G4VSolid* pSolidB,
                        ShapeEnum shape, StorkSixVector<G4double> regionDim, G4ThreeVector offset )
  : G4BooleanSolid(pName,pSolidA,pSolidB)
{
    //sets the shape, dimensions and position of the region that the solids will be contained in
    regShape = shape;
    regDim=regionDim;
    // the offset is the vector from the origin of pSolidA to the origin of the region
    regOffSet = offset;
}

/////////////////////////////////////////////////////////////////////
//
// Constructor
//### this function has been checked ###
StorkUnionSolid::StorkUnionSolid( const G4String& pName,
                                  G4VSolid* pSolidA ,
                                  G4VSolid* pSolidB ,
                                  G4RotationMatrix* rotMatrix,
                            const G4ThreeVector& transVector,
                            ShapeEnum shape, StorkSixVector<G4double> regionDim, G4ThreeVector offset)
  : G4BooleanSolid(pName,pSolidA,pSolidB,rotMatrix,transVector)

{
    //rotMatrix set the rotation of the referance frame of pSolidB relative to pSolidA
    //transVector sets the vector from the origin of pSolidA to the origin of pSolidB before pSolidB is rotated
    //sets the shape, dimensions and position of the region that the solids will be contained in
    regShape = shape;
    regDim=regionDim;
    // the offset is the vector from the origin of pSolidA to the origin of the region
    regOffSet= offset;
}

///////////////////////////////////////////////////////////
//
// Constructor
//### this function has been checked ###
StorkUnionSolid::StorkUnionSolid( const G4String& pName,
                                  G4VSolid* pSolidA ,
                                  G4VSolid* pSolidB ,
                            const G4Transform3D& transform,
                            ShapeEnum shape, StorkSixVector<G4double> regionDim, G4ThreeVector offset)
  : G4BooleanSolid(pName,pSolidA,pSolidB,transform)
{
    //transform sets the vector from the origin of pSolidA to the origin of pSolidB and
    //then it sets the rotation of the referance frame of pSolidB relative to pSolidA
    //sets the shape, dimensions and position of the region that the solids will be contained in
    regShape = shape;
    regDim=regionDim;
    // the offset is the vector from the origin of pSolidA to the origin of the region
    regOffSet= offset;
}

//StorkUnionSolid::StorkUnionSolid( StorkUnionSolid* solid, G4double addRegion[], ShapeEnum shape, DirEnum dir)
// : G4BooleanSolid(solid->GetEntityType(),,pSolidB,transform)
//{
//    solid->AddRegionToMe(dir, addRegion);
//    *this = *solid;
//}

//////////////////////////////////////////////////////////////////
//
// Fake default constructor - sets only member data and allocates memory
//                            for usage restricted to object persistency.
//### this function has been checked ###
StorkUnionSolid::StorkUnionSolid( __void__& a )
  : G4BooleanSolid(a)
{
}

///////////////////////////////////////////////////////////
//
// Destructor
//### this function has been checked ###
StorkUnionSolid::~StorkUnionSolid()
{
}

///////////////////////////////////////////////////////////////
//
// Copy constructor
// ### I fixed the copy constructor 2014 Aug 19
//### this function has been checked ###
StorkUnionSolid::StorkUnionSolid(const StorkUnionSolid& rhs)
  : G4BooleanSolid (rhs)
{
    regShape=rhs.GetRegionShape();
    regDim=rhs.GetRegionDim();
    regOffSet=rhs.GetRegionOffSet();
}

///////////////////////////////////////////////////////////////
//
// Assignment operator
// ### I fixed the assignment operator 2014 Aug 19
//### this function has been checked ###
StorkUnionSolid& StorkUnionSolid::operator = (const StorkUnionSolid& rhs)
{
  // Check assignment to self
  //
  if (this == &rhs)  { return *this; }

  // Copy base class data
  //
  G4BooleanSolid::operator=(rhs);

  regShape=rhs.GetRegionShape();
  regDim=rhs.GetRegionDim();
  regOffSet=rhs.GetRegionOffSet();

  return *this;
}

///////////////////////////////////////////////////////////////
//
//


// taken from G4Unionsolid, this calculates the extent of the volume based off an axis or in other words the distance the particle will travel in the solid based off its current trajectory
//### this function has been checked ###
G4bool
StorkUnionSolid::CalculateExtent( const EAxis pAxis,
                               const G4VoxelLimits& pVoxelLimit,
                               const G4AffineTransform& pTransform,
                                     G4double& pMin,
                                     G4double& pMax ) const
{
  G4bool   touchesA, touchesB, out ;
  G4double minA =  kInfinity, minB =  kInfinity,
           maxA = -kInfinity, maxB = -kInfinity;

  touchesA = fPtrSolidA->CalculateExtent( pAxis, pVoxelLimit,
                                          pTransform, minA, maxA);
  touchesB= fPtrSolidB->CalculateExtent( pAxis, pVoxelLimit,
                                         pTransform, minB, maxB);
  if( touchesA || touchesB )
  {
    pMin = std::min( minA, minB );
    pMax = std::max( maxA, maxB );
    out  = true ;
  }
  else out = false ;

  return out ;  // It exists in this slice if either one does.
}

/////////////////////////////////////////////////////
//
// Important comment: When solids A and B touch together along flat
// surface the surface points will be considered as kSurface, while points
// located around will correspond to kInside

//### this function has been checked ###
EInside StorkUnionSolid::Inside( const G4ThreeVector& p ) const
{
    EInside positionA = kOutside, positionB = kOutside;

    if(this->InsideRegion(p))
    {
        positionA = fPtrSolidA->Inside(p);
        positionB = fPtrSolidB->Inside(p);
    }

    if( positionA == kInside || positionB == kInside  ||
    ( positionA == kSurface && positionB == kSurface &&
        ( fPtrSolidA->SurfaceNormal(p) +
          fPtrSolidB->SurfaceNormal(p) ).mag2() <
          1000*G4GeometryTolerance::GetInstance()->GetRadialTolerance() ) )
    {
        return kInside;
    }
    else
    {
        if( ( positionB == kSurface ) || ( positionA == kSurface ) )
          { return kSurface; }
        else
          { return kOutside; }
    }
}

//////////////////////////////////////////////////////////////
//
//
//### fixed 2014 Aug 19
//### this function has been checked ###
G4ThreeVector
StorkUnionSolid::SurfaceNormal( const G4ThreeVector& p ) const
{
    G4ThreeVector normal=G4ThreeVector(0.,0.,0.), check = G4ThreeVector(0.,0.,0.);

#ifdef G4BOOLDEBUG
    if( Inside(p) == kOutside )
    {
      G4cout << "WARNING - Invalid call in "
             << "StorkUnionSolid::SurfaceNormal(p)" << G4endl
             << "  Point p is outside !" << G4endl;
      G4cout << "          p = " << p << G4endl;
      G4cerr << "WARNING - Invalid call in "
             << "StorkUnionSolid::SurfaceNormal(p)" << G4endl
             << "  Point p is outside !" << G4endl;
      G4cerr << "          p = " << p << G4endl;
    }
#endif

    if(this->InsideRegion(p))
    {
        if(fPtrSolidA->Inside(p) == kSurface && fPtrSolidB->Inside(p) != kInside)
        {
           normal= fPtrSolidA->SurfaceNormal(p) ;
        }
        else if(fPtrSolidB->Inside(p) == kSurface &&
                fPtrSolidA->Inside(p) != kInside)
        {
           normal= fPtrSolidB->SurfaceNormal(p) ;
        }
        else
        {
          normal= fPtrSolidA->SurfaceNormal(p) ;
        }
    }
        else
            normal= fPtrSolidA->SurfaceNormal( p );

#ifdef G4BOOLDEBUG
      if(Inside(p)==kInside)
      {
        G4cout << "WARNING - Invalid call in "
             << "StorkUnionSolid::SurfaceNormal(p)" << G4endl
             << "  Point p is inside !" << G4endl;
        G4cout << "          p = " << p << G4endl;
        G4cerr << "WARNING - Invalid call in "
             << "StorkUnionSolid::SurfaceNormal(p)" << G4endl
             << "  Point p is inside !" << G4endl;
        G4cerr << "          p = " << p << G4endl;
      }
#endif
    return normal;
}

/////////////////////////////////////////////////////////////
//
// The same algorithm as in DistanceToIn(p)

G4double
StorkUnionSolid::DistanceToIn( const G4ThreeVector& p,
                                   const G4ThreeVector& v  ) const
{
#ifdef G4BOOLDEBUG
  if( Inside(p) == kInside )
  {
    G4cout << "WARNING - Invalid call in "
           << "StorkUnionSolid::DistanceToIn(p,v)" << G4endl
           << "  Point p is inside !" << G4endl;
    G4cout << "          p = " << p << G4endl;
    G4cout << "          v = " << v << G4endl;
    G4cerr << "WARNING - Invalid call in "
           << "StorkUnionSolid::DistanceToIn(p,v)" << G4endl
           << "  Point p is inside !" << G4endl;
    G4cerr << "          p = " << p << G4endl;
    G4cerr << "          v = " << v << G4endl;
  }
#endif
    G4double distA = kInfinity, distB = kInfinity;

    if(this->DistInRegion(p, v))
    {
        distA = fPtrSolidA->DistanceToIn(p,v);
        distB = fPtrSolidB->DistanceToIn(p,v);
    }

    return std::min(distA,
                    distB) ;
}

////////////////////////////////////////////////////////
//
// Approximate nearest distance from the point p to the union of
// two solids

G4double StorkUnionSolid::DistanceToIn( const G4ThreeVector& p) const
{
#ifdef G4BOOLDEBUG
  if( Inside(p) == kInside )
  {
    G4cout << "WARNING - Invalid call in "
           << "StorkUnionSolid::DistanceToIn(p)" << G4endl
           << "  Point p is inside !" << G4endl;
    G4cout << "          p = " << p << G4endl;
    G4cerr << "WARNING - Invalid call in "
           << "StorkUnionSolid::DistanceToIn(p)" << G4endl
           << "  Point p is inside !" << G4endl;
    G4cerr << "          p = " << p << G4endl;
  }
#endif
//use functions definced in the various solid classes
    G4double minDist= kInfinity, distB=kInfinity, distA=kInfinity;
    G4bool check = false, check2 = false;
    StorkUnionSolid *tempPointer;

    if(fPtrSolidA->GetEntityType()=="StorkUnionSolid")
    {
        tempPointer = dynamic_cast<StorkUnionSolid*>(fPtrSolidA);
        check = tempPointer->InsideRegion(p);
        if(check)
        {
            distA = tempPointer->DistanceToIn(p, minDist) ;
        }
    }
    else if(fPtrSolidA->Inside(p)!=kOutside)
    {
        check=true;
        distA = 0.0;
    }
    else
    {
        distA = fPtrSolidA->DistanceToIn(p);
    }


    if(!check)
    {
        if(fPtrSolidB->GetEntityType()=="StorkUnionSolid")
        {
            tempPointer = dynamic_cast<StorkUnionSolid*>(fPtrSolidB);
            check2 = tempPointer->InsideRegion(p);
            if(check2)
            {
                distB = tempPointer->DistanceToIn(p, minDist) ;
            }
        }
        else if(fPtrSolidB->Inside(p)!=kOutside)
        {
            check=true;
            distB = 0.0;
        }
        else
        {
            distB = fPtrSolidA->DistanceToIn(p);
        }
    }

    G4double safety = std::min(distA,distB) ;
    if(safety < 0.0) safety = 0.0 ;
    return safety ;
}

G4double
StorkUnionSolid::DistanceToIn( const G4ThreeVector& p, G4double minDist) const
{
#ifdef G4BOOLDEBUG
  if( Inside(p) == kInside )
  {
    G4cout << "WARNING - Invalid call in "
           << "StorkUnionSolid::DistanceToIn(p)" << G4endl
           << "  Point p is inside !" << G4endl;
    G4cout << "          p = " << p << G4endl;
    G4cerr << "WARNING - Invalid call in "
           << "StorkUnionSolid::DistanceToIn(p)" << G4endl
           << "  Point p is inside !" << G4endl;
    G4cerr << "          p = " << p << G4endl;
  }
#endif
//use functions definced in the various solid classes
    G4double regDis = this->DistInRegion(p), distA, distB;

    if(regDis<=minDist)
    {
        minDist=regDis;
        if(fPtrSolidA->GetEntityType()=="StorkUnionSolid")
        {
            StorkUnionSolid *tempPointer = dynamic_cast<StorkUnionSolid*>(fPtrSolidA);
            distA = tempPointer->DistanceToIn(p, minDist) ;
        }
        else
            distA = fPtrSolidA->DistanceToIn(p) ;

        if(fPtrSolidB->GetEntityType()=="StorkUnionSolid")
        {
            StorkUnionSolid *tempPointer = dynamic_cast<StorkUnionSolid*>(fPtrSolidB);
            distB = tempPointer->DistanceToIn(p, minDist) ;
        }
        else
            distB = fPtrSolidB->DistanceToIn(p) ;

        G4double safety = std::min(distA,distB) ;

        if(safety < 0.0)
            safety = 0.0 ;

        return safety;
    }
    return kInfinity;
}

//////////////////////////////////////////////////////////
//
// The same algorithm as DistanceToOut(p)

G4double
StorkUnionSolid::DistanceToOut( const G4ThreeVector& p,
           const G4ThreeVector& v,
           const G4bool calcNorm,
                 G4bool *validNorm,
                 G4ThreeVector *n      ) const
{
    G4double  dist = 0.0;
    G4ThreeVector Tmp;
    G4ThreeVector* nTmp=&Tmp;

    if (this->InsideRegion(p))
    {
        dist = fPtrSolidA->DistanceToOut(p,v,calcNorm, validNorm,nTmp);

        dist += fPtrSolidB->DistanceToOut(p+dist*v,v,calcNorm, validNorm,nTmp);
    }
    if( calcNorm )
    {
        *validNorm = false ;
        *n         = *nTmp ;
    }

    return dist;
}

//////////////////////////////////////////////////////////////
//
// Inverted algorithm of DistanceToIn(p)

G4double
StorkUnionSolid::DistanceToOut( const G4ThreeVector& p ) const
{
  G4double distout = 0.0;

    if(this->InsideRegion(p))
    {
      distout= std::max(fPtrSolidA->DistanceToOut(p),
                          fPtrSolidB->DistanceToOut(p) ) ;
    }

  return distout;
}

//////////////////////////////////////////////////////////////
//
//

G4GeometryType StorkUnionSolid::GetEntityType() const
{
  return G4String("StorkUnionSolid");
}

//////////////////////////////////////////////////////////////////////////
//
// Make a clone of the object

G4VSolid* StorkUnionSolid::Clone() const
{
  return new StorkUnionSolid(*this);
}

//////////////////////////////////////////////////////////////
//
//
G4bool StorkUnionSolid::InsideRegion( const G4ThreeVector& p ) const
{
    G4ThreeVector q=p-regOffSet;

    static const G4double delta=0.5*kCarTolerance;
    static const G4double delta2=0.5*(G4GeometryTolerance::GetInstance()->GetRadialTolerance());
    static const G4double delta3=0.5*(G4GeometryTolerance::GetInstance()->GetAngularTolerance());


    if(regShape==0)
    {
        if((q.rho()>=regDim[0]-delta2)&&(q.rho()<=regDim[1]+delta2)&&(q.phi()>=regDim[2]-delta3)&&(q.phi()<=regDim[3]+delta3)&&(q.z()>=regDim[4]-delta)&&(q.z()<=regDim[5]+delta))
        {
            return true;
        }
    }
    else if(regShape==1)
    {
        if((q.x()>=regDim[0]-delta)&&(q.x()<=regDim[1]+delta)&&(q.y()>=regDim[2]-delta)&&(q.y()<=regDim[3]+delta)&&(q.z()>=regDim[4]-delta)&&(q.z()<=regDim[5]+delta))
        {
            return true;
        }
    }

    else
    {
        if((q.r()>=regDim[0]-delta2)&&(q.r()<=regDim[1]+delta2)&&(q.phi()>=regDim[2]-delta3)&&(q.phi()<=regDim[3]+delta3)&&(q.theta()>=regDim[4]-delta3)&&(q.theta()<=regDim[5]+delta3))
        {
            return true;
        }
    }

    return false;

}

G4bool StorkUnionSolid::DistInRegion( const G4ThreeVector& q, const G4ThreeVector& v ) const
{
    G4double smax, smin, swap3;
    G4double swap[6];

    if (this->InsideRegion(q))
        return true;

    G4ThreeVector p=q-regOffSet;

    //This function finds the intervals over which the given trajectory is within the boundaries of each dimension
    //and then it checks to see if the intervals overlap, if they do then the trajectory will pass through the region
    //if no then the trajectory will not pass through the region
    if(regShape==0)
    {
        swap[0] = (regDim[4]-p[2])/(v[2]);
        swap[1] = (regDim[5]-p[2])/(v[2]);
        if(swap[0]>swap[1])
        {
            smax=swap[0];
            smin=swap[1];
        }
        else
        {
            smax=swap[1];
            smin=swap[0];
        }

            swap[0] = (-(p[1]*v[1]+p[0]*v[0])+pow(2*p[1]*v[1]*p[0]*v[0]+(pow(regDim[1],2))*(v.perp2()),0.5))/(v.perp2());
            swap[1] = (-(p[1]*v[1]+p[0]*v[0])-pow(2*p[1]*v[1]*p[0]*v[0]+(pow(regDim[1],2))*(v.perp2()),0.5))/(v.perp2());
            swap[2] = (-(p[1]*v[1]+p[0]*v[0])+pow(2*p[1]*v[1]*p[0]*v[0]+(pow(regDim[0],2))*(v.perp2()),0.5))/(v.perp2());
            swap[3] = (-(p[1]*v[1]+p[0]*v[0])-pow(2*p[1]*v[1]*p[0]*v[0]+(pow(regDim[0],2))*(v.perp2()),0.5))/(v.perp2());
            swap[4] = -(p[0]*tan(regDim[2])-p[1])/(v[0]*tan(regDim[2])-v[1]);
            swap[5] = -(p[0]*tan(regDim[3])-p[1])/(v[0]*tan(regDim[3])-v[1]);

            if(swap[4]<swap[5])
            {
                swap3=swap[4];
                swap[4]=swap[5];
                swap[5]=swap3;
            }

            if(swap[4]<smax)
                    smax=swap[4];

            if(swap[5]>smin)
                    smin=swap[5];

            if(!((swap[1]>=swap[4]&&swap[3]>=swap[4])||(swap[1]<=swap[5]&&swap[3]<=swap[5])))
            {
                if(swap[1]<swap[3])
                {
                    swap3=swap[1];
                    swap[1]=swap[3];
                    swap[3]=swap3;
                }
                if(swap[1]<smax)
                    smax=swap[1];

                if(swap[3]>smin)
                    smin=swap[3];
            }
            else if(!((swap[0]>=swap[4]&&swap[2]>=swap[4])||(swap[0]<=swap[5]&&swap[2]<=swap[5])))
            {
                if(swap[0]<swap[2])
                {
                    swap3=swap[0];
                    swap[0]=swap[2];
                    swap[2]=swap3;
                }
                if(swap[0]<smax)
                    smax=swap[0];

                if(swap[2]>smin)
                    smin=swap[2];
            }
            else
                return false;

        if(smax<=0.)
            return false;

        if(smin<smax)
            return true;
    }

    else if(regShape==1)
    {
        swap[0] = (regDim[0]-p[0])/(v[0]);
        swap[1] = (regDim[1]-p[0])/(v[0]);
        if(swap[0]>swap[1])
        {
            smax=swap[0];
            smin=swap[1];
        }
        else
        {
            smax=swap[1];
            smin=swap[0];
        }

        for (G4int i=2; i<5; i=i+2)
        {
            swap[0] = (regDim[i]-p[G4int(i/2)])/(v[G4int(i/2)]);
            swap[1] = (regDim[i+1]-p[G4int(i/2)])/(v[G4int(i/2)]);
            if(swap[0]<swap[1])
            {
                swap3=swap[0];
                swap[0]=swap[1];
                swap[1]=swap3;
            }

            if(swap[0]<smax)
                smax=swap[0];

            if(swap[1]>smin)
                smin=swap[1];
        }

        if(smax<=0.)
            return false;

        if(smin<smax)
            return true;
    }

    else
    {
        return true;
    }

    return false;
}

G4double StorkUnionSolid::DistInRegion( const G4ThreeVector& q ) const
{
    G4ThreeVector p=q-regOffSet;
    if(regShape==0)
    {
        G4double safe=0.0, rho, safe1, safe2, safe3 ;
        G4double safePhi, cosPsi, cosSPhi=cos(regDim[2]), cosCPhi = cos((regDim[2]+regDim[3])/2), cosEPhi=cos(regDim[3]) ;
        G4double sinSPhi=sin(regDim[2]), sinCPhi=sin((regDim[2]+regDim[3])/2), sinEPhi=sin(regDim[3]);
        rho   = std::sqrt(p.x()*p.x() + p.y()*p.y()) ;
        safe1 = regDim[0] - rho ;
        safe2 = rho - regDim[1] ;
        safe3 = (p.z()>0.) ? ((p.z())-regDim[1]) : (-(p.z())+regDim[0]) ;

        if ( safe1 > safe2 ) { safe = safe1; }
        else                 { safe = safe2; }
        if ( safe3 > safe )  { safe = safe3; }

        if ( ((regDim[3]-regDim[2])==2*CLHEP::pi) && (rho) )
        {
        // Psi=angle from central phi to point
        //
        cosPsi = (p.x()*cosCPhi + p.y()*sinCPhi)/rho ;

        if ( cosPsi < std::cos((regDim[3]-regDim[2])*0.5) )
        {
          // Point lies outside phi range

          if ( (p.y()*cosCPhi - p.x()*sinCPhi) <= 0 )
          {
            safePhi = std::fabs(p.x()*sinSPhi - p.y()*cosSPhi) ;
          }
          else
          {
            safePhi = std::fabs(p.x()*sinEPhi - p.y()*cosEPhi) ;
          }
          if ( safePhi > safe )  { safe = safePhi; }
        }
        }
        if ( safe < 0. )  { safe = 0.; }
        return safe ;
    }

    else if(regShape==1)
    {
        G4double safex, safey, safez, safe = 0.0 ;

        safex = (p.x()>0) ? ((p.x())-regDim[1]) : (-(p.x())+regDim[0]) ;
        safey = (p.y()>0) ? ((p.y())-regDim[3]) : (-(p.y())+regDim[2]) ;
        safez = (p.z()>0) ? ((p.z())-regDim[5]) : (-(p.z())+regDim[4]) ;

        if (safex > safe) { safe = safex ; }
        if (safey > safe) { safe = safey ; }
        if (safez > safe) { safe = safez ; }

        return safe ;
    }

    else
    {
        G4cerr << "This position in a sphere region has not yet been added to the StorkUnionSolid";
        return 0.;
    }
}

void StorkUnionSolid::AddRegionToMe( DirEnum dir, StorkSixVector<G4double> regionDim )
{
    if(dir==right)
    {
        if(regShape==cylUnit)
        {
            (regDim[2])=(regionDim[2]);
        }
        else if(regShape==cubicUnit)
        {
            regDim[1]+=regionDim[1]-regionDim[0];
        }
        else
        {
            regDim[2]=regionDim[2];
        }
    }
    else if(dir==left)
    {
        if(regShape==cylUnit)
        {
            regDim[3]=regionDim[3];
        }
        else if(regShape==cubicUnit)
        {
            regDim[0]+=regionDim[0]-regionDim[1];
        }
        else
        {
            regDim[3]=regionDim[3];
        }
    }
    else if(dir==up)
    {
        if(regShape==cylUnit)
        {
            regDim[1]=regionDim[1];
        }
        else if(regShape==cubicUnit)
        {
            regDim[3]+=regionDim[3]-regionDim[2];
        }
        else
        {
            regDim[1]=regionDim[1];
        }
    }
    else if(dir==down)
    {
        if(regShape==cylUnit)
        {
            regDim[0]=regionDim[0];
        }
        else if(regShape==cubicUnit)
        {
            regDim[2]+=regionDim[2]-regionDim[3];
        }
        else
        {
            regDim[0]=regionDim[0];
        }
    }
    else if(dir==above)
    {
        if(regShape==cylUnit)
        {
            regDim[5]+=regionDim[5]-regionDim[4];
        }
        else if(regShape==cubicUnit)
        {
            regDim[5]+=regionDim[5]-regionDim[4];
        }
        else
        {
            regDim[4]=regionDim[4];
        }
    }
    else if(dir==below)
    {
        if(regShape==cylUnit)
        {
            regDim[4]+=regionDim[4]-regionDim[5];
        }
        else if(regShape==cubicUnit)
        {
            regDim[4]+=regionDim[2];
        }
        else
        {
            regDim[5]=regionDim[5];
        }
    }
}


void
StorkUnionSolid::ComputeDimensions(       G4VPVParameterisation*,
                                 const G4int,
                                 const G4VPhysicalVolume* )
{
}

/////////////////////////////////////////////////
//
//

void
StorkUnionSolid::DescribeYourselfTo ( G4VGraphicsScene& scene ) const
{
  scene.AddSolid (*this);
}

////////////////////////////////////////////////////
//
//

G4Polyhedron*
StorkUnionSolid::CreatePolyhedron () const
{
  HepPolyhedronProcessor processor;
  // Stack components and components of components recursively
  // See G4BooleanSolid::StackPolyhedron
  G4Polyhedron* top = StackPolyhedron(processor, this);
  G4Polyhedron* result = new G4Polyhedron(*top);
  if (processor.execute(*result)) { return result; }
  else { return 0; }
}

/////////////////////////////////////////////////////////
//
//
/*
G4NURBS*
StorkUnionSolid::CreateNURBS      () const
{
  // Take into account boolean operation - see CreatePolyhedron.
  // return new G4NURBSbox (1.0, 1.0, 1.0);
  return 0;
}
*/
