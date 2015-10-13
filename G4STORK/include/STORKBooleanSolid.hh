#ifndef STORKBOOLEANSOLID_H
#define STORKBOOLEANSOLID_H

#include "G4DisplacedSolid.hh"

#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "G4Transform3D.hh"

class HepPolyhedronProcessor;


class STORKBooleanSolid: public G4VSolid
{
    public:  // with description

    STORKBooleanSolid( const G4String& pName,
                          G4VSolid* pSolidA ,
                          G4VSolid* pSolidB   );

    STORKBooleanSolid( const G4String& pName,
                          G4VSolid* pSolidA ,
                          G4VSolid* pSolidB,
                          G4RotationMatrix* rotMatrix,
                    const G4ThreeVector& transVector    );

    STORKBooleanSolid( const G4String& pName,
                          G4VSolid* pSolidA ,
                          G4VSolid* pSolidB ,
                    const G4Transform3D& transform   );

    virtual ~STORKBooleanSolid();

    virtual const G4VSolid* GetConstituentSolid(G4int no) const;
    virtual       G4VSolid* GetConstituentSolid(G4int no);
      // If Solid is made up from a Boolean operation of two solids,
      // return the corresponding solid (for no=0 and 1).
      // If the solid is not a "Boolean", return 0.

    inline G4double GetCubicVolume();
    inline G4double GetSurfaceArea();

    virtual G4GeometryType  GetEntityType() const;
    virtual G4Polyhedron* GetPolyhedron () const;

    std::ostream& StreamInfo(std::ostream& os) const;

    inline G4int GetCubVolStatistics() const;
    inline G4double GetCubVolEpsilon() const;
    inline void SetCubVolStatistics(G4int st);
    inline void SetCubVolEpsilon(G4double ep);

    inline G4int GetAreaStatistics() const;
    inline G4double GetAreaAccuracy() const;
    inline void SetAreaStatistics(G4int st);
    inline void SetAreaAccuracy(G4double ep);

    G4ThreeVector GetPointOnSurface() const;

  public:  // without description

    STORKBooleanSolid(__void__&);
      // Fake default constructor for usage restricted to direct object
      // persistency for clients requiring preallocation of memory for
      // persistifiable objects.

    STORKBooleanSolid(const STORKBooleanSolid& rhs);
    STORKBooleanSolid& operator=(const STORKBooleanSolid& rhs);
      // Copy constructor and assignment operator.

  protected:

    G4Polyhedron* StackPolyhedron(HepPolyhedronProcessor&,
                                  const G4VSolid*) const;
      // Stack polyhedra for processing. Return top polyhedron.

    inline G4double GetAreaRatio() const;
      // Ratio of surface areas of SolidA to total A+B

  protected:

    G4VSolid* fPtrSolidA;
    G4VSolid* fPtrSolidB;

    mutable G4double fAreaRatio; // Calculation deferred to GetPointOnSurface()

  private:

    G4int    fStatistics;
    G4double fCubVolEpsilon;
    G4double fAreaAccuracy;
    G4double fCubicVolume;
    G4double fSurfaceArea;

    mutable G4Polyhedron* fpPolyhedron;

    G4bool  createdDisplacedSolid;
      // If & only if this object created it, it must delete it
};

#include "STORKBooleanSolid.icc"

#endif
