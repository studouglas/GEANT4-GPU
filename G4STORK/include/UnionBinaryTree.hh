#include "G4UnionSolid.hh"
#include "StorkUnionSolid.hh"
#include "G4ThreeVector.hh"
#include "G4String.hh"
#include "G4VSolid.hh"
#include "G4Tubs.hh"
#include "G4Box.hh"
#include "G4Sphere.hh"
#include "G4Orb.hh"
#include <sstream>
#include <vector>
#include <map>
#include "G4GeometryTolerance.hh"


typedef std::pair<G4VSolid*, G4ThreeVector> solidPos;
typedef std::pair<StorkSixVector<G4double>, G4double> regionInfo;
typedef std::vector<solidPos> solidList;
typedef std::vector<G4int> intVec;
typedef std::vector<regionInfo> regList;
typedef std::vector<solidPos>::iterator solidListItr;
typedef std::vector<regionInfo>::iterator regListItr;

enum coorEnum {radSph=0, radCyl, phi, xDir, yDir, ZDir};

class UnionBinaryTree
{
    public:
        UnionBinaryTree(solidList* List);
        ~UnionBinaryTree(void);

        solidPos GetUnionSolid(G4String name, G4int equalP, ShapeEnum regShape, G4double unitRegionDim[], G4double regionDim[], G4double offset=0.0, coorEnum axis=radCyl, G4double order=1.0, G4double inPriority[]=NULL, G4bool delRepeatedEntries=false);
        StorkSixVector<G4double> AddRegion( StorkSixVector<G4double> regionDim, StorkSixVector<G4double> regionDim2, ShapeEnum shape, DirEnum dir  ) ;
        solidList GetSolidList(void)
        {
            return *Unions;
        }


    private:
        solidList* Unions;
        G4double* priority;
        G4int numSolids;
        regList regionList;

        void SortSolids(solidList* Temp1, regList* rList, solidList* ToBeAdded );
        void SortEqualSolids(solidList* Temp1, solidList* ToBeAdded);
        void PriorityByVol(G4double offset);
        void PriorityByPos(coorEnum axis, G4double dir, G4double offset);
        void RemoveDuplicates();
        intVec createRegions(solidList* Temp1, ShapeEnum RegShape, G4double unitRegionDim[], G4double regionDim[]);
        G4bool InsideRegion( const G4ThreeVector& p, ShapeEnum regShape, StorkSixVector<G4double> regDim);
};
