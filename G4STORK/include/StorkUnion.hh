#include "G4UnionSolid.hh"
#include "G4ThreeVector.hh"
#include "G4String.hh"
#include "G4VSolid.hh"
#include <sstream>
#include <vector>


typedef std::pair<G4VSolid*, G4ThreeVector> solidPos;
typedef std::vector<solidPos> solidList;

class StorkUnion
{
    public:
        StorkUnion(solidList* List);
        ~StorkUnion(void);

        solidPos GetUnionSolid(G4String name);
        solidList GetSolidList(void)
        {
            return *Unions;
        }


    private:
        solidList* Unions;
};
