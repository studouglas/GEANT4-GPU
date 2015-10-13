#include "StorkBCTransform.hh"

StorkBCTransform::StorkBCTransform()
{
    //ctor
}

StorkBCTransform::~StorkBCTransform()
{
    //dtor
}

void StorkBCTransform::Transform(G4ThreeVector &pos, G4ThreeVector &mom)
{
    G4ThreeVector newPos=G4ThreeVector(0.,0.,0.), newMom=G4ThreeVector(0.,0.,0.);
    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            newPos[i]+=TransPos[i][j]*pos[j];
            newMom[i]+=TransMom[i][j]*mom[j];
        }
    }

    pos=newPos;
    mom=newMom;
}
