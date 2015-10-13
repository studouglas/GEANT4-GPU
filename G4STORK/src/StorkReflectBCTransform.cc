#include "StorkReflectBCTransform.hh"

StorkReflectBCTransform::StorkReflectBCTransform()
{
    //ctor
}

StorkReflectBCTransform::StorkReflectBCTransform(G4ThreeVector n1)
{
    InitializeTransform(n1);
}

StorkReflectBCTransform::~StorkReflectBCTransform()
{
    //dtor
}

void StorkReflectBCTransform::InitializeTransform(G4ThreeVector n1)
{

    for(G4int i=0; i<3; i++)
    {
        for(G4int j=0; j<3; j++)
        {
            if(i==j)
            {
                TransPos[i][j]=1;
            }
            else
            {
                TransPos[i][j]=0;
            }
        }
    }

    for(G4int i=0; i<3; i++)
    {
        for(G4int j=0; j<3; j++)
        {
            if(n1[j]!=0)
            {
                TransMom[i][j]=-1*TransPos[i][j];
            }
            else
            {
                TransMom[i][j]=TransPos[i][j];
            }
        }
    }

}
