#include "StorkPeriodicBCTransform.hh"

StorkPeriodicBCTransform::StorkPeriodicBCTransform()
{
    //ctor
}

StorkPeriodicBCTransform::~StorkPeriodicBCTransform()
{
    //dtor
}

StorkPeriodicBCTransform::StorkPeriodicBCTransform(G4ThreeVector n1, G4ThreeVector n2)
{
    InitializeTransform(n1, n2);
}

void StorkPeriodicBCTransform::InitializeTransform(G4ThreeVector n1, G4ThreeVector n2)
{

    if(n1==n2)
    {
        G4cout << "\n Error: Periodic Boundary was linked to itself" << G4endl;
        return;
    }

    // make n2 point inside the world volume;
    n2=-1*n2;

    G4int index=-1;
    for(G4int i=0; i<3; i++)
    {
        if(n2[i]!=0)
        {
            index=i;
        }
    }
    if(index==-1)
    {
        G4cout << "\n Error: Periodic Boundary was given a zero vector for n2" << G4endl;
        return;
    }

    for(G4int i=0; i<3; i++)
    {
        for(G4int j=0; j<3; j++)
        {
            if(i==j)
            {
                TransMom[i][j]=1;
            }
            else
            {
                TransMom[i][j]=0;
            }
        }
    }
    for(G4int i=0; i<3; i++)
    {
        for(G4int j=0; j<3; j++)
        {
            if(n1[j]!=0)
            {
                TransMom[i][j]=n2[i]/n1[j];
                if(i==j)
                {
                    TransMom[index][index]=TransMom[i][j];
                }
                else
                {
                    TransMom[j][i]=-n2[i]/n1[j];
                }
            }
        }
    }

    for(G4int i=0; i<3; i++)
    {
        for(G4int j=0; j<3; j++)
        {
            if((n2[i]!=0)&&(n1[j]!=0))
            {
                TransPos[i][j]=-1*TransMom[i][j];
            }
            else
            {
                TransPos[i][j]=TransMom[i][j];
            }
        }
    }

}

