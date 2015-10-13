#include "StorkUnion.hh"

StorkUnion::StorkUnion(solidList* List)
{
    Unions = List;
}

StorkUnion::~StorkUnion(void)
{
    delete Unions;
}

solidPos StorkUnion::GetUnionSolid(G4String name)
{
    std::stringstream unionName;
    solidList Temp1 = *Unions, ToBeAdded = solidList();
    G4int size = Temp1.size();
    G4bool Add = false;
    G4int n = 0;

    while(size > 2)
    {
        solidList Temp2 = solidList();

        if(size%2 != 0 && ToBeAdded.size() == 1)
        {
            size = size - 1;
            Add = true;
        }
        else if(size%2 != 0)
        {
            size = size - 1;
            ToBeAdded.push_back(Temp1[size]);
        }

        for(G4int i = 0; i < int(size/2); i++)
        {
            unionName.str(name);
            unionName << n;
            Temp2.push_back(std::make_pair(new G4UnionSolid(unionName.str(), Temp1[2*i].first, Temp1[2*i+1].first, 0, Temp1[2*i+1].second-Temp1[2*i].second), Temp1[2*i].second));
            n++;
        }

        if(Add)
        {
            unionName.str(name);
            unionName << n;
            Temp2.push_back(std::make_pair(new G4UnionSolid(unionName.str(), Temp1[size].first, ToBeAdded[0].first, 0, ToBeAdded[0].second-Temp1[size].second), Temp1[size].second));
            ToBeAdded.pop_back();
            n++;
            Add = false;
        }
        Temp1 = Temp2;
        size = Temp1.size();
    }

    if(Temp1.size() == 1)
    {
        return std::make_pair(new G4UnionSolid(name, Temp1[0].first, ToBeAdded[0].first, 0, ToBeAdded[0].second-Temp1[0].second), Temp1[0].second);
    }
    else if(ToBeAdded.size() == 0)
    {
        return std::make_pair(new G4UnionSolid(name, Temp1[0].first, Temp1[1].first, 0, Temp1[1].second-Temp1[0].second), Temp1[0].second);
    }
    else
    {
        unionName.str(name);
        unionName << n;
        solidPos Temp = std::make_pair(new G4UnionSolid(unionName.str(), Temp1[0].first, Temp1[1].first, 0, Temp1[1].second-Temp1[0].second), Temp1[0].second);
        return std::make_pair(new G4UnionSolid(name, Temp.first, ToBeAdded[0].first, 0, ToBeAdded[0].second-Temp.second), Temp.second);
    }
}
