#include "UnionBinaryTree.hh"

UnionBinaryTree::UnionBinaryTree(solidList* List)
{
    Unions = List;
}

UnionBinaryTree::~UnionBinaryTree(void)
{
    delete Unions;
    delete priority;
}

// creates Union heirarchy
//Note: the order determines the rate of change of the priority with respect to distance along the specified axis and
//the direction of increase and decrease from the centre outwards (a positive order mean decreasing priority away from the centre)
//
solidPos UnionBinaryTree::GetUnionSolid(G4String name, G4int equalP, ShapeEnum regShape, G4double unitRegionDim[], G4double regionDim[], G4double offset, coorEnum axis, G4double order, G4double inPriority[], G4bool delRepeatedEntries)
{
    if(delRepeatedEntries)
        RemoveDuplicates();

    std::stringstream unionName;
    solidList Temp1 = *Unions, Temp2, Temp3;
    solidPos ToBeAdded;
    solidListItr Temp1Itr;
    regList regListTemp, regListTemp2;
    regionInfo regToBeAdded;
    regListItr regionListItr;
    numSolids = Temp1.size();
    priority = new G4double[numSolids];
    G4int n=0;
    G4bool check=false;

    if(equalP==0)
    {
        for(G4int i=0; i<numSolids; i++)
        {
            priority[i]=1;
        }
    }
    else if(equalP==1)
        priority = inPriority;
    else if(equalP==2)
        PriorityByVol(offset);
    else if(equalP==3)
        PriorityByPos(axis, order, offset);
    else
        G4cerr << "\n ### Invalid priority setting for union binary tree solid " << name << " ### \n";

    intVec regIndicies = createRegions(&Temp1, regShape, unitRegionDim, regionDim);

    for(G4int j=0; j<G4int(regIndicies.size()); j++)
    {
        Temp1Itr=Temp1.begin();
        regionListItr=regionList.begin();
        Temp2.assign(Temp1Itr+j, Temp1Itr+regIndicies[j]+j);
        regListTemp.assign(regionListItr+j, regionListItr+regIndicies[j]+j);
        Temp1.erase(Temp1Itr+j, Temp1Itr+regIndicies[j]+j);
        regionList.erase(regionListItr+j, regionListItr+regIndicies[j]+j);
        check=false;
        while(regIndicies[j]>1)
        {
            if(regIndicies[j]%2!=0)
            {
                regIndicies[j]--;

                if(!check)
                {
                    ToBeAdded=Temp2.front();
                    regToBeAdded=regListTemp.front();
                    Temp2.erase(Temp2.begin());
                    regListTemp.erase(regListTemp.begin());
                    check=true;
                }
                else
                {
                    ToBeAdded=Temp2.back();
                    regToBeAdded=regListTemp.back();
                    Temp2.pop_back();
                    regListTemp.pop_back();
                    check=false;
                }
            }
            for(G4int i=0; i<G4int(Temp2.size())/2; i++)
            {
                unionName.str("");
                unionName << n;

                regListTemp2.push_back(regionInfo(AddRegion(regListTemp[2*i+check+1].first, regListTemp[2*i+check].first, cylUnit, left), regListTemp[2*i+check+1].second + regListTemp[2*i+check].second));

                Temp3.push_back(solidPos(new StorkUnionSolid(name+unionName.str(), Temp2[2*i+1+check].first, Temp2[2*i+check].first, 0,
                                Temp2[2*i+check].second-Temp2[2*i+1+check].second, cylUnit, regListTemp2[i].first, Temp2[2*i+1+check].second), Temp2[2*i+1+check].second));
                regIndicies[j]--;
                n++;
            }
            Temp2=Temp3;
            regListTemp=regListTemp2;

            if(ToBeAdded.first!=NULL)
            {
                if(check)
                {
                    Temp2.insert(Temp2.begin(), ToBeAdded);
                    regListTemp.insert(regListTemp.begin(), regToBeAdded);
                    ToBeAdded=solidPos();
                    regToBeAdded=regionInfo();
                }
                else
                {
                    Temp2.insert(Temp2.end(), ToBeAdded);
                    regListTemp.insert(regListTemp.end(), regToBeAdded);
                    ToBeAdded=solidPos();
                    regToBeAdded=regionInfo();
                }
                regIndicies[j]++;
            }
//            G4cout<<"\n number of regions in row: " << regIndicies[j] << "\n";
            regListTemp2.clear();
            Temp3.clear();
        }
        Temp1.insert(Temp1Itr+j, Temp2.begin(), Temp2.end());
        Temp2.clear();
        regionList.insert(regionListItr+j, regListTemp.begin(), regListTemp.end());
        regListTemp.clear();
    }

    G4cout << "\n ### The position of all the Volumes ###\n";
    for(G4int j = 0; j<G4int(Temp1.size()); j++)
        {
            G4double temp = ((Temp1)[j].second)[0];
            G4double temp2 = ((Temp1)[j].second)[1];
            G4cout << "\n" << temp << ", " << temp2;
        }

    while(G4int(Temp1.size())>2)
    {
        if(Temp1.size()%2!=0)
        {
            if(!check)
            {
                ToBeAdded=Temp1.front();
                regToBeAdded=regionList.front();
                Temp1.erase(Temp1.begin());
                regionList.erase(regionList.begin());
                check=true;
            }
            else
            {
                ToBeAdded=Temp1.back();
                regToBeAdded=regionList.back();
                Temp1.pop_back();
                regionList.pop_back();
                check=false;
            }
        }

        for(G4int i=0; i< G4int(Temp1.size()/2) ; i++)
        {
            unionName.str("");
            unionName << n;

            regListTemp.push_back(regionInfo(AddRegion(regionList[2*i+1].first, regionList[2*i].first, cylUnit, up), regionList[2*i+1].second+regionList[2*i].second));

            Temp2.push_back(solidPos(new StorkUnionSolid(name+unionName.str(), Temp1[2*i+1].first, Temp1[2*i].first, 0,
                            Temp1[2*i].second-Temp1[2*i+1].second, cylUnit, regListTemp[i].first, Temp1[2*i+1].second), Temp1[2*i+1].second));
            regIndicies.pop_back();
            n++;
        }
        if(ToBeAdded.first!=NULL)
        {
            if(check)
            {
                Temp2.insert(Temp2.begin(), ToBeAdded);
                regListTemp.insert(regListTemp.begin(), regToBeAdded);
                ToBeAdded=solidPos();
                regToBeAdded=regionInfo();
            }
            else
            {
                Temp2.insert(Temp2.end(), ToBeAdded);
                regListTemp.insert(regListTemp.end(), regToBeAdded);
                ToBeAdded=solidPos();
                regToBeAdded=regionInfo();
            }
        }
        Temp1=Temp2;
        regionList=regListTemp;

        Temp2.clear();
        regListTemp.clear();
    }

    G4cout << "\n ### The position of all the Volumes ###\n";
    for(G4int j = 0; j<G4int(Temp1.size()); j++)
        {
            G4double temp = ((Temp1)[j].second)[0];
            G4double temp2 = ((Temp1)[j].second)[1];
            G4cout << "\n" << temp << ", " << temp2;
        }

    return solidPos(new StorkUnionSolid(name, Temp1[1].first, Temp1[0].first, 0,
                        Temp1[0].second-Temp1[1].second, cylUnit, AddRegion((regionList[1]).first,
                         (regionList[0]).first, cylUnit, up ), Temp1[1].second), Temp1[1].second) ;
}

void UnionBinaryTree::PriorityByVol(G4double offset)
{
    for(G4int i=0; i< G4int(Unions->size()); i++)
    {
        priority[i]=(((*Unions)[i]).first)->GetCubicVolume()+offset;
    }
}

void UnionBinaryTree::PriorityByPos(coorEnum axis, G4double order, G4double offset)
{
    if(G4int(axis)==0)
    {
        for(G4int i=0; i< G4int(Unions->size()); i++)
        {
            priority[i]=pow((0.000001+(((*Unions)[i]).second).mag()),(-0.25*order))+offset;
        }
    }
    else if(G4int(axis)==1)
    {
        for(G4int i=0; i< G4int(Unions->size()); i++)
        {
            priority[i]=pow((0.000001+(((*Unions)[i]).second).rho()),(-0.25*order))+offset;
        }
    }
    else if(G4int(axis)==2)
    {
        for(G4int i=0; i< G4int(Unions->size()); i++)
        {
            priority[i]=pow((0.000001+(((*Unions)[i]).second).phi()),(-0.5*order))+offset;
        }
    }
    else if(G4int(axis)==3)
    {
        for(G4int i=0; i< G4int(Unions->size()); i++)
        {
            priority[i]=pow((0.000001+(((*Unions)[i]).second).x()),(-0.25*order))+offset;
        }
    }
    else if(G4int(axis)==4)
    {
        for(G4int i=0; i< G4int(Unions->size()); i++)
        {
            priority[i]=pow((0.000001+(((*Unions)[i]).second).y()),(-0.25*order))+offset;
        }
    }
    else
    {
        for(G4int i=0; i < G4int(Unions->size()); i++)
        {
            priority[i]=pow((0.000001+(((*Unions)[i]).second).z()),(-0.25*order))+offset;
        }
    }
}

void UnionBinaryTree::RemoveDuplicates()
{
    std::map<solidPos,G4int> duplicateMap;
    for(solidListItr itr = Unions->begin(); itr<Unions->end(); itr++)
    {
        if(duplicateMap.find(*itr)==duplicateMap.end())
        {
            duplicateMap[*itr]=1;
        }
        else
        {
            Unions->erase(itr);
        }
    }
}

intVec UnionBinaryTree::createRegions(solidList* Temp1, ShapeEnum RegShape, G4double unitRegionDim[], G4double regionDim[])
{
    if(RegShape == cylUnit)
    {
        G4int regionIndices[3];
        regionIndices[0] = ceil((regionDim[1]-regionDim[0])/(unitRegionDim[1]-unitRegionDim[0]));
        regionIndices[1] = ceil((regionDim[3]-regionDim[2])/(unitRegionDim[3]-unitRegionDim[2]));
        regionIndices[2] = ceil((regionDim[5]-regionDim[4])/(unitRegionDim[5]-unitRegionDim[4]));

        unitRegionDim[0] = regionDim[1]-(regionDim[1]-regionDim[0])/(regionIndices[0]);
        unitRegionDim[1] = regionDim[1];
        unitRegionDim[2] = regionDim[3]-(regionDim[3]-regionDim[2])/(regionIndices[1]);
        unitRegionDim[3] = regionDim[3];
        unitRegionDim[4] = regionDim[5]-(regionDim[5]-regionDim[4])/(regionIndices[2]);
        unitRegionDim[5] = regionDim[5];

        G4double unitRegDim[6];
        intVec elemsRow(regionIndices[0], 0);

        for(G4int i=0; i<regionIndices[0]; i++)
        {
            unitRegDim[0]=regionDim[1]-(unitRegionDim[1]-unitRegionDim[0])*(i+1);
            unitRegDim[1]=regionDim[1]-(unitRegionDim[1]-unitRegionDim[0])*(i);
            elemsRow[i]=ceil((unitRegDim[0]*(regionDim[3]-regionDim[2]))/(unitRegionDim[0]*(unitRegionDim[3]-unitRegionDim[2])));

            for(G4int j=0; j<(elemsRow[i]); j++)
            {
                unitRegDim[2]=regionDim[3]-(j+1)*(regionDim[3]-regionDim[2])/(elemsRow[i]);
                unitRegDim[3]=regionDim[3]-(j)*(regionDim[3]-regionDim[2])/(elemsRow[i]);

                for(G4int k=0; k<regionIndices[2]; k++)
                {
                    unitRegDim[4]=regionDim[5]-(k+1)*(regionDim[5]-regionDim[4])/(regionIndices[2]);
                    unitRegDim[5]=regionDim[5]-(k)*(regionDim[5]-regionDim[4])/(regionIndices[2]);
                    StorkSixVector<G4double> unitRegDimTemp(unitRegDim);
                    regionList.push_back(regionInfo(unitRegDimTemp, 0.));
                }
            }
        }
//        G4cout<<"\n ### SIZE OF REGION LIST " << regionList.size() << "### \n";

        solidList Temp2 = *Temp1, Temp3;
        Temp1->clear();
        solidPos toBeAdded;

        G4double swap;
        solidPos swapP;
        // indexs min, second min and max priority
        G4int minimum = 0;

        for(G4int i = 0; i < G4int(Temp2.size()-1); i++)
        {
            minimum=i;
            for(G4int j = i; j < G4int(Temp2.size()); j++)
            {
                if(priority[j]<priority[minimum])
                    minimum=j;
            }

            swap = priority[minimum];
            swapP = Temp2[minimum];
            priority[minimum] = priority[i];
            Temp2[minimum] = Temp2[i];
            priority[i]=swap;
            Temp2[i]=swapP;
        }

        G4double polarPos[3];
        G4ThreeVector cartPos, regCent;
        std::stringstream unionName;
        G4VSolid* volume;
        G4int count =0;
        G4int count2=0, rowIndex=0, volCount=0;
        intVec volCopy(Temp2.size(), 0);

        G4cout << "\n ### The number of shapes to be placed in regions ###" << Temp2.size() << "\n";

        G4cout << "\n ### The position of all the Volumes ###\n";
        for(G4int j = 0; j<G4int(Temp2.size()); j++)
        {
            G4double temp = ((Temp2)[j].second)[0];
            G4double temp2 = ((Temp2)[j].second)[1];
            G4cout << "\n" << temp << ", " << temp2;
        }

        for(G4int j = 0; j<G4int(regionList.size()); j++)
        {
            if(count2==elemsRow[rowIndex])
            {
                if(elemsRow[rowIndex]==0)
                {
                    elemsRow.erase(elemsRow.begin()+rowIndex);
                    rowIndex--;
                }
                rowIndex++;
                count2=0;
            }
            count2++;

            for(G4int i = 0; i<G4int(Temp2.size()); i++)
            {
                if(InsideRegion((Temp2[i]).second, cylUnit, (regionList[j]).first))
                {
                    Temp3.push_back(Temp2[i]);
                    regionList[j].second +=(priority[i]);
                    volCount++;
                    volCopy[i]++;
                }
                else
                {
                // Finds the point on the region closest to the centre of the shape and  checks if it is inside
                    if(((Temp2[i]).second)[2]<((regionList[j]).first)[4])
                    {
                        polarPos[2]=((regionList[j]).first)[4];
                    }
                    else if(((Temp2[i]).second)[2]>((regionList[j]).first)[5])
                    {
                        polarPos[2]=((regionList[j]).first)[5];
                    }
                    else
                    {
                        polarPos[2]=((Temp2[i]).second)[2];
                    }
                    if(((Temp2[i]).second).phi()<((regionList[j]).first)[2])
                    {
                        polarPos[1]=((regionList[j]).first)[2];
                    }
                    else if(((Temp2[i]).second).phi()>((regionList[j]).first)[3])
                    {
                        polarPos[1]=((regionList[j]).first)[3];
                    }
                    else
                    {
                        polarPos[1]=((Temp2[i]).second).phi();
                    }
                    if(((Temp2[i]).second).rho()<((regionList[j]).first)[0])
                    {
                        polarPos[0]=((regionList[j]).first)[0];
                    }
                    else if(((Temp2[i]).second).rho()>((regionList[j]).first)[1])
                    {
                       polarPos[0]=((regionList[j]).first)[1];
                    }
                    else
                    {
                       polarPos[0]=((Temp2[i]).second).rho();
                    }
                    cartPos.setRhoPhiZ(polarPos[0],polarPos[1], polarPos[2]);
                    if(((Temp2[i]).first)->Inside(cartPos-Temp2[i].second)==kInside)
                    {
                        if(volCopy[i]>0)
                        {
                            G4String volShape = (Temp2[i].first)->GetEntityType();
                            StorkSixVector<G4double> newDim;
                            unionName.str("");
                            unionName << volCopy[i];

                            if(volShape=="G4Tubs")
                            {
                                G4Tubs *tempTube = dynamic_cast<G4Tubs*>(Temp2[i].first);
                                newDim.data[0] = (((tempTube)->GetInnerRadius())>0) ? ((tempTube)->GetInnerRadius())*(pow(1.0001,volCopy[i])):0.;
                                newDim.data[1] = (((tempTube)->GetOuterRadius()))*(pow(0.9999,volCopy[i]));
                                if(((tempTube)->GetDeltaPhiAngle())!=2*CLHEP::pi)
                                {
                                    newDim.data[2] = (((tempTube)->GetStartPhiAngle()))*(pow(1.0001,volCopy[i]));
                                    newDim.data[3] = (((tempTube)->GetDeltaPhiAngle()))*(pow(0.9999,volCopy[i]));
                                }
                                else
                                {
                                    newDim.data[2] = 0.;
                                    newDim.data[3] = 2*CLHEP::pi;
                                }

                                newDim.data[4] = (((tempTube)->GetZHalfLength()))*(pow(0.9999,volCopy[i]));
                                Temp3.push_back(solidPos(new G4Tubs("G4TubsMod"+unionName.str(), newDim[0], newDim[1], newDim[4], newDim[2], newDim[3]),Temp2[i].second));
                            }
                            else if(volShape=="G4Box")
                            {
                                G4cerr << "\n Missing code on line 486 of UnionBinaryTree.cc \n";
                            }
                            else if(volShape=="G4Sphere")
                            {
                                G4cerr << "\n Missing code on line 486 of UnionBinaryTree.cc \n";
                            }
                            else if(volShape=="G4Sphere")
                            {
                                G4cerr << "\n Missing code on line 486 of UnionBinaryTree.cc \n";
                            }

                        }
                        else
                        {
                            Temp3.push_back(Temp2[i]);
                        }
                        regionList[j].second += (priority[i]*0.5);
                        volCount++;
                        volCopy[i]++;

                    }
                }
            }
            if(toBeAdded.first!=NULL)
            {

                Temp3.insert(Temp3.begin(), toBeAdded);
                regionList[j] = regionInfo(AddRegion(regionList[j].first, regionList[j-1].first, cylUnit, left), regionList[j-1].second+regionList[j].second);
                regionList.erase(regionList.begin()+j-1);
                j--;
                toBeAdded=solidPos();
            }
            if(Temp3.size()>1)
            {
                if(Temp3.size()>2)
                {
                    unionName.str("");
                    unionName << count;
                    count++;
                    volume = new G4UnionSolid("volume"+unionName.str(), Temp3[1].first, Temp3[0].first, 0, Temp3[0].second-Temp3[1].second);

                    G4int i=2;
                    for(; i<G4int(Temp3.size()-1); i++)
                    {
                        unionName.str("");
                        unionName << count;
                        count++;
                        volume = new G4UnionSolid("volume"+unionName.str(), Temp3[i].first, volume, 0, Temp3[i-1].second-Temp3[i].second);
                    }
                    unionName.str("");
                    unionName << count;
                    count++;

                    Temp1->push_back(solidPos(new StorkUnionSolid("volume"+unionName.str(), Temp3[i].first, volume, 0, Temp3[i-1].second-Temp3[i].second, cylUnit,
                                    regionList[j].first, Temp3[i].second ), Temp3[i].second));
                }
                else
                {
                    unionName.str("");
                    unionName << count;
                    count++;

                    Temp1->push_back(solidPos(new StorkUnionSolid("volume"+unionName.str(), Temp3[1].first, Temp3[0].first, 0, Temp3[0].second-Temp3[1].second, cylUnit,
                                    regionList[j].first, Temp3[1].second ), Temp3[1].second));
                }


            }
            else if(Temp3.size()==1)
            {
                if(elemsRow[rowIndex]!=count2)
                {
                    toBeAdded=Temp3[0];
                }
                else if(count2>1)
                {
                    unionName.str("");
                    unionName << count;
                    count++;

                    (*Temp1)[Temp1->size()-1] = solidPos(new StorkUnionSolid("volume"+unionName.str(), (*Temp1)[Temp1->size()-1].first, Temp3[0].first, 0, Temp3[0].second-(*Temp1)[Temp1->size()-1].second, cylUnit,
                                    AddRegion(regionList[j-1].first, regionList[j].first, cylUnit, right), (*Temp1)[Temp1->size()-1].second), (*Temp1)[Temp1->size()-1].second);
                }
                else
                {
                    for(G4int i=0; i<elemsRow[rowIndex+1]; i++)
                    {
                      regionList[j+1+i] = regionInfo(AddRegion(regionList[j+1+i].first,regionList[j].first, cylUnit, up),regionList[j+1+i].second+regionList[j].second);
                    }
                }
                elemsRow[rowIndex]--;
                count2--;
            }
            else
            {
                if(elemsRow[rowIndex]!=count2)
                {
                    regionList[j+1] = regionInfo(AddRegion(regionList[j+1].first,regionList[j].first, cylUnit, left),regionList[j+1].second+regionList[j].second);
                }
                else if(count2>1)
                {
                    StorkUnionSolid* tempSolid = dynamic_cast<StorkUnionSolid*>((*Temp1)[Temp1->size()-1].first);
                    tempSolid->AddRegionToMe(right, regionList[j].first);
                    (*Temp1)[Temp1->size()-1].first = tempSolid;
                }
                else
                {
                    for(G4int i=0; i<elemsRow[rowIndex+1]; i++)
                    {
                      regionList[j+1+i] = regionInfo(AddRegion(regionList[j+1+i].first,regionList[j].first, cylUnit, up),regionList[j+1+i].second+regionList[j].second);
                    }
                }
                regionList.erase(regionList.begin()+j);
                j--;
                elemsRow[rowIndex]--;
                count2--;
            }
            Temp3.clear();
        }
        for(G4int d=0; d< G4int(volCopy.size()); d++)
        {
            if(volCopy[d]==0)
            {
                G4cerr << "\n volume " << d << " of " << volCopy.size() << " was not placed \n";
            }
        }
        G4cout << "\n ### The number of shapes placed in regions ### " << volCount << "\n";
        G4cout << "\n ### The position of all the Volumes ###\n";
        for(G4int j = 0; j<G4int(Temp1->size()); j++)
        {
            G4double temp = ((*Temp1)[j].second)[0];
            G4double temp2 = ((*Temp1)[j].second)[1];
            G4cout << "\n" << temp << ", " << temp2;
        }
        Temp2.clear();
        return elemsRow;
    }
    else
    {
        G4cerr<<"Region Shape has not been added to the UnionBinaryTree code";
    }
    return intVec();
}

StorkSixVector<G4double> UnionBinaryTree::AddRegion( StorkSixVector<G4double> regionDim, StorkSixVector<G4double> regionDim2, ShapeEnum shape, DirEnum dir )
{
    StorkSixVector<G4double> tmpRegDim;
    tmpRegDim=regionDim;
    if(dir==right)
    {
        if(shape==cylUnit)
        {
            (tmpRegDim.data)[2]=(regionDim2.data)[2];
        }
        else if(shape==cubicUnit)
        {
            (tmpRegDim.data)[1]+=(regionDim2.data)[1]-(regionDim2.data)[0];
        }
        else
        {
            (tmpRegDim.data)[2]=(regionDim2.data)[2];
        }
    }
    else if(dir==left)
    {
        if(shape==cylUnit)
        {
            (tmpRegDim.data)[3]=(regionDim2.data)[3];
        }
        else if(shape==cubicUnit)
        {
            (tmpRegDim.data)[0]+=(regionDim2.data)[0]-(regionDim2.data)[1];
        }
        else
        {
            (tmpRegDim.data)[3]=(regionDim2.data)[3];
        }
    }
    else if(dir==up)
    {
        if(shape==cylUnit)
        {
            (tmpRegDim.data)[1]=(regionDim2.data)[1];
        }
        else if(shape==cubicUnit)
        {
            (tmpRegDim.data)[3]+=(regionDim2.data)[3]-(regionDim2.data)[2];
        }
        else
        {
            (tmpRegDim.data)[1]=(regionDim2.data)[1];
        }
    }
    else if(dir==down)
    {
        if(shape==cylUnit)
        {
            (tmpRegDim.data)[0]=(regionDim2.data)[0];
        }
        else if(shape==cubicUnit)
        {
            (tmpRegDim.data)[2]+=(regionDim2.data)[2]-(regionDim2.data)[3];
        }
        else
        {
            (tmpRegDim.data)[0]=(regionDim2.data)[0];
        }
    }
    else if(dir==above)
    {
        if(shape==cylUnit)
        {
            (tmpRegDim.data)[5]+=(regionDim2.data)[5]-(regionDim2.data)[4];
        }
        else if(shape==cubicUnit)
        {
            (tmpRegDim.data)[5]+=(regionDim2.data)[5]-(regionDim2.data)[4];
        }
        else
        {
            (tmpRegDim.data)[4]=(regionDim2.data)[4];
        }
    }
    else if(dir==below)
    {
        if(shape==cylUnit)
        {
            (tmpRegDim.data)[4]+=(regionDim2.data)[4]-(regionDim2.data)[5];
        }
        else if(shape==cubicUnit)
        {
            (tmpRegDim.data)[4]+=(regionDim2.data)[2];
        }
        else
        {
            (tmpRegDim.data)[5]=(regionDim2.data)[5];
        }
    }
    return tmpRegDim;
}

G4bool UnionBinaryTree::InsideRegion( const G4ThreeVector& p, ShapeEnum regShape, StorkSixVector<G4double> regDim )
{
    G4double kCarTolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
    G4double kAngTolerance = G4GeometryTolerance::GetInstance()->GetAngularTolerance();
    static const G4double delta=0.5*kCarTolerance;
    static const G4double delta2=0.5*kAngTolerance;

    if(regShape==0)
    {
        if(p.rho()>regDim[0]-delta&&p.rho()<regDim[1]+delta&&p.phi()>regDim[2]-delta2&&p.phi()<regDim[3]+delta2&&p.z()>regDim[4]-delta&&p.z()<regDim[5]+delta)
        {
            return true;
        }
    }
    else if(regShape==1)
    {
        if(p.x()>regDim[0]-delta&&p.x()<regDim[1]+delta&&p.y()>regDim[2]-delta&&p.y()<regDim[3]+delta&&p.z()>regDim[4]-delta&&p.z()<regDim[5]+delta)
        {
            return true;
        }
    }

    else
    {
        if(p.r()>regDim[0]-delta&&p.r()<regDim[1]+delta&&p.phi()>regDim[2]-delta2&&p.phi()<regDim[3]+delta2&&p.theta()>regDim[4]-delta2&&p.theta()<regDim[5]+delta2)
        {
            return true;
        }
    }

    return false;

}



