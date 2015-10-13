
#include <iomanip>

#include "StorkMaterial.hh"
#include "G4NistManager.hh"
#include "G4UnitsTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Constructor to create a material from scratch


StorkMaterial::StorkMaterial(const G4String& name, G4double z,
                       G4double a, G4double density,
                       G4State state, G4double temp, G4double pressure)
  : G4Material(name, z, a, density, state, temp, pressure)
{
    G4int index;
    G4Material* mat = dynamic_cast<G4Material*>(this);
    G4ElementVector* elemVec = const_cast<G4ElementVector*>(mat->GetElementVector());

    (*elemVec)[0]->GetElementTable()->pop_back();

    StorkElement* elem = new StorkElement(*((*elemVec)[0]));
    delete (*elemVec)[0];
    //When a G4Element object is deleted it sets the pointer in the G4ElementTable in the index that the element was added to when it was created to zero, so we must reAdd the new StorkElement here
    elem->GetElementTable()->back()=elem;
    elem->SetTemperature(-1);

    if(elem->Exists(temp,index))
    {
        (*elemVec)[0]=(*((*elemVec)[0]->GetElementTable()))[index];
        delete elem;
        (*elemVec)[0]->GetElementTable()->pop_back();
    }
    else
    {
        elem->SetTemperature(temp);
        std::stringstream ss;
        ss.str("");
        ss<<'T'<<elem->GetTemperature()<<'k';
        G4String elemName = name+ss.str();
        elem->SetName(elemName);
        (*elemVec)[0]=elem;
    }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Constructor to create a material from a List of constituents
// (elements and/or materials)  added with AddElement or AddMaterial

StorkMaterial::StorkMaterial(const G4String& name, G4double density,
                       G4int nComponents,
                       G4State state, G4double temp, G4double pressure)
  : G4Material(name, density, nComponents, state, temp, pressure)
{

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Constructor to create a material from base material

StorkMaterial::StorkMaterial(const G4String& name, G4double density,
                       StorkMaterial* bmat,
                       G4State state, G4double temp, G4double pressure)
  : G4Material(name, density, 1, state, temp, pressure)
{
    this->AddMaterial(bmat, 1.);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Fake default constructor - sets only member data and allocates memory
//                            for usage restricted to object persistency

StorkMaterial::StorkMaterial(__void__& fake)
  : G4Material(fake)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

StorkMaterial::~StorkMaterial()
{
  //  G4cout << "### Destruction of material " << fName << " started" <<G4endl;
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// AddElement -- composition by atom count

void StorkMaterial::AddElement(StorkElement* element, G4int nAtoms)
{
    G4int index;
    G4Material* mat = dynamic_cast<G4Material*>(this);

    if(element->GetTemperature()<0.)
    {
        std::stringstream ss;
        ss.str("");
        ss<<'T'<<GetTemperature()<<'k';
        element->SetTemperature(GetTemperature());
        element->SetName(element->GetName()+ss.str());
        mat->AddElement((element), nAtoms);
    }
    else if(element->Exists(GetTemperature(),index))
    {
        mat->AddElement(dynamic_cast<StorkElement*>((*(element->GetElementTable()))[index]), nAtoms);
    }
    else
    {
        std::stringstream ss;
        ss.str("");
        ss<<'T'<<element->GetTemperature()<<'k';
        G4String elemName = element->GetName(), check;
        int pos=elemName.find_last_of('T'), pos2=elemName.find_last_of('k');

        if((pos>0)&&(pos2>0)&&(pos2>pos))
            check= elemName.substr(pos, pos2-pos+1);

        StorkElement *newElem = new StorkElement(*element);

        if(check==G4String(ss.str()))
        {
            ss.str("");
            ss.clear();
            ss<<'T'<<GetTemperature()<<'k';
            newElem->SetName(elemName.substr(0, elemName.find_last_of('T'))+ss.str());
        }
        else
        {
            ss.str("");
            ss.clear();
            ss<<'T'<<GetTemperature()<<'k';
            newElem->SetName(element->GetName()+ss.str());
        }
        newElem->SetTemperature(GetTemperature());
        mat->AddElement((newElem), nAtoms);
    }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// AddElement -- composition by fraction of mass

void StorkMaterial::AddElement(StorkElement* element, G4double fraction)
{
    G4Material* mat = dynamic_cast<G4Material*>(this);
  // filling ...
    G4int index;
    if(element->GetTemperature()<0.)
    {
        element->SetTemperature(GetTemperature());
        std::stringstream elemName;
        elemName << element->GetName() << 'T' << element->GetTemperature() << 'k';
        element->SetName(elemName.str());
        mat->AddElement((element), fraction);
    }
    else if(element->Exists(GetTemperature(),index))
    {
        mat->AddElement(dynamic_cast<StorkElement*>((*(element->GetElementTable()))[index]), fraction);
    }
    else
    {
        std::stringstream ss;
        ss.str("");
        ss<<'T'<<element->GetTemperature()<<'k';
        G4String elemName = element->GetName(), check;
        int pos=elemName.find_last_of('T'), pos2=elemName.find_last_of('k');

        if((pos>0)&&(pos2>0)&&(pos2>pos))
            check= elemName.substr(pos, pos2-pos+1);

        StorkElement *newElem = new StorkElement(*element);

        if(check==G4String(ss.str()))
        {
            ss.str("");
            ss.clear();
            ss<<'T'<<GetTemperature()<<'k';
            newElem->SetName(elemName.substr(0, elemName.find_last_of('T'))+ss.str());
        }
        else
        {
            ss.str("");
            ss.clear();
            ss<<'T'<<GetTemperature()<<'k';
            newElem->SetName(element->GetName()+ss.str());
        }
        newElem->SetTemperature(GetTemperature());
        mat->AddElement((newElem), fraction);
    }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// AddMaterial -- composition by fraction of mass

void StorkMaterial::AddMaterial(StorkMaterial* material, G4double fraction)
{

G4int index;

G4Material* mat = dynamic_cast<G4Material*>(this);
G4Material* mat2 = dynamic_cast<G4Material*>(material);
G4String name = mat->GetName()+"-"+mat2->GetName();
G4Material* temp = new G4Material(name, mat2->GetDensity(), mat2->GetNumberOfElements(), mat2->GetState(), mat2->GetTemperature(), mat2->GetPressure());
const G4double* fracVec = mat2->GetFractionVector();

for (size_t elm=0; elm<mat2->GetNumberOfElements(); ++elm)
{
    StorkElement* element = static_cast<StorkElement*>((*(mat2->GetElementVector()))[elm]);
    if(element->Exists(this->GetTemperature(), index))
    {
        temp->AddElement(dynamic_cast<StorkElement*>((*(element->GetElementTable()))[index]), fracVec[elm]);
    }
    else
    {
        std::stringstream ss;
        ss.str("");
        ss<<'T'<<element->GetTemperature()<<'k';
        G4String elemName = element->GetName(), check="";
        int pos=elemName.find_last_of('T'), pos2=elemName.find_last_of('k');

        if((pos>0)&&(pos2>0)&&(pos2>pos))
            check= elemName.substr(pos, pos2-pos+1);

        StorkElement *newElem = new StorkElement(*element);

        if(check==G4String(ss.str()))
        {
            ss.str("");
            ss.clear();
            ss<<'T'<<this->GetTemperature()<<'k';
            newElem->SetName(elemName.substr(0, elemName.find_last_of('T'))+ss.str());
        }
        else
        {
            ss.str("");
            ss.clear();
            ss<<'T'<<this->GetTemperature()<<'k';
            newElem->SetName(element->GetName()+ss.str());
        }
        newElem->SetTemperature(this->GetTemperature());
        temp->AddElement((newElem), fracVec[elm]);
    }
}

mat->AddMaterial(temp, fraction);
delete temp;
(G4Material::GetMaterialTable())->pop_back();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void StorkMaterial::SetTemperature(G4double matTemp, G4bool UpdateElemTemp)
{
    //this should work but seems to change results, even when we use an alternative method by reassigning the logical volumes to point to new materials with the desired properties
    // need to fix when UpdateElemTemp=true
    G4MaterialTable *matTable = (G4MaterialTable*)G4Material::GetMaterialTable();
    int j = this->GetIndex();

    StorkMaterial *tempMat = new StorkMaterial(this->GetName()+"1", this->GetDensity(), this->GetNumberOfElements(), this->GetState(), this->GetTemperature(), this->GetPressure());

    G4Material* g4Mat2 = dynamic_cast<G4Material*>(tempMat);
    G4ElementVector* elemVec2 = const_cast<G4ElementVector*> (this->GetElementVector());
    G4double* fracVec2 = const_cast<G4double*> (this->GetFractionVector());
    for(G4int k=0; k<int(this->GetNumberOfElements()); k++)
    {
        g4Mat2->AddElement(dynamic_cast<StorkElement*>((*elemVec2)[k]), fracVec2[k]);
    }

    this->~StorkMaterial();
    matTable->erase(matTable->begin()+j);
    G4String realName = (tempMat->GetName()).substr(0, (tempMat->GetName()).size()-1);

    std::vector<G4Material*> tempMatTable(matTable->begin()+j, matTable->end());
    matTable->erase(matTable->begin()+j, matTable->end());

    if(UpdateElemTemp)
        new (this) StorkMaterial(realName, tempMat->GetDensity(), tempMat, tempMat->GetState(), matTemp, tempMat->GetPressure());
    else
        new (this) StorkMaterial(realName, tempMat->GetDensity(), tempMat->GetNumberOfElements(), tempMat->GetState(), matTemp, tempMat->GetPressure());

    matTable->insert(matTable->end(), tempMatTable.begin(), tempMatTable.end());

    if(!UpdateElemTemp)
    {
        G4Material* g4Mat = dynamic_cast<G4Material*>(this);
        G4ElementVector* elemVec = const_cast<G4ElementVector*> (tempMat->GetElementVector());
        G4double* fracVec = const_cast<G4double*> (tempMat->GetFractionVector());
        for(G4int k=0; k<int(tempMat->GetNumberOfElements()); k++)
        {
            g4Mat->AddElement(dynamic_cast<StorkElement*>((*elemVec)[k]), fracVec[k]);
        }
    }

    delete tempMat;
    matTable->pop_back();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

StorkMaterial::StorkMaterial(StorkMaterial& right): G4Material(right.GetName(), right.GetDensity(),
                       right.GetNumberOfElements(), right.GetState(), right.GetTemperature(), right.GetPressure())
{
    G4ElementVector* elemVec = const_cast<G4ElementVector*> (right.GetElementVector());
    G4double* fracVec = const_cast<G4double*> (right.GetFractionVector());
    for(G4int i=0; i<int(right.GetNumberOfElements()); i++)
    {
        this->AddElement(dynamic_cast<StorkElement*>((*elemVec)[i]), fracVec[i]);
    }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

const StorkMaterial& StorkMaterial::operator=(StorkMaterial& right)
{
    if (this != &right)
    {
        G4MaterialTable *matTable = (G4MaterialTable*)G4Material::GetMaterialTable();
        int j = this->GetIndex();
        G4String realName = this->GetName();

        this->~StorkMaterial();
        matTable->erase(matTable->begin()+j);

        std::vector<G4Material*> tempMatTable(matTable->begin()+j, matTable->end());
        matTable->erase(matTable->begin()+j, matTable->end());

        new (this) StorkMaterial(realName, right.GetDensity(), &right, right.GetState(), right.GetTemperature(), right.GetPressure());

        matTable->insert(matTable->end(), tempMatTable.begin(), tempMatTable.end());
    }
    return *this;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4int StorkMaterial::operator==(const StorkMaterial& right) const
{
  return (this == (StorkMaterial *) &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4int StorkMaterial::operator!=(const StorkMaterial& right) const
{
  return (this != (StorkMaterial *) &right);
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
