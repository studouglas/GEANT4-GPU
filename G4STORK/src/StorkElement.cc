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
// $Id: StorkElement.cc 81839 2014-06-06 08:47:44Z gcosmo $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// 26-06-96: Code uses operators (+=, *=, ++, -> etc.) correctly, P. Urban
// 09-07-96: new data members added by L.Urban
// 17-01-97: aesthetic rearrangement, M.Maire
// 20-01-97: Compute Tsai's formula for the rad length, M.Maire
// 21-01-97: remove mixture flag, M.Maire
// 24-01-97: ComputeIonisationParameters().
//           new data member: fTaul, M.Maire
// 29-01-97: Forbidden to create Element with Z<1 or N<Z, M.Maire
// 20-03-97: corrected initialization of pointers, M.Maire
// 28-04-98: atomic subshell binding energies stuff, V. Grichine
// 09-07-98: Ionisation parameters removed from the class, M.Maire
// 16-11-98: name Subshell -> Shell; GetBindingEnergy() (mma)
// 09-03-01: assignement operator revised (mma)
// 02-05-01: check identical Z in AddIsotope (marc)
// 03-05-01: flux.precision(prec) at begin/end of operator<<
// 13-09-01: suppression of the data member fIndexInTable
// 14-09-01: fCountUse: nb of materials which use this element
// 26-02-02: fIndexInTable renewed
// 30-03-05: warning in GetElement(elementName)
// 15-11-05: GetElement(elementName, G4bool warning=true)
// 17-10-06: Add fNaturalAbundances (V.Ivanchenko)
// 27-07-07: improve destructor (V.Ivanchenko)
// 18-10-07: move definition of material index to ComputeDerivedQuantities (VI)
// 25.10.11: new scheme for G4Exception  (mma)
// 05-03-12: always create isotope vector (V.Ivanchenko)

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include <iomanip>
#include <sstream>

#include "StorkElement.hh"
#include "G4AtomicShells.hh"
#include "G4NistManager.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//G4ElementTable StorkElement::theElementTable;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Constructor to Generate an element from scratch

StorkElement::StorkElement(const G4String& name, const G4String& symbol, G4double zeff, G4double aeff)
  : G4Element(name, symbol, zeff, aeff)
{
  temperature = -1;
  csDataTemp = -1;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Constructor to Generate element from a List of 'nIsotopes' isotopes, added
// via AddIsotope

StorkElement::StorkElement(const G4String& name, const G4String& symbol, G4int nIsotopes)
  : G4Element( name, symbol, nIsotopes)
{
  temperature = -1;
  csDataTemp = -1;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Fake default constructor - sets only member data and allocates memory
//                            for usage restricted to object persistency

StorkElement::StorkElement( __void__& fake)
  : G4Element(fake)
{
  temperature = -1;
  csDataTemp = -1;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

StorkElement::~StorkElement()
{

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

bool StorkElement::Exists(G4double temp, G4int &index)
{
    std::stringstream ss;
    ss.str("");
    ss<<'T'<<GetTemperature()<<'k';
    G4String elemName = GetName(), check;
    int pos=elemName.find_last_of('T'), pos2=elemName.find_last_of('k');

    if((pos>0)&&(pos2>0)&&(pos2>pos))
        check= elemName.substr(pos, pos2-pos+1);

    if(check==G4String(ss.str()))
    {
        elemName=elemName.substr(0, elemName.find_last_of('T'));
    }
    G4ElementTable *elemTable = (dynamic_cast<G4Element*>(const_cast<StorkElement*>(this)))->GetElementTable();
    for(G4int i=0; i<int(elemTable->size()); i++)
    {
        StorkElement *elem = dynamic_cast <StorkElement*> ((*elemTable)[i]);
        if(elem)
        {
            ss.str("");
            ss.clear();
            ss<<'T'<<elem->GetTemperature()<<'k';
            G4String elemName2 = elem->GetName(); check="";
            pos=elemName2.find_last_of('T'); pos2=elemName2.find_last_of('k');

            if((pos>0)&&(pos2>0)&&(pos2>pos))
                check= elemName2.substr(pos, pos2-pos+1);

            if(check==G4String(ss.str()))
            {
                elemName2=elemName2.substr(0, elemName2.find_last_of('T'));
            }

            if((elemName2==elemName)&&(elem->GetTemperature()==temp))
            {
                index=i;
                return true;
            }
        }
    }

    return false;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


StorkElement::StorkElement(StorkElement& right): G4Element(right.GetName(),
                     right.GetSymbol(), right.GetIsotopeVector()->size())
{
  G4IsotopeVector* isoVec = right.GetIsotopeVector();
  G4double* relVec = right.GetRelativeAbundanceVector();

  for(G4int i=0; i<int(right.GetIsotopeVector()->size()); i++)
  {
    this->AddIsotope((*isoVec)[i], relVec[i]);
  }
  temperature = right.GetTemperature();
  csDataTemp = right.GetCSDataTemp();
}

StorkElement::StorkElement(G4Element& right): G4Element(right.GetName(),
                     right.GetSymbol(), right.GetIsotopeVector()->size())
{
  G4IsotopeVector* isoVec = right.GetIsotopeVector();
  G4double* relVec = right.GetRelativeAbundanceVector();

  for(G4int i=0; i<int(right.GetIsotopeVector()->size()); i++)
  {
    this->AddIsotope((*isoVec)[i], relVec[i]);
  }
  temperature = -1;
  csDataTemp = -1;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

const StorkElement& StorkElement::operator=( StorkElement& right)
{
  if (this != &right)
    {
        G4ElementTable *elemTable = (G4ElementTable*)G4Element::GetElementTable();
        int j = this->GetIndex();
        G4String realName = this->GetName();

        this->~StorkElement();
        elemTable->erase(elemTable->begin()+j);

        std::vector<G4Element*> tempElemTable(elemTable->begin()+j, elemTable->end());
        elemTable->erase(elemTable->begin()+j, elemTable->end());

        new (this) StorkElement(realName, right.GetSymbol(), right.GetIsotopeVector()->size());

        elemTable->insert(elemTable->end(), tempElemTable.begin(), tempElemTable.end());
        G4IsotopeVector* isoVec = right.GetIsotopeVector();
        G4double* relVec = right.GetRelativeAbundanceVector();

        for(G4int i=0; i<int(right.GetIsotopeVector()->size()); i++)
        {
            this->AddIsotope((*isoVec)[i], relVec[i]);
        }
        temperature = right.GetTemperature();
        csDataTemp = right.GetCSDataTemp();
    }
  return *this;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4int StorkElement::operator==(const StorkElement& right) const
{
  return (this == (StorkElement*) &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4int StorkElement::operator!=(const StorkElement& right) const
{
  return (this != (StorkElement*) &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
/*
std::ostream& operator<<(std::ostream& flux, StorkElement* elem)
{
  G4Element* element = dynamic_cast<G4Element*>(elem)

  std::ios::fmtflags mode = flux.flags();
  flux.setf(std::ios::fixed,std::ios::floatfield);
  G4long prec = flux.precision(3);

  flux
    << " Element: " << element->GetName()   << " (" << element->fSymbol << ")"
    << "   Z = " << std::setw(4) << std::setprecision(1) <<  element->fZeff
    << "   N = " << std::setw(5) << std::setprecision(1) <<  element->fNeff
    << "   A = " << std::setw(6) << std::setprecision(2)
                 << (element->fAeff)/(g/mole) << " g/mole";

  for (size_t i=0; i<element->fNumberOfIsotopes; i++)
  flux
    << "\n         ---> " << (*(element->theIsotopeVector))[i]
    << "   abundance: " << std::setw(6) << std::setprecision(2)
    << (element->fRelativeAbundanceVector[i])/perCent << " %";

  flux.precision(prec);
  flux.setf(mode,std::ios::floatfield);
  return flux;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

 std::ostream& operator<<(std::ostream& flux, StorkElement& element)
{
  flux << &element;
  return flux;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

std::ostream& operator<<(std::ostream& flux, G4ElementTable ElementTable, G4bool Stork)
{
 //Dump info for all known elements
   flux << "\n***** Table : Nb of elements = " << ElementTable.size()
        << " *****\n" << G4endl;
    if(Stork)
    {
       for (size_t i=0; i<ElementTable.size(); i++)
       {
        flux << dynamic_cast<StorkElement*>(ElementTable[i])<< G4endl << G4endl;
       }
    }

    else
    {
       for (size_t i=0; i<ElementTable.size(); i++)
       {
        flux <<ElementTable[i]<< G4endl << G4endl;
       }
    }


   return flux;
}
*/
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
