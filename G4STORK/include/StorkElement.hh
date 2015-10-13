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
// $Id: StorkElement.hh 69704 2013-05-13 09:06:12Z gcosmo $
//

//---------------------------------------------------------------------------
//
// ClassName:   StorkElement
//
// Description: Contains element properties
//
// Class description:
//
// An element is a chemical element either directly defined in terms of
// its characteristics: its name, symbol,
//                      Z (effective atomic number)
//                      N (effective number of nucleons)
//                      A (effective mass of a mole)
// or in terms of a collection of constituent isotopes with specified
// relative abundance (i.e. fraction of nb of atoms per volume).
//
// Quantities, with physical meaning or not, which are constant in a given
// element are computed and stored here as Derived data members.
//
// The class contains as a private static member the table of defined
// elements (an ordered vector of elements).
//
// Elements can be assembled singly or in mixtures into materials used
// in volume definitions via the G4Material class.
//
// It is strongly recommended do not delete StorkElement instance in the
// user code. All StorkElements will be automatically deleted at the end
// of Geant4 session

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// 09-07-96, new data members added by L.Urban
// 17-01-97, aesthetic rearrangement, M.Maire
// 20-01-97, Tsai formula for the rad length, M.Maire
// 21-01-97, remove mixture flag, M.Maire
// 24-01-97, new data member: fTaul
//           new method: ComputeIonisationPara, M.Maire
// 20-03-97, corrected initialization of pointers, M.Maire
// 27-06-97, new function GetIsotope(int), M.Maire
// 24-02-98, fWeightVector becomes fRelativeAbundanceVector
// 27-04-98, atomic shell stuff, V. Grichine
// 09-07-98, Ionisation parameters removed from the class, M.Maire
// 04-08-98, new method GetElement(elementName), M.Maire
// 16-11-98, Subshell -> Shell, mma
// 30-03-01, suppression of the warning message in GetElement
// 17-07-01, migration to STL, M. Verderi
// 13-09-01, stl migration. Suppression of the data member fIndexInTable
// 14-09-01, fCountUse: nb of materials which use this element
// 26-02-02, fIndexInTable renewed
// 01-04-05, new data member fIndexZ to count the number of elements with same Z
// 17-10-06: Add Get/Set fNaturalAbundance (V.Ivanchenko)
// 17.09.09, add fNbOfShellElectrons and methods (V. Grichine)
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef StorkElement_HH
#define StorkElement_HH 1

#include "globals.hh"
#include <vector>
#include "G4ios.hh"
#include "G4Isotope.hh"
#include "G4IonisParamElm.hh"
#include "G4IsotopeVector.hh"
#include "G4Element.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class StorkElement : public G4Element
{
public:  // with description

  //
  // Constructor to Build an element directly; no reference to isotopes
  //
  StorkElement(const G4String& name,		//its name
            const G4String& symbol,		//its symbol
                  G4double  Zeff,		//atomic number
                  G4double  Aeff);		//mass of mole

  //
  // Constructor to Build an element from isotopes via AddIsotope
  //
  StorkElement(const G4String& name,		//its name
            const G4String& symbol,		//its symbol
            G4int nbIsotopes);			//nb of isotopes

  ~StorkElement();

  inline G4double GetTemperature() const {return temperature;}
  inline void SetTemperature(G4double temp) {temperature=temp;}

  inline G4double GetCSDataTemp() const {return csDataTemp;}
  inline void SetCSDataTemp(G4double CSDataTemp) {csDataTemp=CSDataTemp;}
/*
  inline const G4String& GetName() const {return (dynamic_cast<G4Element*>(this))->GetName();}
  inline void SetName(const G4String& name)  {dynamic_cast<G4Element*>(this)->SetName(name);}


  G4ElementTable* GetElementTable()
  {
    dynamic_cast<G4Element*>(this)->GetElementTable();
  }

*/

  bool Exists(G4double temp, G4int &index);

  // printing methods
  //
  /*
  friend std::ostream& operator<<(std::ostream&, StorkElement*);
  friend std::ostream& operator<<(std::ostream&, StorkElement&);
  friend std::ostream& operator<<(std::ostream&, G4ElementTable);
  */

public:  // without description

  G4int operator==(const StorkElement&) const;
  G4int operator!=(const StorkElement&) const;

  StorkElement(StorkElement&);
  StorkElement(G4Element&);
  StorkElement(__void__&);
    // Fake default constructor for usage restricted to direct object
    // persistency for clients requiring preallocation of memory for
    // persistifiable objects.

private:

  const StorkElement & operator=(StorkElement&);

private:

  G4double temperature;
  G4double csDataTemp;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
