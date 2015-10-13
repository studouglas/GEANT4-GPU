#ifndef StorkMaterial_HH
#define StorkMaterial_HH 1

#include <sstream>

#include "StorkElement.hh"
#include "G4Material.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class StorkMaterial : public G4Material
{
public:  // with description
  //
  // Constructor to create a material from single element
  //
  StorkMaterial(const G4String& name,				//its name
                   G4double  z, 				//atomic number
                   G4double  a,					//mass of mole
                   G4double  density, 				//density
                   G4State   state    = kStateUndefined,	//solid,gas
                   G4double  temp     = CLHEP::STP_Temperature,	//temperature
                   G4double  pressure = CLHEP::STP_Pressure);	//pressure

  //
  // Constructor to create a material from a combination of elements
  // and/or materials subsequently added via AddElement and/or AddMaterial
  //
  StorkMaterial(const G4String& name,				//its name
                   G4double  density, 				//density
                   G4int     nComponents,			//nbOfComponents
                   G4State   state    = kStateUndefined,	//solid,gas
                   G4double  temp     = CLHEP::STP_Temperature,	//temperature
                   G4double  pressure = CLHEP::STP_Pressure);	//pressure

  //
  // Constructor to create a material from the base material
  //
  StorkMaterial(const G4String& name,				//its name
                   G4double  density, 				//density
                    StorkMaterial* baseMaterial,			//base material
                   G4State   state    = kStateUndefined,	//solid,gas
                   G4double  temp     = CLHEP::STP_Temperature,	//temperature
                   G4double  pressure = CLHEP::STP_Pressure);	//pressure

  //
  // Add an element, giving number of atoms
  //
  void AddElement(StorkElement* element,				//the element
                  G4int      nAtoms);				//nb of atoms in
		    						// a molecule
  //
  // Add an element or material, giving fraction of mass
  //
  void AddElement (StorkElement* element ,				//the element
                   G4double   fraction);			//fractionOfMass

  void AddMaterial(StorkMaterial* material,			//the material
                   G4double   fraction);			//fractionOfMass

  ~StorkMaterial();
  //
  //printing methods
  //
  /*
  friend std::ostream& operator<<(std::ostream&, StorkMaterial*);
  friend std::ostream& operator<<(std::ostream&, StorkMaterial&);
  friend std::ostream& operator<<(std::ostream&, G4MaterialTable);
  */

public:  // without description

  G4int operator==(const StorkMaterial&) const;
  G4int operator!=(const StorkMaterial&) const;
  StorkMaterial(__void__&);
  // Fake default constructor for usage restricted to direct object
    // persistency for clients requiring preallocation of memory for
    // persistifiable objects.


  void SetTemperature(G4double matTemp, G4bool UpdateElemTemp);
  StorkMaterial(StorkMaterial&);
  const StorkMaterial& operator=(StorkMaterial&);

private:

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
