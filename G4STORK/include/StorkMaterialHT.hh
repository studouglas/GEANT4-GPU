#ifndef StorkMaterialHT_H
#define StorkMaterialHT_H

#include "G4Material.hh"

class StorkMaterialHT: public G4Material
{
public:
    StorkMaterialHT(const G4String& name,
                  G4double density,
                  G4int nComponents,
                  G4double SHC = 0.,
                  G4double TC = 0.,
                  G4State state     = kStateUndefined,
                  G4double temp     = CLHEP::STP_Temperature,
                  G4double pressure = CLHEP::STP_Pressure):
    G4Material(name, density, nComponents, state, temp, pressure)
    {
        specificHeatCapacity = SHC;
        thermalConductivity = TC;
    }

    StorkMaterialHT(const G4String& name,
                  G4double z,
                  G4double a,
                  G4double density,
                  G4double SHC = 0.,
                  G4double TC = 0.,
                  G4State state     = kStateUndefined,
                  G4double temp     = CLHEP::STP_Temperature,
                  G4double pressure = CLHEP::STP_Pressure):
    G4Material(name, z, a, density, state, temp, pressure)
    {
        specificHeatCapacity = SHC;
        thermalConductivity = TC;
    }

    StorkMaterialHT(const G4String& name,
                  G4double density,
                  const G4Material* baseMaterial,
                  G4double SHC = 0.,
                  G4double TC = 0.,
                  G4State state     = kStateUndefined,
                  G4double temp     = CLHEP::STP_Temperature,
                  G4double pressure = CLHEP::STP_Pressure):
    G4Material(name, density, baseMaterial, state, temp, pressure)
    {
        specificHeatCapacity = SHC;
        thermalConductivity = TC;
    }

    ~StorkMaterialHT(){;}

    // Returns wanted properties of the material
    G4double GetVolumetricHeatCapacity(void) {return specificHeatCapacity*this->GetDensity();}
    G4double GetSpecificHeatCapacity(void) { return specificHeatCapacity; }
    G4double GetThermalConductivity(void) {return thermalConductivity; }
    G4double GetThermalDiffusivity(void) {return thermalConductivity/(specificHeatCapacity*this->GetDensity());}

protected:
    // Stores propertieas of the material
    G4double specificHeatCapacity;
    G4double thermalConductivity;
};

#endif // StorkMaterialHT_H
