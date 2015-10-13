/*
StorkHeatTransfer.hh

Definition for StorkHeatTransfer class.

This is the basics class that is used to keep track of the temperature of
the different geometries in the nuclear reactor. It's a simple grid representation
of the reactor where each mesh is given a specific temperature and the heat equation
is used to track keep track of heat transfer with time
*/

#ifndef STORKHEATTRANSFER_H
#define STORKHEATTRANSFER_H

#include "StorkWorld.hh"
#include "StorkContainers.hh"
#include "StorkMaterialHT.hh"
#include "G4TransportationManager.hh"
#include <math.h>
#include <string.h>
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

class StorkHeatTransfer
{
    public:
        // Default constructor
        StorkHeatTransfer(const StorkParseInput* fIn);
        ~StorkHeatTransfer(void);

        void InitializeSlowpokeModel(const StorkParseInput* fIn);
        void RunThermalCalculation(MSHSiteVector fissionSites);
        void RunSlowpokeModel();
        void SetWorld(StorkWorld* world);
        void SetFuelDimensions(G4ThreeVector fDim) {fuelDimensions = fDim;}


    protected:
        void CalcHeatDistribution();
        void UpdateFuelProperties(G4double FuelTemperatures[],G4double FuelDensities[],G4double FuelRadii[]);
        G4double CalcFuelTemperature(G4double heatGeneration, G4double mass, G4double surfaceArea, G4double oldTemp, G4double heatcapacity);
        G4double CalcFuelDensity(G4double temperature);
        G4double CalcFuelRadius(G4double oldRadius, G4double oldDensity, G4double newDensity);
        void SaveSLOWPOKEFuelProperties(G4String filename);
        G4double CalcEffectiveHeatGenerated(G4double currentFissions);


    protected:
        StorkWorld* theWorld;
        MSHSiteVector* fSites;
        G4double reactorPower;
        G4String worldname;
        G4double dt;
        G4double heatTransferCoeff;
        G4double temp_infty;
        std::map< G4String, G4double > fnDistribution;
        std::vector<G4String> fnMaterialList;
        G4double effectiveFissionEnergy;
        G4ThreeVector fuelDimensions;
        G4bool createHeader;
        G4bool saveMaterialData;
        G4String saveMaterialfilename;
        G4double fuelTempAvg;
        G4double fuelDensityAvg;
        G4double fuelRadiusAvg;
        G4double fissionToEnergyCoeff;
        G4double baselineFissionRate;

};

#endif // STORKHEATTRANSFER_H
