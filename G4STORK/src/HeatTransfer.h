//
//  HeatTransfer.h
//  G4-STORK-1.0
//
//  Created by Andrew Tan on 2014-11-09.
//  Copyright (c) 2014 andrewtan. All rights reserved.
//

#ifndef __G4_STORK_1_0__HeatTransfer__
#define __G4_STORK_1_0__HeatTransfer__

#include <iostream>
#include "StorkContainers.hh"
#include "StorkMaterialHT.hh"
#include "G4TransportationManager.hh"
#include "G4Tubs.hh"
#include "G4VSolid.hh"

class HeatTransfer
{
public:
    HeatTransfer(const G4LogicalVolume* FuelRod,const G4double Fo, const G4double Bi, const G4double T_Inf, const G4int SpatialSteps, const G4double heat);

    ~HeatTransfer();
    void Initialize(const G4double Fo, const G4double Bi, const G4double T_Inf, const G4int SpatialSteps);
    void SetHeatGeneration();
    void GetFissionSites();
    void OutputResults();
    void SetMaterialProperties();
    void StartCalculation();
    void ConvergenceRun(std::vector<G4double> array_, G4int precision);
    std::vector<G4double> ExplicitMethod(std::vector<G4double> array_);
    G4bool PrecisionCheck(std::vector<G4double> array1, std::vector<G4double> array2, G4int precision);



protected:

    G4double timeStep;
    G4double spatialStep;
    G4double FourierNumber;
    G4double BiotNumber;
    G4double thermalDiffusivity;
    G4double thermalConductivity;
    G4double avgTemperature;
    G4double HeatTransferCoefficient;
    G4double HeatFlux;
    G4double CoolentTemp;
    G4double heatGeneration;
    G4int numSteps;
    std::vector<G4double> InitialCondition;
    std::vector<G4double> Output;
};

#endif /* defined(__G4_STORK_1_0__HeatTransfer__) */
