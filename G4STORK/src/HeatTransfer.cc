//
//  HeatTransfer.cc
//  G4-STORK-1.0
//
//  Created by Andrew Tan on 2014-11-09.
//  Copyright (c) 2014 andrewtan. All rights reserved.
//

#include "HeatTransfer.h"

//Constructor
HeatTransfer::HeatTransfer(const G4LogicalVolume* FuelRod,const G4double Fo, const G4double Bi, const G4double T_Inf, const G4int NumSpatialSteps, const G4double heat)
{
    G4Tubs* theSolid = dynamic_cast<G4Tubs*>(FuelRod->GetSolid());
    StorkMaterialHT* theMaterial = dynamic_cast<StorkMaterialHT*>(FuelRod->GetMaterial());
    G4double radius = theSolid->GetInnerRadius();

    numSteps = NumSpatialSteps;
    HeatFlux = heat;
    FourierNumber = Fo;
    BiotNumber = Bi;
    CoolentTemp = T_Inf;
    avgTemperature = theMaterial->GetTemperature();
    spatialStep = G4double(radius/numSteps);

    thermalConductivity = theMaterial->GetThermalConductivity();
    thermalDiffusivity = theMaterial->GetThermalDiffusivity();

    timeStep = (1/thermalDiffusivity)*FourierNumber*spatialStep*spatialStep;

    InitialCondition = std::vector<G4double> (avgTemperature,numSteps);

}

//Destructor
HeatTransfer::~HeatTransfer()
{
}

std::vector<G4double> HeatTransfer::ExplicitMethod(std::vector<G4double> array)
{

    //Begin explicit method heat transfer

    G4double temp = FourierNumber*(2*array[1]+(1/thermalConductivity)*heatGeneration*spatialStep*spatialStep)+(1-2*FourierNumber)*array[0];

    Output.push_back(temp);

    for(G4int i = 1; i<(numSteps-1);i++)
    {
        temp =  FourierNumber*(array[i-1]+array[i+1]+(1/thermalConductivity)*heatGeneration*spatialStep*spatialStep) + (1 - 2*FourierNumber)*array[i];
        Output.push_back(temp);

    }
    temp = 2*FourierNumber*(array[numSteps-2] + BiotNumber*CoolentTemp + (0.5/thermalConductivity)*heatGeneration*spatialStep*spatialStep) + (1 - 2*FourierNumber - 2*BiotNumber*FourierNumber)*array[numSteps-1];
    Output.push_back(temp);

    return Output;
}

G4bool HeatTransfer::PrecisionCheck(std::vector<G4double> array1, std::vector<G4double> array2,G4int precision)
{
    if(array1.size() != array2.size()){
        G4cout<< "Error: arrays are different sizes!" << G4endl;
        return false;
    }
    G4int size = array1.size();

    for(G4int i = 0; i<size; i++)
    {
        if(pow(array1[i]-array2[i],2)>pow(10,precision))
        {
            return false;
        }
    }
    return true;
}

void HeatTransfer::ConvergenceRun(std::vector<G4double> array_, G4int precision)
{
    std::vector<G4double> temparray1 = ExplicitMethod(array_);
    std::vector<G4double> temparray2;
    G4bool converged = false;
    while(!converged){
        temparray2 = ExplicitMethod(temparray1);
        converged = PrecisionCheck(temparray1, temparray2, precision);
        temparray1 = temparray2;
    }
}

