#ifndef STORKDELAYEDNEUTRONDATA_H
#define STORKDELAYEDNEUTRONDATA_H

/*
 Decay constant U235 data taken from ENDF/B-VII.1 and fission yields for U235 and U238 for ENDF/B-V.
 Energy yields in MeV taken from ENDF/B-V for U235 and U238.
 0 = U235 thermal
 1 = U238 fast
 2 = Pu239 thermal
 3 = U233 thermal
 4 = U235 fast
 5 = Pu239 fast
 6 = U233 fast
 7 = Pu240 fast
 8 = Th232 fast
 */
// Delayed neutron data.

//Decay Constants (1/s)
const G4double DecayConstants[9][6] = { {0.013336, 0.032379, 0.12078, 0.30278, 0.84949, 2.8530},
    {0.0132, 0.0321, 0.139, 0.358, 1.41, 4.02},
    {0.0128, 0.0301, 0.124, 0.325, 1.12, 2.69},
    {0.0126, 0.0337, 0.139, 0.325, 1.13, 2.50},
    {0.0127, 0.0317, 0.115, 0.311, 1.40, 3.87},
    {0.0129, 0.0211, 0.134, 0.331, 1.26, 3.21},
    {0.0126, 0.0334, 0.131, 0.302, 1.27, 3.13},
    {0.0129, 0.0313, 0.135, 0.333, 1.36, 4.04},
    {0.0124, 0.0334, 0.121, 0.321, 1.21, 3.29},};

//Fission Yields (1/fission)
const G4double FissionYields[9][6] = {  {0.0006346, 0.003557, 0.00314,  0.006797, 0.002138, 0.0004342},
    {0.000572, 0.006028, 0.007128, 0.01707, 0.0099, 0.0033},
    {0.00057, 0.00197, 0.00166, 0.00184, 0.00034, 0.00022},
    {0.00063, 0.00351, 0.0031,  0.00672, 0.00211, 0.00043},
    {0.00024, 0.00176, 0.00136, 0.00207, 0.00065, 0.00022},
    {0.0006,  0.00192, 0.00159, 0.00222, 0.00051, 0.0016},
    {0.00054, 0.00564, 0.00667, 0.01599, 0.00927, 0.00309},
    {0.0028,  0.00238, 0.00162, 0.00315, 0.00119, 0.00024},
    {0.00169, 0.00744, 0.00769, 0.02212, 0.00853, 0.00213},};

//Summed Fission Yields
const G4double TotalYields[9] = {0.0167008, 0.043998, 0.0066, 0.0165, 0.0063, 0.00844, 0.0412, 0.01138, 0.0496};

//Summed Decay Constants
const G4double TotalDecayConstants[9] = {4.171765, 5.9723, 4.3019, 4.1403, 5.7404, 4.969, 4.879, 5.9122, 4.9878};

//Precursor constants FY/DecayConstant (only u235, add in more later)
const G4double PrecursorConstants[6] ={0.04758548290341932, 0.10985515303128571, 0.025997681735386653, 0.022448642578770064,
    0.0025168042001671594, 0.00015219067648089728};

const G4double TotPrecursorConstants = 0.20855595512550984;

//Energy Yields (MeV)
const G4double EnergyYields[2][6] = {{0.2680, 0.4394, 0.4094, 0.4017, 0.4017, 0.4017},
    {0.2837, 0.4381, 0.4341, 0.4172, 0.3593, 0.3593}};

#endif