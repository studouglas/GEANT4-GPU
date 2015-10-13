
/*
InfiniteUniformLatticeConstructor.cc

Created by:		Liam Russell
Date:			23-05-2012
Modified:       11-03-2013

Source code for the InfiniteUniformLatticeConstructor class.

*/

#include "InfiniteUniformLatticeConstructor.hh"


// Constructor
InfiniteUniformLatticeConstructor::InfiniteUniformLatticeConstructor()
: StorkVWorldConstructor(), reactorLogical(0)
{
	// Set default member variables
    matTemp = 293.6*kelvin;
    matDensity = 18.9*g/cm3;
    u235Conc = 0.05*perCent;
    hwConc = 90.0*perCent;
    latticePitch = 100*cm;

    reactorVisAtt=NULL;
}


// Destructor
InfiniteUniformLatticeConstructor::~InfiniteUniformLatticeConstructor()
{
    // Delete visualization attributes
    if(reactorVisAtt)
        delete reactorVisAtt;
}


// ConstructNewWorld()
// Build the infinite uniform lattice for the first time.  Set the user input
// choices and the variable properties map.
G4VPhysicalVolume* InfiniteUniformLatticeConstructor::ConstructNewWorld(
                                                  const StorkParseInput* infile)
{
    // Select material based on input file
    switch(infile->GetReactorMaterial())
    {
    	case 9:
			materialID = "EU";
			break;
        default:
			materialID = "UHW";
			break;

    }

    // Set up variable property map
    variablePropMap[MatPropPair(all,temperature)] = &matTemp;
    variablePropMap[MatPropPair(all,density)] = &matDensity;
    variablePropMap[MatPropPair(all,dimension)] = &latticePitch;
    variablePropMap[MatPropPair(fuel,concentration)] = &u235Conc;
    variablePropMap[MatPropPair(moderator,concentration)] = &hwConc;


    // Call base class ConstructNewWorld() to complete construction
    return StorkVWorldConstructor::ConstructNewWorld(infile);
}


// ConstructWorld()
// Build the materials and geometry for the infinite uniform lattice.
G4VPhysicalVolume* InfiniteUniformLatticeConstructor::ConstructWorld()
{
	// local variables and enclosed world dimensions
    G4double buffer = 1.0*cm;
    G4double side = latticePitch/2.0;
    reactorDim = G4ThreeVector(side, side, side);
    encWorldDim = 2.0 * G4ThreeVector(side+buffer,side+buffer,side+buffer);
    G4SolidStore* theSolids = G4SolidStore::GetInstance();

    // Set up the materials (if necessary)
    if(matChanged)
    {
    	// Delete any existing materials
    	DestroyMaterials();
    	// Create the materials
    	ConstructMaterials();
    }

	// Clean up volumes
	G4GeometryManager::GetInstance()->OpenGeometry();
	G4PhysicalVolumeStore::GetInstance()->Clean();
	G4LogicalVolumeStore::GetInstance()->Clean();

	// Set up the solids if necessary
	if(geomChanged)
	{
		// Clean up solids
		G4SolidStore::GetInstance()->Clean();

		// Create world box
		new G4Box("worldCube", side+buffer, side+buffer, side+buffer);
		// Create lattice cell
		new G4Box("FuelCube", side, side, side);

		geomChanged = false;
	}

    // Create world volume
    worldLogical = new G4LogicalVolume(theSolids->GetSolid("worldCube"),
									   matMap["World"],"worldLogical",0,0,0);
    worldPhysical = new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), worldLogical,
                                      "worldPhysical",0,0,0);

    // Create the homogenous cube reactor
    reactorLogical = new G4LogicalVolume(theSolids->GetSolid("FuelCube"),
										 matMap[materialID],"reactorLogical",
										 0,0,0);
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), reactorLogical,
                      "reactorPhysical", worldLogical,0,0);


    // Add sensitive detector to ALL logical volumes
	worldLogical->SetSensitiveDetector( sDReactor );
	reactorLogical->SetSensitiveDetector( sDReactor );

    // Set visualization attributes

    if(worldVisAtt)
        delete worldVisAtt;
    if(reactorVisAtt)
        delete reactorVisAtt;

    worldVisAtt = new G4VisAttributes(G4Colour(1.,1.,1.));
    worldVisAtt->SetVisibility(false);
    worldLogical->SetVisAttributes(worldVisAtt);

    reactorVisAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    reactorLogical->SetVisAttributes(reactorVisAtt);


    return worldPhysical;
}


// ConstructMaterials()
// Define and build all materials for the infinite uniform lattice.
void InfiniteUniformLatticeConstructor::ConstructMaterials()
{
	G4double massFracU = 10.73*perCent;
    G4double massFracW = 100.0*perCent - massFracU;
    G4double abunD20 = hwConc;
    G4double abunH20 = 1.0-hwConc;
    G4double abunU235 = u235Conc;
    G4double abunU238 = 1.0-u235Conc;
    G4double density;

    // Elements, isotopes and materials
    G4Isotope *H, *D2, *O16, *U235, *U238;
    G4Element *Uranium, *Oxygen, *Deuterium, *Hydrogen, *UMod;
    G4Material *World, *UHW, *LightWater, *HeavyWater, *EU;


    // Create the world environment
    World = new G4Material("Galactic", 1, 1, 1.e-25*g/cm3, kStateGas,
                                      2.73*kelvin, 3.e-18*pascal);

	// Make uranium isotopes and elements

    U235 = new G4Isotope("U235", 92, 235, 235.0439*g/mole);
    U238 = new G4Isotope("U238", 92, 238, 238.0508*g/mole);

    Uranium = new G4Element("Uranium", "U", 2);
    Uranium->AddIsotope(U235, 0.7204*perCent);
    Uranium->AddIsotope(U238, 100.0*perCent-0.7204*perCent);

    UMod = new G4Element("UraniumMix", "UMod", 2);
    UMod->AddIsotope(U235, abunU235);
    UMod->AddIsotope(U238, abunU238);

    //Make natural Hydrogen
    H = new G4Isotope("H", 1, 1, 1.008*g/mole);
    Hydrogen = new G4Element("Hydrogen", "H", 1);
    Hydrogen->AddIsotope(H, 100.*perCent);

    // Make heavy water isotopes and elements
    D2 = new G4Isotope("D2", 1, 2, 2.014*g/mole);
    Deuterium = new G4Element("Deuterium", "D", 1);
    Deuterium->AddIsotope(D2, 100.*perCent);

    O16 = new G4Isotope("O16", 8, 16, 15.995*g/mole);
    Oxygen = new G4Element("Oxygen", "O", 1);
    Oxygen->AddIsotope(O16, 100*perCent);

    // Make the heavy water material
    HeavyWater = new G4Material("Heavy Water", 1.1056*g/cm3, 2, kStateLiquid,
                                matTemp);
    HeavyWater->AddElement(Deuterium, 2);
    HeavyWater->AddElement(Oxygen, 1);

    LightWater = new G4Material("Light Water", 1.0*g/cm3, 2, kStateLiquid,
                                matTemp);
    LightWater->AddElement(Hydrogen, 2);
    LightWater->AddElement(Oxygen, 1);

    density = massFracW*(abunD20*1.1056*g/cm3 + abunH20*1.0*g/cm3) +
              massFracU*(0.7204*perCent*18.75*g/cm3 +
						 (99.2796*perCent)*18.9*g/cm3);

    UHW = new G4Material("UHW", density, 3, kStateLiquid, matTemp);
    UHW->AddMaterial(HeavyWater, abunD20);
    UHW->AddMaterial(LightWater, abunH20);
    UHW->AddElement(Uranium, massFracU);

    EU = new G4Material("EU", matDensity, 1, kStateSolid, matTemp);
    EU->AddElement(UMod, 100.*perCent);

    // Add materials to the map
    matMap["World"] = World;
    matMap["UHW"] = UHW;
    matMap["EU"] = EU;

    // Reset material change flag
    matChanged = false;

    return;
}


