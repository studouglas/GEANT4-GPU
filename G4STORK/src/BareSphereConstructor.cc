/*
BareSphereConstructor.cc

Created by:		Liam Russell
Date:			23-05-2012
Modified:       10-03-2013

Source code for the bare sphere geometry and materials.

*/

#include "BareSphereConstructor.hh"


// Constructor
BareSphereConstructor::BareSphereConstructor()
: StorkVWorldConstructor(), reactorLogical(0)
{
	// Set default values for member variables
    matTemp = 293.6*kelvin;

    // Set default material densities
    matDensity[e_UHW] = -1.0*g/cm3;
    matDensity[e_Godiva] = 18.7398*g/cm3;
    matDensity[e_U235] = 18.75*g/cm3;

    reactorVisAtt=NULL;
}


// Destructor
BareSphereConstructor::~BareSphereConstructor()
{
    // Delete visualization attributes
    if(reactorVisAtt)
        delete reactorVisAtt;
}


// ConstructNewWorld()
// Build bare sphere world for the first time.  Set default values and user
// inputs.  Also set up the variable property map.
G4VPhysicalVolume*
BareSphereConstructor::ConstructNewWorld(const StorkParseInput* infile)
{
    G4int matIndex = 2;

	// Select material based on input file
    switch(infile->GetReactorMaterial())
    {
    	case 2:
			materialID = "UHW";
			matIndex = e_UHW;
			reactorRadius = 87.5*cm;
			break;
		case 3:
			materialID = "Godiva";
			matIndex = e_Godiva;
			reactorRadius = 8.7*cm;
			break;
		default:	// 92235
			materialID = "U235Mat";
			matIndex = e_U235;
			reactorRadius = 8.7*cm;
			break;
    }

    // Set up variable property map
    variablePropMap[MatPropPair(all,temperature)] = &matTemp;
    variablePropMap[MatPropPair(all,density)] = &(matDensity[matIndex]);
    variablePropMap[MatPropPair(all,dimension)] =&reactorRadius;


    // Call base class ConstructNewWorld() to complete construction
    return StorkVWorldConstructor::ConstructNewWorld(infile);
}


// ConstructWorld
// Construct the geometry and materials of the spheres given the inputs.
G4VPhysicalVolume* BareSphereConstructor::ConstructWorld()
{
	// Set local variables and enclosed world dimensions
	reactorDim = G4ThreeVector(reactorRadius, 0, 0);
    G4double buffer = 1.0*cm;
    encWorldDim = 2.0 * G4ThreeVector(reactorRadius+buffer,
									  reactorRadius+buffer,
									  reactorRadius+buffer);
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

		// Create world solid
		new G4Orb("worldSphere", reactorRadius+buffer);

		// Create the reactor solid
		new G4Orb("reactorSphere", reactorRadius);

		geomChanged = false;
	}


    // Create world volume
    worldLogical = new G4LogicalVolume(theSolids->GetSolid("worldSphere"),
									   matMap["World"], "worldLogical",0,0,0);
    worldPhysical = new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), worldLogical,
                                      "worldPhysical",0,0,0);

    // Create the homogenous spherical reactor
    reactorLogical = new G4LogicalVolume(theSolids->GetSolid("reactorSphere"),
										 matMap[materialID], "reactorLogical",
										 0,0,0);
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), reactorLogical,
                      "reactorPhysical", worldLogical,0,0);


    // Set reactor as sensitive detector
	reactorLogical->SetSensitiveDetector(sDReactor);


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
// Construct all the materials needed for the BareSphereConstructor.
void BareSphereConstructor::ConstructMaterials()
{
	G4double abunD20 = 100.0*perCent;
    G4double abunU235 = 0.7204*perCent;
    G4double abunU238 = 100.0*perCent - abunU235;
    G4double massFracU = 10.73*perCent;
    G4double massFracW = 100.0*perCent - massFracU;

    // Elements, isotopes and materials
    G4Isotope *U234, *U235, *U238, *D2, *O16;
    StorkElement *Uranium, *Oxygen, *Deuterium, *PureU235, *EU;
    StorkMaterial *World, *UHW, *U235Mat, *HeavyWater, *Godiva;

    // Create the world environment
    World = new StorkMaterial("Galactic", 1, 1, 1.e-25*g/cm3, kStateGas,
                                      2.73*kelvin, 3.e-18*pascal);

    // Make the uranium isotopes and element
    U234 = new G4Isotope("U234", 92, 234, 234.0410*g/mole);
    U235 = new G4Isotope("U235", 92, 235, 235.0439*g/mole);
    U238 = new G4Isotope("U238", 92, 238, 238.0508*g/mole);

    Uranium = new StorkElement("Uranium", "U", 2);
    Uranium->AddIsotope(U235, abunU235);
    Uranium->AddIsotope(U238, abunU238);

    // Make heavy water isotopes and elements
    D2 = new G4Isotope("D2", 1, 2, 2.014*g/mole);
    Deuterium = new StorkElement("Deuterium", "D", 1);
    Deuterium->AddIsotope(D2, 100*perCent);

    O16 = new G4Isotope("O16", 8, 16, 15.995*g/mole);
    Oxygen = new StorkElement("Oxygen", "O", 1);
    Oxygen->AddIsotope(O16, 100*perCent);

    // Make the U235 material
    U235Mat = new StorkMaterial("U235 Material", matDensity[e_U235], 1,
							 kStateSolid, matTemp);
    PureU235 = new StorkElement("Uranium-235","U235",1);
    PureU235->AddIsotope(U235, 100*perCent);
    U235Mat->AddElement(PureU235,1);

    // Make Godiva material
    Godiva = new StorkMaterial("Godiva", matDensity[e_Godiva], 1, kStateSolid,
							matTemp);
    EU = new StorkElement("Enriched Uranium","EU",3);
    EU->AddIsotope(U234, 1.0252*perCent);
    EU->AddIsotope(U235, 93.7695*perCent);
    EU->AddIsotope(U238, 5.2053*perCent);
    Godiva->AddElement(EU,1);

    // Make the heavy water material
    HeavyWater = new StorkMaterial("Heavy Water", 1.1056*g/cm3, 2, kStateLiquid,
                                matTemp);
    HeavyWater->AddElement(Deuterium, 2);
    HeavyWater->AddElement(Oxygen, 1);

    // Make the UHW material
    if(matDensity[e_UHW] < 0)
    {
    	matDensity[e_UHW] = (abunD20 * 1.1056*g/cm3) * massFracW +
				(abunU235 * 18.75*g/cm3 + abunU238 * 18.9*g/cm3) * massFracU;
    }

    UHW = new StorkMaterial("UHW", matDensity[e_UHW], 2, kStateSolid, matTemp);
    UHW->AddMaterial(HeavyWater, massFracW);
    UHW->AddElement(Uranium, massFracU);

    // Add materials to the map indexed by either ZA (format ZZAAA or ZZ)
    // For composite materials:  world is 0, heavy water is 1, UHW is 2
    matMap["World"] = World;
    matMap["UHW"] = UHW;
    matMap["Godiva"] = Godiva;
    matMap["U235Mat"] = U235Mat;

	// Reset material changed flag
	matChanged = false;

    return;
}
