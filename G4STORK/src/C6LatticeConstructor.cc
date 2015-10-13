/*
C6LatticeConstructor.cc

Created by:		Liam Russell
Date:			23-05-2012
Modified:       11-03-2013

Source code for the CANDU 6 lattice geometry and materials

*/

#include "C6LatticeConstructor.hh"


// Constructor
C6LatticeConstructor::C6LatticeConstructor()
: StorkVWorldConstructor(), cellLogical(0), cTubeLogical(0), gasAnnLogical(0),
  pressTubeLogical(0), coolantLogical(0), sheatheLogical(0), fuelLogical(0)
{
	// Set default member variables (from file or default values)
	latticePitch = 28.575*cm;
	fuelTemp = 859.99*kelvin;
	fuelDensity = 10.5541*g/cm3;
	coolantTemp = 561.285*kelvin;
	coolantDensity = 0.8074*g/cm3;
	moderatorTemp = 336.16*kelvin;
	moderatorDensity = 1.08875*g/cm3;

	// Set up variable property map
	variablePropMap[MatPropPair(fuel,temperature)] = &fuelTemp;
    variablePropMap[MatPropPair(fuel,density)] = &fuelDensity;
    variablePropMap[MatPropPair(coolant,temperature)] = &coolantTemp;
    variablePropMap[MatPropPair(coolant,density)] = &coolantDensity;
    variablePropMap[MatPropPair(moderator,temperature)] = &moderatorTemp;
    variablePropMap[MatPropPair(moderator,density)] = &moderatorDensity;
    variablePropMap[MatPropPair(all,dimension)] = &latticePitch;

    modVisAtt=NULL;
    fuelVisAtt=NULL;
    cTubeVisAtt=NULL;
    pressTubeVisAtt=NULL;
    gasAnnVisAtt=NULL;
    coolantVisAtt=NULL;
    sheatheVisAtt=NULL;
}


// Desturctor
C6LatticeConstructor::~C6LatticeConstructor()
{
	// Delete visualization attributes
	if(modVisAtt)
        delete modVisAtt;
    if(fuelVisAtt)
        delete fuelVisAtt;
    if(cTubeVisAtt)
        delete cTubeVisAtt;
    if(pressTubeVisAtt)
        delete pressTubeVisAtt;
    if(gasAnnVisAtt)
        delete gasAnnVisAtt;
    if(coolantVisAtt)
        delete coolantVisAtt;
    if(sheatheVisAtt)
        delete sheatheVisAtt;
}


// ConstructWorld()
// Construct the geometry and materials of the CANDU 6 lattice cell.
G4VPhysicalVolume* C6LatticeConstructor::ConstructWorld()
{
	// Lattic cell dimensions
	G4double buffer = 1.0*cm;
	reactorDim = G4ThreeVector(latticePitch/2.0,latticePitch/2.0,
											 2.4765*cm);
	encWorldDim = 2.0 * G4ThreeVector(reactorDim[0]+buffer,reactorDim[1]+buffer,
									  reactorDim[2]+buffer);
	G4SolidStore* theSolids = G4SolidStore::GetInstance();

	// Set static dimensions

    // Calandria Tube dimensions
    G4double cTubeRadmax = 6.5875*cm;
    G4double cTubeLen = reactorDim[2];

    // Create the Gas Annulus
    G4double gasAnnRadmax = 6.4478*cm;
    G4double gasAnnLen = reactorDim[2];

    // Create the Pressure Tube
    G4double pressTubeRadmax = 5.6032*cm;
    G4double pressTubeLen = reactorDim[2];

    // Create the Coolant
    G4double coolantRadmax = 5.1689*cm;
    G4double coolantLen = reactorDim[2];

    // Create the Sheathe
    G4double sheatheRadmax = 0.6540*cm;
//    G4double sheatheLen = reactorDim[2]-0.5*cm;
    G4double sheatheLen = reactorDim[2];

    // Fuel pin dimensions
    G4double pinRad = 0.6122*cm;
//    G4double pinLen = sheatheLen-1.*cm;
    G4double pinLen = sheatheLen;
    G4int rings = 4;
    G4double ringRad[3] = {1.4885*cm,2.8755*cm,4.3305*cm};
    G4double secondRingOffset = 0.261799*radian;


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
		new G4Box("worldBox", reactorDim[0]+buffer, reactorDim[1]+buffer,
				  reactorDim[2]+buffer);
		// Create the lattice cell solid
		new G4Box("cellBox", reactorDim[0], reactorDim[1], reactorDim[2]);
		// Create calandria tube
		new G4Tubs("calTube", 0., cTubeRadmax, cTubeLen, 0., 2.0*CLHEP::pi);
		// Create gas annulus
		new G4Tubs("gasAnnTube", 0., gasAnnRadmax, gasAnnLen, 0.,
					2.0*CLHEP::pi);
		// Create pressure tube
		new G4Tubs("pressTube", 0., pressTubeRadmax, pressTubeLen, 0.,
					2.0*CLHEP::pi);
		// Create coolant solid
		new G4Tubs("coolantTube", 0., coolantRadmax, coolantLen, 0.,
					2.0*CLHEP::pi);
		// Create the sheathe for fuel pins
		new G4Tubs("sheatheTube", 0., sheatheRadmax, sheatheLen, 0.,
					2.0*CLHEP::pi);
		// Create a fuel pins
		new G4Tubs("pinCyl", 0., pinRad, pinLen, 0.,2.0*CLHEP::pi);

		geomChanged = false;
	}

    // Create world volume
    worldLogical = new G4LogicalVolume(theSolids->GetSolid("worldBox"),
									   matMap["Galactic"],"worldLogical",0,0,0);
    worldPhysical = new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), worldLogical,
                                      "worldPhysical",0,0,0);

    // Create the lattice cell (moderator) volume
    cellLogical = new G4LogicalVolume(theSolids->GetSolid("cellBox"),
									  matMap["Moderator"],"cellLogical",0,0,0);
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), cellLogical,"cellPhysical",
                      worldLogical,0,0);

    // Create the Calandria Tube
    cTubeLogical = new G4LogicalVolume(theSolids->GetSolid("calTube"),
									   matMap["CalandriaTube"], "cTubeLogical",
									   0,0,0);
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), cTubeLogical,"cTubePhysical",
					  cellLogical,0,0);

    // Create the Gas Annulus
    gasAnnLogical = new G4LogicalVolume(theSolids->GetSolid("gasAnnTube"),
										matMap["AnnulusGas"], "gasAnnLogical",
										0,0,0);
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), gasAnnLogical,
					  "gasAnnPhysical",cTubeLogical,0,0);

    // Create the Pressure Tube
    pressTubeLogical = new G4LogicalVolume(theSolids->GetSolid("pressTube"),
										   matMap["PressureTube"],
										   "pressTubeLogical",0,0,0);
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), pressTubeLogical,
					  "pressTubePhysical",gasAnnLogical,0,0);

    // Create the Coolant
    coolantLogical = new G4LogicalVolume(theSolids->GetSolid("coolantTube"),
										 matMap["Coolant"], "coolantLogical",
										 0,0,0);
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), coolantLogical,
					  "coolantPhysical",pressTubeLogical,0,0);

    // Create the Sheathe
    sheatheLogical = new G4LogicalVolume(theSolids->GetSolid("sheatheTube"),
										 matMap["Sheathe"], "sheatheLogical",
										 0,0,0);

    // Create a fuel
    fuelLogical = new G4LogicalVolume(theSolids->GetSolid("pinCyl"),
									  matMap["Fuel"], "fuelLogical",0,0,0);


    // Create fuel bundle

    // Rotation and translation of the rod and sheathe
	std::stringstream volName;

    // Set name for sheathe physical volume
    volName << 0;

    // Place centre pin
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), sheatheLogical,
					  "sheathePhysical " + volName.str(),coolantLogical,0,0);
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), fuelLogical,"fuelPhysical ",
					  sheatheLogical,0,0);


    // Place pins for each ring
    for( G4int j = 1; j < rings; j++ )
    {
        for( G4int i = 0; i < j*6; i++ )
        {
            // Reset string stream
            volName.str("");

            volName << j << "-" << i;

            if(j == 2)
            {

				 G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(i)/
												G4double(j*6)+secondRingOffset),
								  ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(i)/
												G4double(j*6)+secondRingOffset),
								  0.);
				 new G4PVPlacement(0, Tm, sheatheLogical,"sheathePhysical " +
									volName.str(),coolantLogical,0,0);
            }
            else
            {
				 G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(i)/
													G4double(j*6)),
								  ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(i)/
														G4double(j*6)), 0.);
				new G4PVPlacement(0, Tm, sheatheLogical,"sheathePhysical " +
									volName.str(),coolantLogical,0,0);
            }

        }
    }


	// Add sensitive detector to ALL logical volumes
	worldLogical->SetSensitiveDetector( sDReactor );
	cellLogical->SetSensitiveDetector( sDReactor );
	cTubeLogical->SetSensitiveDetector( sDReactor );
	gasAnnLogical->SetSensitiveDetector( sDReactor );
	pressTubeLogical->SetSensitiveDetector( sDReactor );
	coolantLogical->SetSensitiveDetector( sDReactor );
	sheatheLogical->SetSensitiveDetector( sDReactor );
	fuelLogical->SetSensitiveDetector( sDReactor );


    // Set visualization attributes

    if(worldVisAtt)
        delete worldVisAtt;
    if(modVisAtt)
        delete modVisAtt;
    if(fuelVisAtt)
        delete fuelVisAtt;
    if(cTubeVisAtt)
        delete cTubeVisAtt;
    if(pressTubeVisAtt)
        delete pressTubeVisAtt;
    if(gasAnnVisAtt)
        delete gasAnnVisAtt;
    if(coolantVisAtt)
        delete coolantVisAtt;
    if(sheatheVisAtt)
        delete sheatheVisAtt;

    worldVisAtt = new G4VisAttributes(G4Colour(1.,1.,1.));
    worldVisAtt->SetVisibility(false);
    worldLogical->SetVisAttributes(worldVisAtt);

    modVisAtt = new G4VisAttributes(G4Colour(0.53,0.81,0.92));
    cellLogical->SetVisAttributes(modVisAtt);

    cTubeVisAtt = new G4VisAttributes(G4Colour(0,0,1));
    cTubeLogical->SetVisAttributes(cTubeVisAtt);

    gasAnnVisAtt = new G4VisAttributes(G4Colour(0.5,0.5,0.5));
    gasAnnLogical->SetVisAttributes(gasAnnVisAtt);

    pressTubeVisAtt = new G4VisAttributes(G4Colour(0,1,0));
    pressTubeLogical->SetVisAttributes(pressTubeVisAtt);

    coolantVisAtt = new G4VisAttributes(G4Colour(0,0.5,0.92));
    coolantLogical->SetVisAttributes(coolantVisAtt);

    sheatheVisAtt = new G4VisAttributes(G4Colour(1,0.55,0));
    sheatheLogical->SetVisAttributes(sheatheVisAtt);

    fuelVisAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    fuelLogical->SetVisAttributes(fuelVisAtt);


    return worldPhysical;
}


// ConstructMaterials()
// Define and build the materials in the C6 lattice cell.
void C6LatticeConstructor::ConstructMaterials()
{
    // Elements, isotopes and materials
    G4Isotope *U234, *U235, *U238, *O16, /* *O17, *O18,*/ *Fe54, *Fe56, *Fe57,
              *Fe58, *B10, *Gd155, *Gd157, *Nb93, *Cr50, *Cr52, *Cr53, *Cr54,
              *Ni58, *Ni60, *Ni64, *Zr90, *Zr91, *Zr92, *Zr94, *Zr96,
              *C12, *C13;
    G4Element *NU, *Oxygen, *H1, *D2, *Fe, *Gd, *Cr, *Ni, *Zr, *C, *Boron,
			  *Niobium;
    G4Material *World, *Fuel, *H2O, *D2O, *Moderator, *Coolant, *PressureTube,
			   *AnnulusGas, *CalandriaTube, *Sheathe;


    // Make the uranium isotopes and element
    U234 = new G4Isotope("U234", 92, 234, 234.0410*g/mole);
    U235 = new G4Isotope("U235", 92, 235, 235.0439*g/mole);
    U238 = new G4Isotope("U238", 92, 238, 238.0508*g/mole);

    NU = new G4Element("Natural Uranium", "NU", 3);
    NU->AddIsotope(U234,  0.0055*perCent);
    NU->AddIsotope(U235,  0.7109*perCent);
    NU->AddIsotope(U238, 99.2836*perCent);

    // Make hydrogen elements
    H1 = new G4Element("Hydrogen", "H", 1);
    H1->AddIsotope(new G4Isotope("H1", 1, 1, 1.008*g/mole), 1.0);

    D2 = new G4Element("Hydrogen", "H", 1);
    D2->AddIsotope(new G4Isotope("H2", 1, 2, 2.014*g/mole), 1.0);

    // Make oxygen isotope and element
    O16 = new G4Isotope("O16", 8, 16, 15.995*g/mole);
//    O17 = new G4Isotope("O17", 8, 17, 16.999*g/mole);
//    O18 = new G4Isotope("O18", 8, 18, 17.999*g/mole);

    Oxygen = new G4Element("Oxygen", "O", 1);
    Oxygen->AddIsotope(O16, 100.*perCent);
//    Oxygen->AddIsotope(O16, 99.757*perCent);
//    Oxygen->AddIsotope(O17, 0.038*perCent);
//    Oxygen->AddIsotope(O18, 0.205*perCent);

    //make iron isotopes and element
    Fe54 = new G4Isotope("Fe54", 26, 54, 53.9396105*g/mole);
    Fe56 = new G4Isotope("Fe56", 26, 56, 55.9349375*g/mole);
    Fe57 = new G4Isotope("Fe57", 26, 57, 56.9353940*g/mole);
    Fe58 = new G4Isotope("Fe58", 26, 58, 57.9332756*g/mole);

    Fe = new G4Element("Iron", "Fe", 4);
    Fe->AddIsotope(Fe54,  5.80*perCent);
    Fe->AddIsotope(Fe56, 91.72*perCent);
    Fe->AddIsotope(Fe57,  2.20*perCent);
    Fe->AddIsotope(Fe58,  0.28*perCent);

    //make boron isotopes and element
    B10 = new G4Isotope("B10", 5, 10, 10.0129370*g/mole);

    Boron = new G4Element("Boron", "B", 1);
    Boron->AddIsotope(B10, 100*perCent);

    //make gadolinium isotopes and element
    Gd155 = new G4Isotope("Gd155", 64, 155, 154.9226220*g/mole);
    Gd157 = new G4Isotope("Gd157", 64, 157, 156.9239601*g/mole);


    Gd = new G4Element("Gadolinium", "Gd", 2);
    Gd->AddIsotope(Gd155, 48.28373787*perCent);
    Gd->AddIsotope(Gd157, 51.71626213*perCent);

    //make niobium isotopes and element
    Nb93 = new G4Isotope("Nb93", 41, 93, 92.9063781*g/mole);

    Niobium = new G4Element("Niobium", "Nb", 1);
    Niobium->AddIsotope(Nb93, 100*perCent);

    //make chromium isotopes and element
    Cr50 = new G4Isotope("Cr50", 24, 50, 49.9460422*g/mole);
    Cr52 = new G4Isotope("Cr52", 24, 52, 51.9405075*g/mole);
    Cr53 = new G4Isotope("Cr53", 24, 53, 52.9406494*g/mole);
    Cr54 = new G4Isotope("Cr54", 24, 54, 53.9388804*g/mole);

    Cr = new G4Element("Chromium", "Cr", 4);
    Cr->AddIsotope(Cr50,  4.1737*perCent);
    Cr->AddIsotope(Cr52, 83.7003*perCent);
    Cr->AddIsotope(Cr53,  9.6726*perCent);
    Cr->AddIsotope(Cr54,  2.4534*perCent);

    //make nickel isotopes and element
    Ni58 = new G4Isotope("Ni58", 28, 58, 57.9353429*g/mole);
    Ni60 = new G4Isotope("Ni60", 28, 60, 59.9307864*g/mole);
    Ni64 = new G4Isotope("Ni64", 28, 64, 63.9279660*g/mole);

    Ni = new G4Element("Nickel", "Ni", 3);
    Ni->AddIsotope(Ni58, 70.913*perCent);
    Ni->AddIsotope(Ni60, 28.044*perCent);
    Ni->AddIsotope(Ni64,  1.043*perCent);

    //make Zirconium isotopes and element
    Zr90 = new G4Isotope("Zr90", 40, 90, 89.9047044*g/mole);
    Zr91 = new G4Isotope("Zr91", 40, 91, 90.9056458*g/mole);
    Zr92 = new G4Isotope("Zr92", 40, 92, 91.9050408*g/mole);
    Zr94 = new G4Isotope("Zr94", 40, 94, 93.9063152*g/mole);
    Zr96 = new G4Isotope("Zr96", 40, 96, 95.9082734*g/mole);

    Zr = new G4Element("Zirconium", "Zr", 5);
    Zr->AddIsotope(Zr90, 50.706645*perCent);
    Zr->AddIsotope(Zr91, 11.180922*perCent);
    Zr->AddIsotope(Zr92, 17.277879*perCent);
    Zr->AddIsotope(Zr94, 17.890875*perCent);
    Zr->AddIsotope(Zr96,  2.943679*perCent);

    //make natural carbon element
//    C = new G4Element("Carbon", "C", 6, 12.011*g/mole);

//    G4NistManager *nistMan = G4NistManager::Instance();
//	C = nistMan->FindOrBuildElement("C");

    C12 = new G4Isotope("C12", 6, 12, 12.0*g/mole);
    C13 = new G4Isotope("C13", 6, 13, 13.00335*g/mole);

    C = new G4Element("Carbon", "C", 2);
    C->AddIsotope(C12, 98.83*perCent);
    C->AddIsotope(C13,  1.07*perCent);


    // Create the world material
    World = new G4Material("Galactic", 1, 1, 1.e-25*g/cm3, kStateGas,
						   2.73*kelvin, 3.e-18*pascal);

    // Create H20 material
    H2O = new G4Material("Light Water", 1.*g/cm3, 2, kStateLiquid);
    H2O->AddElement(H1,2);
    H2O->AddElement(Oxygen,1);

    // Create D20 material
    D2O = new G4Material("Heavy Water", 1.1*g/cm3, 2, kStateLiquid);
    D2O->AddElement(D2,2);
    D2O->AddElement(Oxygen,1);


    // Create Coolant
	Coolant = new G4Material("Coolant", coolantDensity, 2, kStateLiquid,
							 coolantTemp);
    Coolant->AddMaterial(D2O, 99.3777*perCent);
    Coolant->AddMaterial(H2O,  0.6223*perCent);

    //Create Pressure Tube
    PressureTube = new G4Material("PressureTube", 6.5041*g/cm3, 6, kStateSolid,
								  561.285*kelvin);
    PressureTube->AddElement(Niobium,2.5800*perCent);
    PressureTube->AddElement(Fe,0.04678*perCent);
    PressureTube->AddElement(Cr,0.008088*perCent);
    PressureTube->AddElement(Ni,0.0035*perCent);
    PressureTube->AddElement(Boron,0.00002431*perCent);
    PressureTube->AddElement(Zr,97.313*perCent);

    //Create Annulus Gas
    AnnulusGas = new G4Material("AnnulusGas", 0.0012*g/cm3, 2, kStateGas,
								448.72*kelvin);
    AnnulusGas->AddElement(C,27.11*perCent);
    AnnulusGas->AddElement(Oxygen,72.89*perCent);

    //Create Calandra Tube
    CalandriaTube = new G4Material("CalandriaTube", 6.4003*g/cm3, 5,
								   kStateSolid, 336.16*kelvin);
    CalandriaTube->AddElement(Fe,0.1370624917*perCent);
    CalandriaTube->AddElement(Ni,0.05583986327*perCent);
    CalandriaTube->AddElement(Cr,0.1015166605*perCent);
    CalandriaTube->AddElement(Zr,99.7055204*perCent);
    CalandriaTube->AddElement(Boron,0.00006058385298*perCent);

    //Create Moderator
    Moderator = new G4Material("Moderator", moderatorDensity, 2, kStateLiquid,
							   moderatorTemp);
    Moderator->AddMaterial(D2O,99.95895058*perCent);
    Moderator->AddMaterial(H2O,0.04104941778*perCent);

    //Create Fuel
    Fuel = new G4Material("Fuel", fuelDensity, 2, kStateSolid, fuelTemp);
    Fuel->AddElement(Oxygen,11.8502*perCent);
    Fuel->AddElement(NU,88.1498*perCent);

    //Create Sheathe
    Sheathe = new G4Material("Sheathe", 6.3918*g/cm3, 5, kStateSolid,
							 561.285*kelvin);
    Sheathe->AddElement(Zr,99.6781689*perCent);
    Sheathe->AddElement(Fe,0.213200015*perCent);
    Sheathe->AddElement(Cr,0.1015238119*perCent);
    Sheathe->AddElement(Ni,0.007106666832*perCent);
    Sheathe->AddElement(Boron,0.00006052849665*perCent);

    // Add materials to the map indexed by either ZA (format ZZAAA or ZZ)
    // For composite materials:  world is 0, heavy water is 1, UHW is 2
    matMap["Galactic"] = World;
    matMap["H2O"] = H2O;
    matMap["D2O"] = D2O;
    matMap["Moderator"] = Moderator;
    matMap["Fuel"] = Fuel;
    matMap["Sheathe"] = Sheathe;
    matMap["CalandriaTube"] = CalandriaTube;
    matMap["AnnulusGas"] = AnnulusGas;
    matMap["PressureTube"] = PressureTube;
    matMap["Coolant"] = Coolant;

    matChanged = false;

    return;
}
