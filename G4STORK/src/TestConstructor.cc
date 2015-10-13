#include "TestConstructor.hh"

#include "TestConstructor.hh"

TestConstructor::TestConstructor()
: StorkVWorldConstructor(), cellLogical(0), zircGridLogical(0)
{
    cellVisAtt=NULL;
    zircGridVisAtt=NULL;
}

TestConstructor::~TestConstructor()
{
	// Delete visualization attributes
	if(cellVisAtt)
        delete cellVisAtt;
    if(zircGridVisAtt)
        delete zircGridVisAtt;


}

// ConstructWorld()
// Construct the geometry and materials of the SLOWPOKE Reactor.
G4VPhysicalVolume* TestConstructor::ConstructWorld()
{


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

        // Set static dimensions
        //Note the format for cylinder dimensions is (inner radius, outer radius, height)

        // Reactor pool dimensions
        G4double buffer = 1.0*cm;
        // reactor dimension specifies the dimension of the volume for the uniform distribution
        reactorDim = G4ThreeVector(0., 11.049*cm, 22.748*cm);

        G4ThreeVector cellDim = G4ThreeVector(0., 133.0*cm, 564.0*cm);

        // World dimensions
        encWorldDim = G4ThreeVector(cellDim[2]+buffer,cellDim[2]+buffer, cellDim[2]+buffer);

        // Lattice cell dimensions
        // hole pattern format radius, angle of increment and the offset angle, check radius
        G4double sheatheDim[3] = {0., 0.262*cm , 1*cm};
        G4double gridPlateDim[3] = {1.331*cm, 11.049*cm, 0.279*cm};
        //G4double unitRegionDim[6]={0, (gridPlateDim[1]-gridPlateDim[0])/10, 0., CLHEP::pi/(3*10), 0., 0.3*cm};
        //G4double regionDim[6]={gridPlateDim[0], gridPlateDim[1], CLHEP::pi/3, 2*CLHEP::pi/3, -gridPlateDim[2]/2, gridPlateDim[2]/2};


		//Geometry positioning data
		std::stringstream latticeType1;
		G4ThreeVector latCellPos;
		G4ThreeVector holePos;
		G4ThreeVector holeRPos;

		// Create world solid
		new G4Box("worldBox", encWorldDim[0]/2, encWorldDim[1]/2, encWorldDim[2]/2);

		// Create cell solid
		new G4Tubs("cellTube", 0., cellDim[1], cellDim[2]/2, 0., 2.0*CLHEP::pi);

        //creates the base zirconium grid slice that the holes will be subtracted from to form the upper and lower grid plate
        new G4Tubs("gridPlate",  gridPlateDim[0], gridPlateDim[1], gridPlateDim[2]/2, CLHEP::pi/3, CLHEP::pi/3);

		new G4Tubs("sheatheTube", sheatheDim[0], sheatheDim[1], sheatheDim[2]/2, 0., 2.0*CLHEP::pi);

		//test = new G4UnionSolid("TestSolid1",theSolids->GetSolid("sheatheTube"),theSolids->GetSolid("sheatheTube"), 0, G4ThreeVector(1.0*cm,1.0*cm,0.));

		test = new StorkUnionSolid("TestSolid1",theSolids->GetSolid("sheatheTube"),theSolids->GetSolid("sheatheTube"), 0, G4ThreeVector(1.0*cm,1.0*cm,0.)
                    ,cylUnit, StorkSixVector<G4double>(0.,4.*cm,0.,2*CLHEP::pi,-2.0*cm, 2.0*cm),G4ThreeVector(0.*cm,2.0*cm,0.));

       // test = new StorkUnionSolid("TestSolid2",theSolids->GetSolid("gridPlate"),theSolids->GetSolid("TestSolid1"), 0, G4ThreeVector(0.*cm,2.0*cm,0.)
       //             ,cylUnit, StorkSixVector<G4double>(0.,4.*cm,0.,1*CLHEP::pi,-2.0*cm, 2.0*cm),G4ThreeVector(0.,0.,0.));

////		new StorkUnionSolid("TestSolid",theSolids->GetSolid("sheatheTube"),theSolids->GetSolid("sheatheTube"), 0, G4ThreeVector(1.0*cm,1.0*cm,0.)
////                    ,cylUnit, StorkSixVector<G4double>(0.,4.*cm,0.,0.5*CLHEP::pi,-2.0*cm, 2.0*cm),G4ThreeVector(0.,0.,0.));
////
////        new StorkUnionSolid("TestSolid2",theSolids->GetSolid("TestSolid"),theSolids->GetSolid("sheatheTube"), 0, G4ThreeVector(-1.0*cm,1.0*cm,0.)
////                    ,cylUnit, StorkSixVector<G4double>(0.,4.*cm,0.,CLHEP::pi,-2.0*cm, 2.0*cm),G4ThreeVector(0.,0.,0.));
////
////        new StorkUnionSolid("TestSolid3",theSolids->GetSolid("TestSolid2"),theSolids->GetSolid("sheatheTube"), 0, G4ThreeVector(-1.0*cm, -1.0*cm,0.)
////                    ,cylUnit, StorkSixVector<G4double>(0.,4.*cm,0.,1.5*CLHEP::pi,-2.0*cm, 2.0*cm),G4ThreeVector(0.,0.,0.));
////
////		test = new StorkUnionSolid("TestSolid4",theSolids->GetSolid("TestSolid3"),theSolids->GetSolid("sheatheTube"), 0, G4ThreeVector(1.0*cm,-1.0*cm,0.)
////                    ,cylUnit, StorkSixVector<G4double>(0.,4.*cm,0.,2*CLHEP::pi,-2.0*cm, 2.0*cm),G4ThreeVector(0.,0.,0.));
//
////        solidList sheatheTubes;
//
//		G4int regionIndices[3];
//        regionIndices[0] = ceil((regionDim[1]-regionDim[0])/(unitRegionDim[1]-unitRegionDim[0]));
//        regionIndices[1] = ceil((regionDim[3]-regionDim[2])/(unitRegionDim[3]-unitRegionDim[2]));
//        regionIndices[2] = ceil((regionDim[5]-regionDim[4])/(unitRegionDim[5]-unitRegionDim[4]));
//
//        unitRegionDim[0] = regionDim[1]-(regionDim[1]-regionDim[0])/(regionIndices[0]);
//        unitRegionDim[1] = regionDim[1];
//        unitRegionDim[2] = regionDim[3]-(regionDim[3]-regionDim[2])/(regionIndices[1]);
//        unitRegionDim[3] = regionDim[3];
//        unitRegionDim[4] = regionDim[5]-(regionDim[5]-regionDim[4])/(regionIndices[2]);
//        unitRegionDim[5] = regionDim[5];
//
//        G4double unitRegDim[6];
//        intVec elemsRow(regionIndices[0], 0);
//        G4int count=0;
//
//        for(G4int i=0; i<regionIndices[0]; i++)
//        {
//            unitRegDim[0]=regionDim[1]-(unitRegionDim[1]-unitRegionDim[0])*(i+1);
//            unitRegDim[1]=regionDim[1]-(unitRegionDim[1]-unitRegionDim[0])*(i);
//            elemsRow[i]=ceil((unitRegDim[0]*(regionDim[3]-regionDim[2]))/(unitRegionDim[0]*(unitRegionDim[3]-unitRegionDim[2])));
//
//            for(G4int j=0; j<(elemsRow[i]); j++)
//            {
//                unitRegDim[2]=regionDim[3]-(j+1)*(regionDim[3]-regionDim[2])/(elemsRow[i]);
//                unitRegDim[3]=regionDim[3]-(j)*(regionDim[3]-regionDim[2])/(elemsRow[i]);
//
//                for(G4int k=0; k<regionIndices[2]; k++)
//                {
//                    unitRegDim[4]=regionDim[5]-(k+1)*(regionDim[5]-regionDim[4])/(regionIndices[2]);
//                    unitRegDim[5]=regionDim[5]-(k)*(regionDim[5]-regionDim[4])/(regionIndices[2]);
//                    StorkSixVector<G4double> unitRegDimTemp(unitRegDim);
//                    holeRPos={(unitRegDimTemp[1]+unitRegDimTemp[0])/2,(unitRegDimTemp[3]+unitRegDimTemp[2])/2,(unitRegDimTemp[5]+unitRegDimTemp[4])/2};
//                    holePos.setRhoPhiZ(holeRPos[0],holeRPos[1],holeRPos[2]);
//                    latticeType1.str("");
//                    latticeType1 << count;
//                    count++;
//                    test = new G4UnionSolid("TestSolid"+latticeType1.str(),test, theSolids->GetSolid("sheatheTube"), 0, holePos);
////                                                ,cylUnit, regionDim, G4ThreeVector(0.,0.,0.));
////                    sheatheTubes.push_back(solidPos(theSolids->GetSolid("sheatheTube"),holePos));
//
//                }
//            }
//        }
//
////		UnionBinaryTree* sheatheTubeLat = new UnionBinaryTree(&sheatheTubes);
////
////		sheatheTubeLatPair = sheatheTubeLat->GetUnionSolid("sheatheTubeLat", 0, cylUnit, unitRegionDim, regionDim, 0.0, radCyl, 1.0, NULL, true);
//
//		// creates the upGridPlate and the lowGridPlate from the unions of the upGridHolesLat and lowGridHolesLat with the base gridPlate
////		new G4SubtractionSolid("upGridPlate", theSolids->GetSolid("gridPlate"), upGridHolesLatPair.first, 0, upGridHolesLatPair.second);
////		new G4SubtractionSolid("lowGridPlate", theSolids->GetSolid("gridPlate"), lowGridHolesLatPair.first, 0, lowGridHolesLatPair.second);
////
////        // creates the zirconium grid slice (zircGridPlate) from the union of the upGridPlate and the lowGridPlate
////		new G4UnionSolid("zircGridSliceP1", theSolids->GetSolid("upGridPlate"), theSolids->GetSolid("lowGridPlate"), 0, disUpGridToLowGrid);
////		gridSlice = new G4UnionSolid("zircGridSlice", theSolids->GetSolid("gridPlate"), sheatheTubeLatPair.first, 0, sheatheTubeLatPair.second);

		geomChanged = false;
		std::vector<G4VSolid*> *Check = dynamic_cast<std::vector<G4VSolid*>*>(theSolids);
		for(G4int i=0; i<int(Check->size()); i++)
		{
            G4cout << "\n ###" << ((*Check)[i])->GetName() << " " << ((*Check)[i])->GetEntityType() <<" ### \n";
		}
		latticeType1.str("");
	}

    //Initialize stringstream for volume naming purposes
    std::stringstream volName;
    // Initialize positioning objects
    G4ThreeVector volPos;

    // Create world volume
    worldLogical = new G4LogicalVolume(theSolids->GetSolid("worldBox"),
									   matMap["Galactic"],"worldLogical");
    worldPhysical = new G4PVPlacement(0, G4ThreeVector(0,0,0.), worldLogical,
                                      "worldPhysical",0,0,0);

    // Create the lattice cell (moderator) volume
    cellLogical = new G4LogicalVolume(theSolids->GetSolid("cellTube"),
									  matMap["H2O"],"cellLogical");
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), cellLogical,"cellPhysical",
                      worldLogical,0,0);

    //    //create Zirconium grid plates
    zircGridLogical = new G4LogicalVolume(theSolids->GetSolid("TestSolid1"), matMap["Zirconium"], "zircGridLogical");

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), zircGridLogical,"zircGridSlicePhysical",
                      cellLogical,0,0,0);

	// Add sensitive detector to ALL logical volumes
	worldLogical->SetSensitiveDetector( sDReactor );
	cellLogical->SetSensitiveDetector( sDReactor );

	zircGridLogical->SetSensitiveDetector( sDReactor );

    // Set visualization attributes

    if(worldVisAtt)
        delete worldVisAtt;
    if(cellVisAtt)
        delete cellVisAtt;
    if(zircGridVisAtt)
        delete zircGridVisAtt;

    worldVisAtt = new G4VisAttributes(G4Colour(1.,1.,1.));
    worldVisAtt->SetVisibility(false);
    worldLogical->SetVisAttributes(worldVisAtt);

// light blue
    cellVisAtt = new G4VisAttributes(G4Colour(47.0/255.0,225.0/255.0,240.0/255.0));
    cellVisAtt->SetVisibility(false);
    cellLogical->SetVisAttributes(cellVisAtt);



    zircGridVisAtt = new G4VisAttributes(G4Colour(0.,0.,1.));
    zircGridVisAtt->SetVisibility(true);
    zircGridLogical->SetVisAttributes(zircGridVisAtt);

    return worldPhysical;
}


// ConstructMaterials()
// Define and build the materials in the C6 lattice cell.
void TestConstructor::ConstructMaterials()
{
// Density Of Defined Materials
    G4double ReflectorDensity = 1.85*g/cm3;
    G4double LWDensity = 0.998*g/cm3;
    G4double FuelDensity = 10.6*g/cm3;
    G4double AirDensity = 5.0807e-5*g/cm3;
    G4double ZrDensity = 6.49*g/cm3;
    G4double AlAlloyDensity = 2.70*g/cm3;
    G4double CadmiumDensity = 8.65*g/cm3;
    G4double HWDensity = 1.105*g/cm3;

    // Temperature Of Defined Materials
    // using data from 20043405
    G4double ReflectorTemp=(22.5+273.15);
    G4double LWTemp=(30.6+273.15);
    G4double FuelTemp=(57.32+273.15);
    G4double AirTemp=(18.0+273.15);
    G4double ZrTemp=(52.14+273.15);
    G4double AlAlloyTemp1=(20.0+273.15);
    G4double AlAlloyTemp2=(21.0+273.15);
    G4double AlAlloyTemp3=(22.0+273.15);
    G4double AlAlloyTemp4=(48.0+273.15);
    G4double CadmiumTemp=(50.0+273.15);
    G4double HWTemp=(20.5+273.15);

    // Defining all the pointers
    G4Isotope *C12, *C13, *N14, *N15, *O16, *O17, /*O18,*/ *Mg24,
              *Mg25, *Mg26, *Al27, *Si28, *Si29, *Si30, *Cr50,
              *Cr52, *Cr53, *Cr54, *Mn55, *Fe54, *Fe56, *Fe57,
              *Fe58, *Cu63, *Cu65, *Zr90, *Zr91, *Zr92, *Zr94,
              *Zr96, *Cd106, *Cd108, *Cd110, *Cd111, *Cd112,
              *Cd113, *Cd114, *Cd116, *U235, *U238;
    G4Element *H1, *D2, *Be, *Li6, *Li7, *B10, *B11, *C, *N, *Oxygen,
              *Mg, *Al, *Si, *Cr, *Mn, *Fe, *Cu, *Zirc, *ECd112,
              *ECd113, *Cd, *In115, *Sm148, *Sm149, *Sm150, *Sm152,
              *Gd155, *Gd157, *Eu151, *Eu153, *Ir191, *Ir193,*LEU;
    G4Material *World, *Air, *Reflector, *LW, *Fuel, *Zr, *AlAlloy1,
               *AlAlloy2, *AlAlloy3, *AlAlloy4, *Cadmium, *HW;

//    G4NistManager* manager = G4NistManager::Instance();

    // Hydrogen And Isotopes
    H1 = new G4Element("Hydrogen1", "H1", 1);
    H1->AddIsotope(new G4Isotope("H1", 1, 1, 1.0078250321*g/mole), 1);
    D2 = new G4Element("Hydrogen2", "H2", 1);
    D2->AddIsotope(new G4Isotope("H2", 1, 2, 2.0141017780*g/mole), 1);

    // Lithium Isotopes
    Li6 = new G4Element("Lithium6", "Li6", 1);
    Li6->AddIsotope(new G4Isotope("Li6", 3, 6, 6.0151223*g/mole), 1);
    Li7 = new G4Element("Lithium7", "Li7", 1);
    Li7->AddIsotope(new G4Isotope("Li7", 3, 7, 7.0160040*g/mole), 1);

    // Berylium And Isotopes
    Be = new G4Element("Berylium", "Be", 1);
    Be->AddIsotope(new G4Isotope("Be9", 4, 9, 9.0121822*g/mole), 1);

    // Boron Isotopes
    B10 = new G4Element("Boron10", "B10", 1);
    B10->AddIsotope(new G4Isotope("B10", 5, 10, 10.012937*g/mole), 1);
    B11 = new G4Element("Boron11", "B11", 1);
    B11->AddIsotope(new G4Isotope("B11", 5, 11, 11.009305*g/mole), 1);

    // Making Carbon isotopes
    C12 = new G4Isotope("C12", 6, 12, 12.000000*g/mole);
    C13 = new G4Isotope("C13", 6, 13, 13.003354*g/mole);

    // Naturally occuring Carbon
//    C = manager->FindOrBuildElement(6);
    C = new G4Element("Carbon", "C", 2);
    C->AddIsotope(C12, 98.93*perCent);
    C->AddIsotope(C13,  1.07*perCent);

    // Nitrogen Isotopes
    N14 = new G4Isotope("N14", 7, 14, 14.0030740052*g/mole);
    N15 = new G4Isotope("N15", 7, 15, 15.0001088984*g/mole);

    // Naturally occuring Nitrogen
    N = new G4Element("Nitrogen", "N", 2);
    N->AddIsotope(N14, 99.632*perCent);
    N->AddIsotope(N15,  0.368*perCent);

    // Make oxygen isotope and element
    O16 = new G4Isotope("O16", 8, 16, 15.995*g/mole);
    O17 = new G4Isotope("O17", 8, 17, 16.999*g/mole);
//    O18 = new G4Isotope("O18", 8, 18, 17.999*g/mole);

    // Natural occuring oxygen
    Oxygen = new G4Element("Oxygen", "O", 2);
    Oxygen->AddIsotope(O16, 99.962*perCent);
    Oxygen->AddIsotope(O17, 0.038*perCent);
//    Oxygen->AddIsotope(O18, 0.205*perCent);

    // Magnesium Isotopes
    Mg24 = new G4Isotope("Mg24", 12, 24, 23.9850423*g/mole);
    Mg25 = new G4Isotope("Mg25", 12, 25, 24.9858374*g/mole);
    Mg26 = new G4Isotope("Mg26", 12, 26, 25.9825937*g/mole);

    // Naturally Occuring Magnesium
    Mg = new G4Element("Magnesium", "Mg", 3);
    Mg->AddIsotope(Mg24, 78.99*perCent);
    Mg->AddIsotope(Mg25, 10.00*perCent);
    Mg->AddIsotope(Mg26, 11.01*perCent);

    // Making Aluminum Isotopes
    Al27 = new G4Isotope("Al27", 13, 27, 26.9815386*g/mole);

    // Naturally occuring Aluminum
    Al = new G4Element("Aluminum", "Al", 1);
    Al->AddIsotope(Al27, 1);

     // Making Silicon Isotopes
    Si28 = new G4Isotope("Si28", 14, 28, 27.9769271*g/mole);
    Si29 = new G4Isotope("Si29", 14, 29, 28.9764949*g/mole);
    Si30 = new G4Isotope("Si30", 14, 30, 29.9737707*g/mole);

    // Naturally occuring Silicon
    Si = new G4Element("Silicon", "Si", 3);
    Si->AddIsotope(Si28, 92.2297*perCent);
    Si->AddIsotope(Si29,  4.6832*perCent);
    Si->AddIsotope(Si30,  3.0871*perCent);

    // Chromium Isotopes
    Cr50 = new G4Isotope("Cr50", 24, 50, 49.9460464*g/mole);
    Cr52 = new G4Isotope("Cr52", 24, 52, 51.9405098*g/mole);
    Cr53 = new G4Isotope("Cr53", 24, 53, 52.9406513*g/mole);
    Cr54 = new G4Isotope("Cr54", 24, 54, 53.9388825*g/mole);

    // Naturally Occuring Chromium
    Cr = new G4Element("Chromium", "Cr", 4);
    Cr->AddIsotope(Cr50,  4.345*perCent);
    Cr->AddIsotope(Cr52, 83.789*perCent);
    Cr->AddIsotope(Cr53,  9.501*perCent);
    Cr->AddIsotope(Cr54,  2.365*perCent);

    // Manganese Isotopes
    Mn55 = new G4Isotope("Mn55", 25, 55, 54.9380471*g/mole);

    // Naturally occuring Manganese
    Mn = new G4Element("Manganese", "Mn", 1);
    Mn->AddIsotope(Mn55, 1.);

    // Making Iron Isotopes
    Fe54 = new G4Isotope("Fe54", 26, 54, 53.9396127*g/mole);
    Fe56 = new G4Isotope("Fe56", 26, 56, 55.9349393*g/mole);
    Fe57 = new G4Isotope("Fe57", 26, 57, 56.9353958*g/mole);
    Fe58 = new G4Isotope("Fe58", 26, 58, 57.9332773*g/mole);

    // Naturally Occuring Iron
    Fe = new G4Element("Iron", "Fe", 4);
    Fe->AddIsotope(Fe54,  5.845*perCent);
    Fe->AddIsotope(Fe56, 91.754*perCent);
    Fe->AddIsotope(Fe57,  2.119*perCent);
    Fe->AddIsotope(Fe58,  0.282*perCent);

    // Copper Isotopes
    Cu63 = new G4Isotope("Cu63", 29, 63, 62.9295989*g/mole);
    Cu65 = new G4Isotope("Cu65", 29, 65, 64.9277929*g/mole);

    // Naturally Occuring Copper
    Cu = new G4Element("Copper", "Cu", 2);
    Cu->AddIsotope(Cu63, 69.17*perCent);
    Cu->AddIsotope(Cu65, 30.83*perCent);

    // Making Zirconium isotopes and elements
    Zr90 = new G4Isotope("Zr90", 40, 90, 89.9047044*g/mole);
    Zr91 = new G4Isotope("Zr91", 40, 91, 90.9056458*g/mole);
    Zr92 = new G4Isotope("Zr92", 40, 92, 91.9050408*g/mole);
    Zr94 = new G4Isotope("Zr94", 40, 94, 93.9063152*g/mole);
    Zr96 = new G4Isotope("Zr96", 40, 96, 95.9082734*g/mole);

    // Natural Zirconium composition
    Zirc = new G4Element("Zirconium", "Zr", 5);
    Zirc->AddIsotope(Zr90, 50.706645*perCent);
    Zirc->AddIsotope(Zr91, 11.180922*perCent);
    Zirc->AddIsotope(Zr92, 17.277879*perCent);
    Zirc->AddIsotope(Zr94, 17.890875*perCent);
    Zirc->AddIsotope(Zr96,  2.943679*perCent);

    // Cadmium Isotopes
    Cd106 = new G4Isotope("Cd106", 48, 106, 105.906461*g/mole);
    Cd108 = new G4Isotope("Cd108", 48, 108, 107.904176*g/mole);
    Cd110 = new G4Isotope("Cd110", 48, 110, 109.903005*g/mole);
    Cd111 = new G4Isotope("Cd111", 48, 111, 110.904182*g/mole);
    Cd112 = new G4Isotope("Cd112", 48, 112, 111.902757*g/mole);
    Cd113 = new G4Isotope("Cd113", 48, 113, 112.904400*g/mole);
    Cd114 = new G4Isotope("Cd114", 48, 114, 113.903357*g/mole);
    Cd116 = new G4Isotope("Cd116", 48, 116, 115.904755*g/mole);


    // Cadmium Isotopes
    ECd112 = new G4Element("Cadmium112", "Cd112", 1);
    ECd112->AddIsotope(Cd112, 1);
    ECd113 = new G4Element("Cadmium113", "Cd113", 1);
    ECd113->AddIsotope(Cd113, 1);

    // Naturally Occuring Cadmium
    Cd = new G4Element("Cadmium", "Cd", 8);
    Cd->AddIsotope(Cd106,  1.25*perCent);
    Cd->AddIsotope(Cd108,  0.89*perCent);
    Cd->AddIsotope(Cd110, 12.49*perCent);
    Cd->AddIsotope(Cd111, 12.80*perCent);
    Cd->AddIsotope(Cd112, 24.13*perCent);
    Cd->AddIsotope(Cd113, 12.22*perCent);
    Cd->AddIsotope(Cd114, 28.73*perCent);
    Cd->AddIsotope(Cd116,  7.49*perCent);

    // Indium Isotopes
    In115 = new G4Element("Indium115", "In115", 1);
    In115->AddIsotope(new G4Isotope("In115", 49, 115, 114.903882*g/mole), 1);

    // Samarium Isotopes (Note: Could not get info on Sm137)
    Sm148 = new G4Element("Samatium148", "Am148", 1);
    Sm148->AddIsotope(new G4Isotope("Sm148", 62, 148, 147.914819*g/mole), 1);
    Sm149 = new G4Element("Samatium149", "Am149", 1);
    Sm149->AddIsotope(new G4Isotope("Sm149", 62, 149, 149.917180*g/mole), 1);
    Sm150 = new G4Element("Samatium150", "Am150", 1);
    Sm150->AddIsotope(new G4Isotope("Sm150", 62, 150, 149.917273*g/mole), 1);
    Sm152 = new G4Element("Samatium152", "Am152", 1);
    Sm152->AddIsotope(new G4Isotope("Sm152", 62, 152, 151.919728*g/mole), 1);

    // Gadolium Isotopes
    Gd155 = new G4Element("Gadolinium155", "Gd155", 1);
    Gd155->AddIsotope(new G4Isotope("Gd155", 64, 155, 154.922618*g/mole), 1);
    Gd157 = new G4Element("Gadolinium157", "Gd157", 1);
    Gd157->AddIsotope(new G4Isotope("Gd157", 64, 157, 156.923956*g/mole), 1);

    // Europium Isotopes
    Eu151 = new G4Element("Europium151", "Eu151", 1);
    Eu151->AddIsotope(new G4Isotope("Eu151", 63, 151, 150.919702*g/mole), 1);
    Eu153 = new G4Element("Europium153", "Eu153", 1);
    Eu153->AddIsotope(new G4Isotope("Eu153", 63, 153, 152.921225*g/mole), 1);

    // Iridium Isotopes
    Ir191 = new G4Element("Iridium191", "Ir191", 1);
    Ir191->AddIsotope(new G4Isotope("Ir191", 77, 191, 190.960584*g/mole), 1);
    Ir193 = new G4Element("Iridium193", "Ir193", 1);
    Ir193->AddIsotope(new G4Isotope("Ir193", 77, 193, 192.962917*g/mole), 1);

    // Making the Uranium isotopes
    U235 = new G4Isotope("U235", 92, 235, 235.0439*g/mole);
    U238 = new G4Isotope("U238", 92, 238, 238.0508*g/mole);

    // Low Enriched Uranium (LEU)
    LEU = new G4Element("Low Enriched Uranium", "LEU", 2);
    LEU->AddIsotope(U235, 19.89*perCent);
    LEU->AddIsotope(U238, 80.11*perCent);

    // (Marteial #0) Void Material
    World = new G4Material("Galactic", 1, 1, 1.e-25*g/cm3, kStateGas,
						   2.73*kelvin, 3.e-18*pascal);

    // (Material #1) Beryllium Sheild with Impurities
    Reflector = new G4Material("Reflector", ReflectorDensity, 17, kStateSolid, ReflectorTemp);
    Reflector->AddElement(Be,     0.9953863);
    Reflector->AddElement(Oxygen, 9.70113e-6);
    Reflector->AddElement(Al,     1.010534e-3);
    Reflector->AddElement(C,      1.515802e-3);
    Reflector->AddElement(Fe,     1.31695e-3);
    Reflector->AddElement(Si,     6.063207e-4);
    Reflector->AddElement(B10,    4.345298e-7);
    Reflector->AddElement(B11,    1.616855e-6);
    Reflector->AddElement(Mn,     1.515802e-4);
    Reflector->AddElement(Cd,     7.376901e-7);
    Reflector->AddElement(Li6,    1.313695e-7);
    Reflector->AddElement(Li7,    1.92001e-6);
    Reflector->AddElement(Sm149,  6.497736e-7);
    Reflector->AddElement(Gd155,  3.53687e-8);
    Reflector->AddElement(Gd157,  3.132657e-8);
    Reflector->AddElement(Eu151,  2.425283e-7);
    Reflector->AddElement(Eu153,  2.627389e-7);

    // (Material #2) Light Water
    LW = new G4Material("LightWater", LWDensity, 2, kStateLiquid, LWTemp);
    LW->AddElement(Oxygen, 1);
    LW->AddElement(H1,     2);

    // (Material #3) Fuel Rods (19.95% Enriched Uranium in (UO2))
    Fuel = new G4Material("Fuel", FuelDensity, 2, kStateSolid, FuelTemp);
    Fuel->AddElement(Oxygen, 2);
    Fuel->AddElement(LEU,    1);

    // (Material #4) Air
    Air = new G4Material("Air", AirDensity, 2, kStateGas, AirTemp);
    Air->AddElement(Oxygen, 0.21174);
    Air->AddElement(N,      0.78826);
    // (Material #5) Zr
    Zr = new G4Material("Zirconium", ZrDensity, 1, kStateSolid, ZrTemp);
    Zr->AddElement(Zirc, 1);

    // (Material #6) Aluminum with impurities
    AlAlloy1 = new G4Material("AlAlloy1", AlAlloyDensity, 5, kStateSolid, AlAlloyTemp1);
    AlAlloy1->AddElement(Al, 0.9792);
    AlAlloy1->AddElement(Si, 0.0060);
    AlAlloy1->AddElement(Cu, 0.0028);
    AlAlloy1->AddElement(Mg, 0.0100);
    AlAlloy1->AddElement(Cr, 0.0020);

    AlAlloy2 = new G4Material("AlAlloy2", AlAlloyDensity, 5, kStateSolid, AlAlloyTemp2);
    AlAlloy2->AddElement(Al, 0.9792);
    AlAlloy2->AddElement(Si, 0.0060);
    AlAlloy2->AddElement(Cu, 0.0028);
    AlAlloy2->AddElement(Mg, 0.0100);
    AlAlloy2->AddElement(Cr, 0.0020);

    AlAlloy3 = new G4Material("AlAlloy3", AlAlloyDensity, 5, kStateSolid, AlAlloyTemp3);
    AlAlloy3->AddElement(Al, 0.9792);
    AlAlloy3->AddElement(Si, 0.0060);
    AlAlloy3->AddElement(Cu, 0.0028);
    AlAlloy3->AddElement(Mg, 0.0100);
    AlAlloy3->AddElement(Cr, 0.0020);

    AlAlloy4 = new G4Material("AlAlloy4", AlAlloyDensity, 5, kStateSolid, AlAlloyTemp4);
    AlAlloy4->AddElement(Al, 0.9792);
    AlAlloy4->AddElement(Si, 0.0060);
    AlAlloy4->AddElement(Cu, 0.0028);
    AlAlloy4->AddElement(Mg, 0.0100);
    AlAlloy4->AddElement(Cr, 0.0020);

    // (Material #7) Cadmium
    Cadmium = new G4Material("Cadmium", CadmiumDensity, 1, kStateSolid, CadmiumTemp);
    Cadmium->AddElement(Cd, 1);

    // (Materail #8) Heavy Water
    HW = new G4Material("Heavy Water", HWDensity, 2, kStateLiquid, HWTemp);
    HW->AddElement(D2,     2);
    HW->AddElement(Oxygen, 1);


    // Add materials to the map indexed by either ZA (format ZZAAA or ZZ)
    // For composite materials:  world is 0, heavy water is 1, UHW is 2
    matMap["Galactic"] = World;
    matMap["H2O"] = LW;
    matMap["D2O"] = HW;
    matMap["Fuel"] = Fuel;
    matMap["Zirconium"] = Zr;
    matMap["AlAlloy1"] = AlAlloy1;
    matMap["AlAlloy2"] = AlAlloy2;
    matMap["AlAlloy3"] = AlAlloy3;
    matMap["AlAlloy4"] = AlAlloy4;
    matMap["Reflector"] = Reflector;
    matMap["Cadmium"] = Cadmium;
    matMap["Air"] = Air;

    matChanged = false;

    return;
}
