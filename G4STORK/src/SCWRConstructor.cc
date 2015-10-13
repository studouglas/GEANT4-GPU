/*
SCWRConstructor.cc

Created by:		Wesley Ford
Date:			23-05-2012
Modified:       11-03-2013

Source code for the CANDU 6 lattice geometry and materials

*/

#include "SCWRConstructor.hh"

// Constructor
SCWRConstructor::SCWRConstructor()
: StorkVWorldConstructor(), cellLogical(0), pressTubeLogical1(0), pressTubeLogical2(0), pressTubeLogical3(0), pressTubeLogical4(0), outLinerLogical(0), insulatorLogical1(0),
  insulatorLogical2(0), insulatorLogical3(0), insulatorLogical4(0), linerLogical(0), coolantLogical(0), outSheatheLogical(0), inSheatheLogical(0), outFuelLogical1(0),
  outFuelLogical2(0), outFuelLogical3(0), outFuelLogical4(0), inFuelLogical1(0), inFuelLogical2(0), inFuelLogical3(0), inFuelLogical4(0), /*outFlowTubeLogical(0), flowTubeLogical1(0), flowTubeLogical2(0), inFlowTubeLogical(0)*/ flowTubeLogical(0),
  centralCoolantLogical(0)
{
	// Set default member variables (from file or default values)



	// Set up variable property map
	/*
	variablePropMap[MatPropPair(fuel,temperature)] = &fuelTemp;
    variablePropMap[MatPropPair(fuel,density)] = &fuelDensity;
    variablePropMap[MatPropPair(coolant,temperature)] = &coolantTemp;
    variablePropMap[MatPropPair(coolant,density)] = &coolantDensity;
    variablePropMap[MatPropPair(moderator,temperature)] = &moderatorTemp;
    variablePropMap[MatPropPair(moderator,density)] = &moderatorDensity;
    variablePropMap[MatPropPair(all,dimension)] = &latticePitch;
    */
    cellVisAtt=NULL;
    pressTubeVisAtt=NULL;
    outLinerVisAtt=NULL;
    insulatorVisAtt=NULL;
    linerVisAtt=NULL;
    coolantVisAtt=NULL;
    outSheatheVisAtt=NULL;
    inSheatheVisAtt=NULL;
    outFuelVisAtt=NULL;
    inFuelVisAtt=NULL;
    flowTubeVisAtt=NULL;
    centralCoolantVisAtt=NULL;

    // Stored variables from infile
    latticePitch= 25.*cm;

    moderatorTemp=342.8200*kelvin;
    moderatorDensity=1.0851*g/cm3;

    pressTubeTemp[0]=459.8850*kelvin;
    pressTubeTemp[1]=441.8700*kelvin;
    pressTubeTemp[2]=424.2300*kelvin;
    pressTubeTemp[3]=406.9300*kelvin;

    pressTubeDensity=6.52*g/cm3;

    outLinerTemp=470.5200*kelvin;
    outLinerDensity=6.52*g/cm3;

    insulatorTemp[0]=643.2050*kelvin;
    insulatorTemp[1]=592.4750*kelvin;
    insulatorTemp[2]=543.2950*kelvin;
    insulatorTemp[3]=495.5700*kelvin;
    insulatorDensity=5.83*g/cm3;

    linerTemp=670.2600*kelvin;
    linerDensity=7.9*g/cm3;

    coolantTemp=680.62*kelvin;
    //coolantDensity=0.1512*g/cm3;
    coolantDensity=0.001*g/cm3;

    inSheatheTemp=768.8200*kelvin;
    inSheatheDensity=7.9*g/cm3;

    outSheatheTemp=780.9300*kelvin;
    outSheatheDensity=7.9*g/cm3;

    innerFuelTemp[0]=1389.8000*kelvin;
    innerFuelTemp[1]=1288.6600*kelvin;
    innerFuelTemp[2]=1118.6700*kelvin;
    innerFuelTemp[3]=899.625*kelvin;
    innerFuelDensity=9.91*g/cm3;

    outerFuelTemp[0]=1625.0500*kelvin;
    outerFuelTemp[1]=1480.2500*kelvin;
    outerFuelTemp[2]=1241.7850*kelvin;
    outerFuelTemp[3]=945.8700*kelvin;
    outerFuelDensity=9.87*g/cm3;

    flowTubeTemp=659.1800*kelvin;
    flowTubeDensity=7.9*g/cm3;

    centralCoolantTemp=629.0100*kelvin;
    //centralCoolantDensity=0.6059*g/cm3;
    centralCoolantDensity=0.001*g/cm3;
}


// Desturctor
SCWRConstructor::~SCWRConstructor()
{
	// Delete visualization attributes
	if(cellVisAtt)
        delete cellVisAtt;
	if(pressTubeVisAtt)
        delete pressTubeVisAtt;
    if(outLinerVisAtt)
        delete outLinerVisAtt;
    if(insulatorVisAtt)
        delete insulatorVisAtt;
    if(linerVisAtt)
        delete linerVisAtt;
    if(coolantVisAtt)
        delete coolantVisAtt;
    if(outSheatheVisAtt)
        delete outSheatheVisAtt;
    if(inSheatheVisAtt)
        delete inSheatheVisAtt;
    if(outFuelVisAtt)
        delete outFuelVisAtt;
    if(inFuelVisAtt)
        delete inFuelVisAtt;
    //if(outFlowTubeVisAtt)
//    delete outFlowTubeVisAtt;
    if(flowTubeVisAtt)
        delete flowTubeVisAtt;
    //if(inFlowTubeVisAtt)
//    delete inFlowTubeVisAtt;
    if(centralCoolantVisAtt)
        delete centralCoolantVisAtt;
}


// ConstructWorld()
// Construct the geometry and materials of the CANDU 6 lattice cell.
G4VPhysicalVolume* SCWRConstructor::ConstructWorld()
{
	// Lattic cell dimensions
	G4double buffer = 1.0*cm;
	reactorDim = G4ThreeVector(latticePitch/2.0,latticePitch/2.0,latticePitch/2.0);

	encWorldDim = 2.0 * G4ThreeVector(reactorDim[0]+buffer,reactorDim[1]+buffer,
									  reactorDim[2]+buffer);

	G4SolidStore* theSolids = G4SolidStore::GetInstance();


	// Set static dimensions

    // Pressure Tube dimensions
    G4double pressTube1Radmax = 8.15*cm;
    G4double pressTube2Radmax = 8.45*cm;
    G4double pressTube3Radmax = 8.75*cm;
    G4double pressTube4Radmax = 9.05*cm;
    G4double pressTubeLen = reactorDim[2];

    // outer liner dimensions
    G4double outLinerRadmax = 7.85*cm;
    G4double outLinerLen = reactorDim[2];

    // insulator dimensions
    G4double insulator1Radmax = 7.388*cm;
    G4double insulator2Radmax = 7.525*cm;
    G4double insulator3Radmax = 7.663*cm;
    G4double insulator4Radmax = 7.8*cm;
    G4double insulatorLen = reactorDim[2];

    // liner dimensions
    G4double linerRadmax = 7.25*cm;
    G4double linerLen = reactorDim[2];

    // coolant dimensions
    G4double coolantRadmax = 7.2*cm;
    G4double coolantLen = reactorDim[2];

    // outer sheathe dimensions
    G4double outSheatheRadmax = 0.5*cm;
    G4double outSheatheLen = reactorDim[2];

    // inner sheathe dimensions
    G4double inSheatheRadmax = 0.475*cm;
    G4double inSheatheLen = reactorDim[2];

    G4double ringRad[2] = {6.575*cm,5.4*cm};
    //G4double secondRingOffset = 0.*radian;

    // outer fuel dimensions
    G4double outFuel1Radmax = 0.11*cm;
    G4double outFuel2Radmax = 0.22*cm;
    G4double outFuel3Radmax = 0.33*cm;
    G4double outFuel4Radmax = 0.44*cm;
    G4double outFuelLen = reactorDim[2];

    // inner fuel dimensions
    G4double inFuel1Radmax = 0.104*cm;
    G4double inFuel2Radmax = 0.208*cm;
    G4double inFuel3Radmax = 0.311*cm;
    G4double inFuel4Radmax = 0.415*cm;
    G4double inFuelLen = reactorDim[2];

//    // outer fuel dimensions
//    G4double outFlowTubeRadmax = 4.7*cm;
//    G4double outFlowTubeLen = reactorDim[2];

    // Pressure Tube dimensions
//    G4double flowTube1Radmax = 4.65*cm;
//    G4double flowTube2Radmax = 4.7*cm;
    G4double flowTubeRadmax = 4.7*cm;
    G4double flowTubeLen = reactorDim[2];

//    // inner fuel dimensions
//    G4double inFlowTubeRadmax = 3.65*cm;
//    G4double inFlowTubeLen = reactorDim[2];

    // central coolant dimensions
    G4double centralCoolantRadmax = 4.6*cm;
    G4double centralCoolantLen = reactorDim[2];

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

		// Create pressure tube
		new G4Tubs("pressTube1", 0., pressTube1Radmax, pressTubeLen, 0.,
					2.0*CLHEP::pi);

        new G4Tubs("pressTube2", 0., pressTube2Radmax, pressTubeLen, 0.,
					2.0*CLHEP::pi);

        new G4Tubs("pressTube3", 0., pressTube3Radmax, pressTubeLen, 0.,
					2.0*CLHEP::pi);

        new G4Tubs("pressTube4", 0., pressTube4Radmax, pressTubeLen, 0.,
					2.0*CLHEP::pi);

		// Create outer liner tube
		new G4Tubs("outLinerTube", 0., outLinerRadmax, outLinerLen, 0., 2.0*CLHEP::pi);
		// Create insulator

		new G4Tubs("insulatorTube1", 0., insulator1Radmax, insulatorLen, 0.,
					2.0*CLHEP::pi);

        new G4Tubs("insulatorTube2", 0., insulator2Radmax, insulatorLen, 0.,
					2.0*CLHEP::pi);

        new G4Tubs("insulatorTube3", 0., insulator3Radmax, insulatorLen, 0.,
					2.0*CLHEP::pi);

        new G4Tubs("insulatorTube4", 0., insulator4Radmax, insulatorLen, 0.,
					2.0*CLHEP::pi);

		// Create liner tube
		new G4Tubs("linerTube", 0., linerRadmax, linerLen, 0.,
					2.0*CLHEP::pi);
		// Create coolant solid
		new G4Tubs("coolantTube", 0., coolantRadmax, coolantLen, 0.,
					2.0*CLHEP::pi);
		// Create the sheathe for fuel pins
		new G4Tubs("outSheatheTube", 0., outSheatheRadmax, outSheatheLen, 0.,
					2.0*CLHEP::pi);
        // Create the sheathe for fuel pins
		new G4Tubs("inSheatheTube", 0., inSheatheRadmax, inSheatheLen, 0.,
					2.0*CLHEP::pi);

		// Create outer fuel pins
		new G4Tubs("outFuelCyl1", 0., outFuel1Radmax, outFuelLen, 0.,2.0*CLHEP::pi);

		new G4Tubs("outFuelCyl2", 0., outFuel2Radmax, outFuelLen, 0.,2.0*CLHEP::pi);

		new G4Tubs("outFuelCyl3", 0., outFuel3Radmax, outFuelLen, 0.,2.0*CLHEP::pi);

		new G4Tubs("outFuelCyl4", 0., outFuel4Radmax, outFuelLen, 0.,2.0*CLHEP::pi);

		// Create inner fuel pins
		new G4Tubs("inFuelCyl1", 0., inFuel1Radmax, inFuelLen, 0., 2.0*CLHEP::pi);

		new G4Tubs("inFuelCyl2", 0., inFuel2Radmax, inFuelLen, 0., 2.0*CLHEP::pi);

		new G4Tubs("inFuelCyl3", 0., inFuel3Radmax, inFuelLen, 0., 2.0*CLHEP::pi);

		new G4Tubs("inFuelCyl4", 0., inFuel4Radmax, inFuelLen, 0., 2.0*CLHEP::pi);


//		// Create outter flow tube
//		new G4Tubs("outFlowTube", 0., outFlowTubeRadmax, outFlowTubeLen, 0.,
//					2.0*CLHEP::pi);

		// Create flow tube
//		new G4Tubs("flowTube1", 0., flowTube1Radmax, flowTubeLen, 0.,
//					2.0*CLHEP::pi);
//
//        new G4Tubs("flowTube2", 0., flowTube2Radmax, flowTubeLen, 0.,
//					2.0*CLHEP::pi);

        new G4Tubs("flowTube", 0., flowTubeRadmax, flowTubeLen, 0.,
					2.0*CLHEP::pi);

//		// Create inner flow tube
//		new G4Tubs("inFlowTube", 0., inFlowTubeRadmax, inFlowTubeLen, 0.,
//					2.0*CLHEP::pi);
		// Create the central coolant
		new G4Tubs("centralCoolantCyl", 0., centralCoolantRadmax, centralCoolantLen, 0.,
					2.0*CLHEP::pi);

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

    // Create the Pressure Tube
    pressTubeLogical1 = new G4LogicalVolume(theSolids->GetSolid("pressTube1"),
                        matMap["PressTube1"], "pressTubeLogical1",0,0,0);

    pressTubeLogical2 = new G4LogicalVolume(theSolids->GetSolid("pressTube2"),
                        matMap["PressTube2"], "pressTubeLogical2",0,0,0);

    pressTubeLogical3 = new G4LogicalVolume(theSolids->GetSolid("pressTube3"),
                        matMap["PressTube3"], "pressTubeLogical3",0,0,0);

    pressTubeLogical4 = new G4LogicalVolume(theSolids->GetSolid("pressTube4"),
                        matMap["PressTube4"], "pressTubeLogical4",0,0,0);


    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), pressTubeLogical4,
					  "pressTubePhysical4",cellLogical,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), pressTubeLogical3,
					  "pressTubePhysical3",pressTubeLogical4,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), pressTubeLogical2,
					  "pressTubePhysical2",pressTubeLogical3,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), pressTubeLogical1,
					  "pressTubePhysical1",pressTubeLogical2,0,0);

    // Create the Pressure Tube
    outLinerLogical = new G4LogicalVolume(theSolids->GetSolid("outLinerTube"),
                        matMap["OutLiner"], "outLinerLogical",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), outLinerLogical,
					  "outLinerPhysical",pressTubeLogical1,0,0);
    // Create the Pressure Tube
    insulatorLogical1 = new G4LogicalVolume(theSolids->GetSolid("insulatorTube1"),
                        matMap["Insulator1"], "insulatorLogical1",0,0,0);

    insulatorLogical2 = new G4LogicalVolume(theSolids->GetSolid("insulatorTube2"),
                        matMap["Insulator2"], "insulatorLogical2",0,0,0);

    insulatorLogical3 = new G4LogicalVolume(theSolids->GetSolid("insulatorTube3"),
                        matMap["Insulator3"], "insulatorLogical3",0,0,0);

    insulatorLogical4 = new G4LogicalVolume(theSolids->GetSolid("insulatorTube4"),
                        matMap["Insulator4"], "insulatorLogical4",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), insulatorLogical4,
					  "insulatorPhysical4",outLinerLogical,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), insulatorLogical3,
					  "insulatorPhysical3",insulatorLogical4,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), insulatorLogical2,
					  "insulatorPhysical2",insulatorLogical3,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), insulatorLogical1,
					  "insulatorPhysical1",insulatorLogical2,0,0);

	// Create the Pressure Tube
    linerLogical = new G4LogicalVolume(theSolids->GetSolid("linerTube"),
                        matMap["Liner"], "linerLogical",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), linerLogical,
					  "linerPhysical",insulatorLogical1,0,0);

    // Create the Coolant
    coolantLogical = new G4LogicalVolume(theSolids->GetSolid("coolantTube"),
                        matMap["Coolant"], "coolantLogical",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), coolantLogical,
					  "coolantPhysical",linerLogical,0,0);

    // Create the Sheathe
    outSheatheLogical = new G4LogicalVolume(theSolids->GetSolid("outSheatheTube"),
                        matMap["OutSheathe"], "outSheatheLogical",0,0,0);

    inSheatheLogical = new G4LogicalVolume(theSolids->GetSolid("inSheatheTube"),
                        matMap["InSheathe"], "inSheatheLogical",0,0,0);


    // Create a fuel
    outFuelLogical1 = new G4LogicalVolume(theSolids->GetSolid("outFuelCyl1"),
                        matMap["OuterFuel1"], "outFuelLogical1",0,0,0);
    // Create a fuel
    outFuelLogical2 = new G4LogicalVolume(theSolids->GetSolid("outFuelCyl2"),
                        matMap["OuterFuel2"], "outFuelLogical2",0,0,0);
    // Create a fuel
    outFuelLogical3 = new G4LogicalVolume(theSolids->GetSolid("outFuelCyl3"),
                        matMap["OuterFuel3"], "outFuelLogical3",0,0,0);
    // Create a fuel
    outFuelLogical4 = new G4LogicalVolume(theSolids->GetSolid("outFuelCyl4"),
                        matMap["OuterFuel4"], "outFuelLogical4",0,0,0);


    // Create a fuel
    inFuelLogical1 = new G4LogicalVolume(theSolids->GetSolid("inFuelCyl1"),
                        matMap["InnerFuel1"], "inFuelLogical1",0,0,0);

    inFuelLogical2 = new G4LogicalVolume(theSolids->GetSolid("inFuelCyl2"),
                        matMap["InnerFuel2"], "inFuelLogical2",0,0,0);

    inFuelLogical3 = new G4LogicalVolume(theSolids->GetSolid("inFuelCyl3"),
                        matMap["InnerFuel3"], "inFuelLogical3",0,0,0);

    inFuelLogical4 = new G4LogicalVolume(theSolids->GetSolid("inFuelCyl4"),
                        matMap["InnerFuel4"], "inFuelLogical4",0,0,0);

//                        // Create the Pressure Tube
//    outFlowTubeLogical = new G4LogicalVolume(theSolids->GetSolid("outFlowTube"),
//                        matMap["OutFlowTube"], "outFlowTubeLogical",0,0,0);
//
//    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), outFlowTubeLogical,
//					  "outFlowTubePhysical",coolantLogical,0,0);
//
//
//    // Create the Pressure Tube
//    flowTubeLogical1 = new G4LogicalVolume(theSolids->GetSolid("flowTube1"),
//                        matMap["FlowTube1"], "flowTubeLogical1",0,0,0);
//
//    // Create the Pressure Tube
//    flowTubeLogical2 = new G4LogicalVolume(theSolids->GetSolid("flowTube2"),
//                        matMap["FlowTube2"], "flowTubeLogical2",0,0,0);
//
//    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), flowTubeLogical2,
//					  "flowTubePhysical2",outFlowTubeLogical,0,0);
//
//    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), flowTubeLogical1,
//					  "flowTubePhysical1",flowTubeLogical2,0,0);

    flowTubeLogical = new G4LogicalVolume(theSolids->GetSolid("flowTube"),
                        matMap["FlowTube"], "flowTubeLogical",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), flowTubeLogical,
					  "flowTubePhysical",coolantLogical,0,0);


//    // Create the Pressure Tube
//    inFlowTubeLogical = new G4LogicalVolume(theSolids->GetSolid("inFlowTube"),
//                        matMap["InFlowTube"], "inFlowTubeLogical",0,0,0);
//
//    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), inFlowTubeLogical,
//					  "inFlowTubePhysical",flowTubeLogical1,0,0);

	// Create the Pressure Tube
    centralCoolantLogical = new G4LogicalVolume(theSolids->GetSolid("centralCoolantCyl"),
                        matMap["CentralCoolant"], "centralCoolantLogical",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), centralCoolantLogical,
					  "centralCoolantPhysical",flowTubeLogical,0,0);


    // Create fuel bundle

    // Rotation and translation of the rod and sheathe
	std::stringstream volName;

    // Set name for sheathe physical volume
    volName << 0;

    // Place pins for outer ring
    for( G4int i = 0; i < 32; i++ )
    {
        // Reset string stream
        volName.str("");

        volName << i;

             G4ThreeVector Tm(ringRad[0]*cos(2.0*i*CLHEP::pi/32),
                              ringRad[0]*sin(2.0*i*CLHEP::pi/32), 0.);

            new G4PVPlacement(0, Tm, outSheatheLogical,"outSheathePhysical " +
                                volName.str(),coolantLogical,0,0);

    }

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), outFuelLogical4,"outFuelPhysical4",
					  outSheatheLogical,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), outFuelLogical3,"outFuelPhysical3",
					  outFuelLogical4,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), outFuelLogical2,"outFuelPhysical2",
					  outFuelLogical3,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), outFuelLogical1,"outFuelPhysical1",
					  outFuelLogical2,0,0);

	// Place pins for inner ring
    for( G4int i = 0; i < 32; i++ )
    {
        // Reset string stream
        volName.str("");

        volName << i;

             G4ThreeVector Tm(ringRad[1]*cos(2.0*i*CLHEP::pi/32),
                              ringRad[1]*sin(2.0*i*CLHEP::pi/32), 0.);

            new G4PVPlacement(0, Tm, inSheatheLogical,"inSheathePhysical " +
                                volName.str(),coolantLogical,0,0);

    }

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), inFuelLogical4,"inFuelPhysical4",
					  inSheatheLogical,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), inFuelLogical3,"inFuelPhysical3",
					  inFuelLogical4,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), inFuelLogical2,"inFuelPhysical2",
					  inFuelLogical3,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), inFuelLogical1,"inFuelPhysical1",
					  inFuelLogical2,0,0);

	// Add sensitive detector to ALL logical volumes
	worldLogical->SetSensitiveDetector( sDReactor );
	cellLogical->SetSensitiveDetector( sDReactor );
	pressTubeLogical1->SetSensitiveDetector( sDReactor );
	pressTubeLogical2->SetSensitiveDetector( sDReactor );
	pressTubeLogical3->SetSensitiveDetector( sDReactor );
	pressTubeLogical4->SetSensitiveDetector( sDReactor );
	outLinerLogical->SetSensitiveDetector( sDReactor );
	insulatorLogical1->SetSensitiveDetector( sDReactor );
	insulatorLogical2->SetSensitiveDetector( sDReactor );
	insulatorLogical3->SetSensitiveDetector( sDReactor );
	insulatorLogical4->SetSensitiveDetector( sDReactor );
	linerLogical->SetSensitiveDetector( sDReactor );
	coolantLogical->SetSensitiveDetector( sDReactor );
	outSheatheLogical->SetSensitiveDetector( sDReactor );
	inSheatheLogical->SetSensitiveDetector( sDReactor );
	outFuelLogical1->SetSensitiveDetector( sDReactor );
	outFuelLogical2->SetSensitiveDetector( sDReactor );
	outFuelLogical3->SetSensitiveDetector( sDReactor );
	outFuelLogical4->SetSensitiveDetector( sDReactor );
	inFuelLogical1->SetSensitiveDetector( sDReactor );
	inFuelLogical2->SetSensitiveDetector( sDReactor );
	inFuelLogical3->SetSensitiveDetector( sDReactor );
	inFuelLogical4->SetSensitiveDetector( sDReactor );
//	outFlowTubeLogical->SetSensitiveDetector( sDReactor );
//	flowTubeLogical1->SetSensitiveDetector( sDReactor );
//	flowTubeLogical2->SetSensitiveDetector( sDReactor );
	flowTubeLogical->SetSensitiveDetector( sDReactor );
//	inFlowTubeLogical->SetSensitiveDetector( sDReactor );
	centralCoolantLogical->SetSensitiveDetector( sDReactor );


    // Set visualization attributes

    if(worldVisAtt)
        delete worldVisAtt;
    if(cellVisAtt)
        delete cellVisAtt;
	if(pressTubeVisAtt)
        delete pressTubeVisAtt;
    if(outLinerVisAtt)
        delete outLinerVisAtt;
    if(insulatorVisAtt)
        delete insulatorVisAtt;
    if(linerVisAtt)
        delete linerVisAtt;
    if(coolantVisAtt)
        delete coolantVisAtt;
    if(outSheatheVisAtt)
        delete outSheatheVisAtt;
    if(inSheatheVisAtt)
        delete inSheatheVisAtt;
    if(outFuelVisAtt)
        delete outFuelVisAtt;
    if(inFuelVisAtt)
        delete inFuelVisAtt;
    //if(outFlowTubeVisAtt)
//    delete outFlowTubeVisAtt;
    if(flowTubeVisAtt)
        delete flowTubeVisAtt;
    //if(inFlowTubeVisAtt)
//    delete inFlowTubeVisAtt;
    if(centralCoolantVisAtt)
        delete centralCoolantVisAtt;

    worldVisAtt = new G4VisAttributes(G4Colour(1.,1.,1.));
    worldVisAtt->SetVisibility(false);
    worldLogical->SetVisAttributes(worldVisAtt);

    cellVisAtt = new G4VisAttributes(G4Colour(0,0,1));
    cellVisAtt->SetVisibility(1);
    cellLogical->SetVisAttributes(cellVisAtt);

    pressTubeVisAtt = new G4VisAttributes(G4Colour(1,0,0));
    pressTubeVisAtt->SetVisibility(1);
    pressTubeLogical1->SetVisAttributes(pressTubeVisAtt);
    pressTubeLogical2->SetVisAttributes(pressTubeVisAtt);
    pressTubeLogical3->SetVisAttributes(pressTubeVisAtt);
    pressTubeLogical4->SetVisAttributes(pressTubeVisAtt);

    outLinerVisAtt = new G4VisAttributes(G4Colour(1,0.5,0));
    outLinerVisAtt->SetVisibility(1);
    outLinerLogical->SetVisAttributes(outLinerVisAtt);

    insulatorVisAtt = new G4VisAttributes(G4Colour(1,1,0.5));
    insulatorVisAtt->SetVisibility(1);
    insulatorLogical1->SetVisAttributes(insulatorVisAtt);
    insulatorLogical2->SetVisAttributes(insulatorVisAtt);
    insulatorLogical3->SetVisAttributes(insulatorVisAtt);
    insulatorLogical4->SetVisAttributes(insulatorVisAtt);

    linerVisAtt = new G4VisAttributes(G4Colour(0,1,0));
    linerVisAtt->SetVisibility(1);
    linerLogical->SetVisAttributes(linerVisAtt);

    coolantVisAtt = new G4VisAttributes(G4Colour(0,1,1));
    coolantVisAtt->SetVisibility(1);
    coolantLogical->SetVisAttributes(coolantVisAtt);

    outSheatheVisAtt = new G4VisAttributes(G4Colour(0.5,0,1));
    outSheatheVisAtt->SetVisibility(true);
    outSheatheLogical->SetVisAttributes(outSheatheVisAtt);

    inSheatheVisAtt = new G4VisAttributes(G4Colour(1,0,1));
    inSheatheVisAtt->SetVisibility(true);
    inSheatheLogical->SetVisAttributes(inSheatheVisAtt);

    outFuelVisAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    outFuelVisAtt->SetVisibility(true);
    outFuelLogical1->SetVisAttributes(outFuelVisAtt);
    outFuelLogical2->SetVisAttributes(outFuelVisAtt);
    outFuelLogical3->SetVisAttributes(outFuelVisAtt);
    outFuelLogical4->SetVisAttributes(outFuelVisAtt);

    inFuelVisAtt = new G4VisAttributes(G4Colour(1.,0.5,0.));
    inFuelVisAtt->SetVisibility(true);
    inFuelLogical1->SetVisAttributes(inFuelVisAtt);
    inFuelLogical2->SetVisAttributes(inFuelVisAtt);
    inFuelLogical3->SetVisAttributes(inFuelVisAtt);
    inFuelLogical4->SetVisAttributes(inFuelVisAtt);

//    outFlowTubeVisAtt = new G4VisAttributes(G4Colour(1,1,0));
//    outFlowTubeLogical->SetVisAttributes(outFlowTubeVisAtt);
//
    flowTubeVisAtt = new G4VisAttributes(G4Colour(0,1,0));
    flowTubeVisAtt->SetVisibility(1);
    flowTubeLogical->SetVisAttributes(flowTubeVisAtt);
//    flowTubeLogical1->SetVisAttributes(flowTubeVisAtt);
//    flowTubeLogical2->SetVisAttributes(flowTubeVisAtt);
//
//    inFlowTubeVisAtt = new G4VisAttributes(G4Colour(0.5,0.,1.));
//    inFlowTubeLogical->SetVisAttributes(inFlowTubeVisAtt);

    centralCoolantVisAtt = new G4VisAttributes(G4Colour(0.,1.,1.));
    centralCoolantVisAtt->SetVisibility(1);
    centralCoolantLogical->SetVisAttributes(centralCoolantVisAtt);



    return worldPhysical;
}


// ConstructMaterials()
// Define and build the materials in the C6 lattice cell.
void SCWRConstructor::ConstructMaterials()
{
    // Elements, isotopes and materials
    G4Isotope *H1, *H2, *C12, *C13, *O16, *Si28, *Si29, *Si30, *P31, *S32, *S33, *S34, *S36,
              *Cr50, *Cr52, *Cr53, *Cr54, *Mn55, *Fe54, *Fe56, *Fe57, *Fe58, *Ni58, *Ni60, *Ni64,
              *Nb93, *Y89, *Mo92, *Mo94, *Mo95, *Mo96, *Mo97, *Mo98, *Mo100,
              *Sn112, *Sn114, *Sn115, *Sn116, *Sn117, *Sn118, *Sn119, *Sn120, *Sn122, *Sn124,
              *Zr90, *Zr91, *Zr92, *Zr94, *Zr96, *Th232, *Pu238, *Pu239, *Pu240, *Pu241, *Pu242;

    G4Element *H, *D, *C, *Oxygen, *Si, *P, *S, *Cr, *Mn, *Fe, *Ni, *Nb, *Y, *Mo, *Sn, *Zr, *Th, *Pu;

    G4Material *World, *Moderator, *PressTube1, *PressTube2, *PressTube3, *PressTube4, *OutLiner, *Insulator1, *Insulator2, *Insulator3, *Insulator4, *Liner, *OutSheathe, *InSheathe,
    *OuterFuel1, *OuterFuel2, *OuterFuel3, *OuterFuel4, *InnerFuel1, *InnerFuel2, *InnerFuel3, *InnerFuel4, /*OutFlowTube, *FlowTube1, *FlowTube2, *InFlowTube,*/ *FlowTube, *CentralCoolant ,*ExelLiner,
    *ZircSteel, *ZircHydrid, *OuterFuel, *InnerFuel, *H2O, *D2O, *Coolant, *Insulator;


    // Make hydrogen elements
    H1 = new G4Isotope("H1", 1, 1, 1.008*g/mole);
    H = new G4Element("Hydrogen", "H", 1);
    H->AddIsotope(H1, 1.0);

    H2 = new G4Isotope("H2", 1, 2, 2.014*g/mole);
    D = new G4Element("Deterium", "D", 1);
    D->AddIsotope(H2, 1.0);

    C12 = new G4Isotope("C12", 6, 12, 12.0*g/mole);
    C13 = new G4Isotope("C13", 6, 13, 13.00335*g/mole);

    C = new G4Element("Carbon", "C", 2);
    C->AddIsotope(C12, 98.83*perCent);
    C->AddIsotope(C13,  1.07*perCent);


    // Make oxygen isotope and element
    O16 = new G4Isotope("O16", 8, 16, 15.995*g/mole);

    Oxygen = new G4Element("Oxygen", "O", 1);
    Oxygen->AddIsotope(O16, 100.*perCent);

    Si28 = new G4Isotope("Si28", 14, 28, 27.9769*g/mole);
    Si29 = new G4Isotope("Si29", 14, 29, 28.9765*g/mole);
    Si30 = new G4Isotope("Si30", 14, 28, 29.9738*g/mole);

    Si = new G4Element("Silicon", "Si", 3);
    Si->AddIsotope(Si28, 92.223*perCent);
    Si->AddIsotope(Si29,  4.685*perCent);
    Si->AddIsotope(Si30,  3.092*perCent);

    // Make oxygen isotope and element
    P31 = new G4Isotope("P31", 15, 31, 30.97376*g/mole);

    P = new G4Element("Phosphorus", "P", 1);
    P->AddIsotope(P31, 100.*perCent);

    S32 = new G4Isotope("S32", 16, 32, 31.9721*g/mole);
    S33 = new G4Isotope("S33", 16, 33, 32.9715*g/mole);
    S34 = new G4Isotope("S34", 16, 34, 33.9679*g/mole);
    S36 = new G4Isotope("S36", 16, 36, 35.9679*g/mole);

    S = new G4Element("Sulphur", "S", 4);
    S->AddIsotope(S32, 94.93*perCent);
    S->AddIsotope(S33,  0.76*perCent);
    S->AddIsotope(S34,  4.29*perCent);
    S->AddIsotope(S36,  0.02*perCent);

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

    //make chromium isotopes and element
    Mn55 = new G4Isotope("Mn55", 25, 55, 54.9380*g/mole);

    Mn = new G4Element("Manganese", "Mn", 1);
    Mn->AddIsotope(Mn55,  100.*perCent);

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

    //make nickel isotopes and element
    Ni58 = new G4Isotope("Ni58", 28, 58, 57.9353429*g/mole);
    Ni60 = new G4Isotope("Ni60", 28, 60, 59.9307864*g/mole);
    Ni64 = new G4Isotope("Ni64", 28, 64, 63.9279660*g/mole);

    Ni = new G4Element("Nickel", "Ni", 3);
    Ni->AddIsotope(Ni58, 70.913*perCent);
    Ni->AddIsotope(Ni60, 28.044*perCent);
    Ni->AddIsotope(Ni64,  1.043*perCent);

    //make niobium isotopes and element
    Nb93 = new G4Isotope("Nb93", 41, 93, 92.9063781*g/mole);

    Nb = new G4Element("Niobium", "Nb", 1);
    Nb->AddIsotope(Nb93, 100*perCent);

    Y89 = new G4Isotope("Y89", 39, 89, 99.9058*g/mole);

    Y = new G4Element("Yttrium", "Y", 1);
    Y->AddIsotope(Y89,  100.*perCent);

        //make Zirconium isotopes and element
    Mo92 = new G4Isotope("Mo92", 42, 92, 91.9068*g/mole);
    Mo94 = new G4Isotope("Mo94", 42, 94, 93.9051*g/mole);
    Mo95 = new G4Isotope("Mo95", 42, 95, 94.9058*g/mole);
    Mo96 = new G4Isotope("Mo96", 42, 96, 95.9047*g/mole);
    Mo97 = new G4Isotope("Mo97", 42, 97, 96.9060*g/mole);
    Mo98 = new G4Isotope("Mo98", 42, 98, 97.9054*g/mole);
    Mo100 = new G4Isotope("Mo100", 42, 100, 99.9075*g/mole);

    Mo = new G4Element("Molybdenum", "Mo", 7);
    Mo->AddIsotope(Mo92, 14.77*perCent);
    Mo->AddIsotope(Mo94, 9.23*perCent);
    Mo->AddIsotope(Mo95, 15.9*perCent);
    Mo->AddIsotope(Mo96, 16.68*perCent);
    Mo->AddIsotope(Mo97,  9.56*perCent);
    Mo->AddIsotope(Mo98, 24.19*perCent);
    Mo->AddIsotope(Mo100,  9.67*perCent);

    Sn112 = new G4Isotope("Sn112", 50, 112, 111.9048*g/mole);
    Sn114 = new G4Isotope("Sn114", 50, 114, 113.9028*g/mole);
    Sn115 = new G4Isotope("Sn115", 50, 115, 114.9033*g/mole);
    Sn116 = new G4Isotope("Sn116", 50, 116, 115.9017*g/mole);
    Sn117 = new G4Isotope("Sn117", 50, 117, 116.9030*g/mole);
    Sn118 = new G4Isotope("Sn118", 50, 118, 117.9016*g/mole);
    Sn119 = new G4Isotope("Sn119", 50, 119, 118.9033*g/mole);
    Sn120 = new G4Isotope("Sn120", 50, 120, 119.9022*g/mole);
    Sn122 = new G4Isotope("Sn122", 50, 122, 121.9034*g/mole);
    Sn124 = new G4Isotope("Sn124", 50, 124, 123.9053*g/mole);

    Sn = new G4Element("Tin", "Sn", 10);
    Sn->AddIsotope(Sn112, 0.97*perCent);
    Sn->AddIsotope(Sn114, 0.66*perCent);
    Sn->AddIsotope(Sn115, 0.34*perCent);
    Sn->AddIsotope(Sn116, 14.54*perCent);
    Sn->AddIsotope(Sn117, 7.68*perCent);
    Sn->AddIsotope(Sn118, 24.22*perCent);
    Sn->AddIsotope(Sn119, 8.59*perCent);
    Sn->AddIsotope(Sn120, 32.58*perCent);
    Sn->AddIsotope(Sn122, 4.63*perCent);
    Sn->AddIsotope(Sn124, 5.79*perCent);

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

    Th232 = new G4Isotope("Th232", 90, 232, 232.0381*g/mole);

    Th = new G4Element("Thorium", "Th", 1);
    Th->AddIsotope(Th232, 100.*perCent);

    //make Zirconium isotopes and element
    Pu238 = new G4Isotope("Pu238", 94, 238, 238.0496*g/mole);
    Pu239 = new G4Isotope("Pu239", 94, 239, 239.0522*g/mole);
    Pu240 = new G4Isotope("Pu240", 94, 240, 240.0538*g/mole);
    Pu241 = new G4Isotope("Pu241", 94, 241, 241.0569*g/mole);
    Pu242 = new G4Isotope("Pu242", 94, 242, 242.0587*g/mole);

// fixed the isotope compositon so that it is interms of abundance instead of weight percentage
    Pu = new G4Element("Plutonium", "Pu", 5);
    Pu->AddIsotope(Pu238, 2.77*perCent);
    Pu->AddIsotope(Pu239, 52.11*perCent);
    Pu->AddIsotope(Pu240, 22.93*perCent);
    Pu->AddIsotope(Pu241, 15.15*perCent);
    Pu->AddIsotope(Pu242, 7.03*perCent);

    // Create the world material
    World = new G4Material("Galactic", 1, 1, 1.e-25*g/cm3, kStateGas,
						   2.73*kelvin, 3.e-18*pascal);

    // Create H20 material
    H2O = new G4Material("LightWater", 1.*g/cm3, 2, kStateLiquid);
    H2O->AddElement(H,2);
    H2O->AddElement(Oxygen,1);

    // Create D20 material
    D2O = new G4Material("HeavyWater", 1.1*g/cm3, 2, kStateLiquid);
    D2O->AddElement(D,2);
    D2O->AddElement(Oxygen,1);


    // Create Coolant
	Coolant = new G4Material("Coolant", coolantDensity, 1, kStateLiquid,
							 coolantTemp);
    Coolant->AddMaterial(H2O,  100*perCent);

    //Create Moderator
    Moderator = new G4Material("Moderator", moderatorDensity, 2, kStateLiquid,
							   moderatorTemp);
    Moderator->AddMaterial(D2O,99.833*perCent);
    Moderator->AddMaterial(H2O,0.167*perCent);

    //Create Moderator
    ExelLiner = new G4Material("ExelLiner", 1, 4, kStateSolid,
							   1);
    ExelLiner->AddElement(Sn,3.5*perCent);
    ExelLiner->AddElement(Mo,0.8*perCent);
    ExelLiner->AddElement(Nb,0.8*perCent);
    ExelLiner->AddElement(Zr,94.9*perCent);

    ZircSteel = new G4Material("ZircSteel", 1, 10, kStateSolid, 1);
    ZircSteel->AddElement(C,0.034*perCent);
    ZircSteel->AddElement(Si,0.51*perCent);
    ZircSteel->AddElement(Mn,0.74*perCent);
    ZircSteel->AddElement(P,0.016*perCent);
    ZircSteel->AddElement(S,0.002*perCent);
    ZircSteel->AddElement(Ni,20.82*perCent);
    ZircSteel->AddElement(Cr,25.04*perCent);
    ZircSteel->AddElement(Fe,51.738*perCent);
    ZircSteel->AddElement(Mo,0.51*perCent);
    ZircSteel->AddElement(Zr,0.59*perCent);

    ZircHydrid = new G4Material("ZircHydrid", 1, 2, kStateSolid, 1);
    ZircHydrid->AddElement(Zr,98.26*perCent);
    ZircHydrid->AddElement(H,1.74*perCent);

    Insulator = new G4Material("Insulator", 1, 3, kStateSolid, 1);
    Insulator->AddElement(Zr,66.63*perCent);
    Insulator->AddElement(Y,7.87*perCent);
    Insulator->AddElement(Oxygen,25.5*perCent);

    //Create Fuel
    OuterFuel = new G4Material("OuterFuel", 1, 3, kStateSolid, 1);
    OuterFuel->AddElement(Oxygen,12.08*perCent);
    OuterFuel->AddElement(Pu,10.59*perCent);
    OuterFuel->AddElement(Th,77.34*perCent);

    InnerFuel = new G4Material("InnerFuel", 1, 3, kStateSolid, 1);
    InnerFuel->AddElement(Oxygen,12.07*perCent);
    InnerFuel->AddElement(Pu,13.23*perCent);
    InnerFuel->AddElement(Th,74.7*perCent);


//    InFlowTube = new G4Material("InFlowTube", inFlowTubeDensity, 1, kStateSolid, inFlowTubeTemp);
//    InFlowTube->AddMaterial(ZircSteel,100*perCent);

//    OutFlowTube = new G4Material("OutFlowTube", outFlowTubeDensity, 1, kStateSolid, outFlowTubeTemp);
//    OutFlowTube->AddMaterial(ZircSteel,100*perCent);

    OutSheathe = new G4Material("OutSheathe", outSheatheDensity, 1, kStateSolid, outSheatheTemp);
    OutSheathe->AddMaterial(ZircSteel,100*perCent);

    InSheathe = new G4Material("InSheathe", inSheatheDensity, 1, kStateSolid, inSheatheTemp);
    InSheathe->AddMaterial(ZircSteel,100*perCent);

    Liner = new G4Material("Liner", linerDensity, 1, kStateSolid, linerTemp);
    Liner->AddMaterial(ZircSteel,100*perCent);



//    FlowTube1 = new G4Material("FlowTube1", flowTubeDensity, 1, kStateSolid, flowTubeTemp[0]);
//    FlowTube1->AddMaterial(ZircSteel,100*perCent);
//
//    FlowTube2 = new G4Material("FlowTube2", flowTubeDensity, 1, kStateSolid, flowTubeTemp[1]);
//    FlowTube2->AddMaterial(ZircSteel,100*perCent);
//
    FlowTube = new G4Material("FlowTube", flowTubeDensity, 1, kStateSolid, flowTubeTemp);
    FlowTube->AddMaterial(ZircSteel,100*perCent);




    PressTube1 = new G4Material("PressTube1", pressTubeDensity, 1, kStateSolid, pressTubeTemp[0]);
    PressTube1->AddMaterial(ExelLiner,100*perCent);

    PressTube2 = new G4Material("PressTube2", pressTubeDensity, 1, kStateSolid, pressTubeTemp[1]);
    PressTube2->AddMaterial(ExelLiner,100*perCent);

    PressTube3 = new G4Material("PressTube3", pressTubeDensity, 1, kStateSolid, pressTubeTemp[2]);
    PressTube3->AddMaterial(ExelLiner,100*perCent);

    PressTube4 = new G4Material("PressTube4", pressTubeDensity, 1, kStateSolid, pressTubeTemp[3]);
    PressTube4->AddMaterial(ExelLiner,100*perCent);

    OutLiner = new G4Material("OutLiner", outLinerDensity, 1, kStateSolid, outLinerTemp);
    OutLiner->AddMaterial(ExelLiner,100*perCent);



    Insulator1 = new G4Material("Insulator1", insulatorDensity, 1, kStateSolid, insulatorTemp[0]);
    Insulator1->AddMaterial(Insulator,100*perCent);

    Insulator2 = new G4Material("Insulator2", insulatorDensity, 1, kStateSolid, insulatorTemp[1]);
    Insulator2->AddMaterial(Insulator,100*perCent);

    Insulator3 = new G4Material("Insulator3", insulatorDensity, 1, kStateSolid, insulatorTemp[2]);
    Insulator3->AddMaterial(Insulator,100*perCent);

    Insulator4 = new G4Material("Insulator4", insulatorDensity, 1, kStateSolid, insulatorTemp[3]);
    Insulator4->AddMaterial(Insulator,100*perCent);


    InnerFuel1 = new G4Material("InnerFuel1", innerFuelDensity, 1, kStateSolid, innerFuelTemp[0]);
    InnerFuel1->AddMaterial(InnerFuel,100*perCent);

    InnerFuel2 = new G4Material("InnerFuel2", innerFuelDensity, 1, kStateSolid, innerFuelTemp[1]);
    InnerFuel2->AddMaterial(InnerFuel,100*perCent);

    InnerFuel3 = new G4Material("InnerFuel3", innerFuelDensity, 1, kStateSolid, innerFuelTemp[2]);
    InnerFuel3->AddMaterial(InnerFuel,100*perCent);

    InnerFuel4 = new G4Material("InnerFuel4", innerFuelDensity, 1, kStateSolid, innerFuelTemp[3]);
    InnerFuel4->AddMaterial(InnerFuel,100*perCent);



    OuterFuel1 = new G4Material("OuterFuel1", outerFuelDensity, 1, kStateSolid, outerFuelTemp[0]);
    OuterFuel1->AddMaterial(OuterFuel,100*perCent);

    OuterFuel2 = new G4Material("OuterFuel2", outerFuelDensity, 1, kStateSolid, outerFuelTemp[1]);
    OuterFuel2->AddMaterial(OuterFuel,100*perCent);

    OuterFuel3 = new G4Material("OuterFuel3", outerFuelDensity, 1, kStateSolid, outerFuelTemp[2]);
    OuterFuel3->AddMaterial(OuterFuel,100*perCent);

    OuterFuel4 = new G4Material("OuterFuel4", outerFuelDensity, 1, kStateSolid, outerFuelTemp[3]);
    OuterFuel4->AddMaterial(OuterFuel,100*perCent);

    CentralCoolant = new G4Material("CentralCoolant", centralCoolantDensity, 1, kStateLiquid, centralCoolantTemp);
    CentralCoolant->AddMaterial(Coolant,  100*perCent);


    // Add materials to the map indexed by either ZA (format ZZAAA or ZZ)
    // For composite materials:  world is 0, heavy water is 1, UHW is 2
    matMap["Galactic"] = World;
    matMap["Moderator"] = Moderator;
    matMap["PressTube1"] = PressTube1;
    matMap["PressTube2"] = PressTube2;
    matMap["PressTube3"] = PressTube3;
    matMap["PressTube4"] = PressTube4;
    matMap["OutLiner"] = OutLiner;
    matMap["Insulator1"] = Insulator1;
    matMap["Insulator2"] = Insulator2;
    matMap["Insulator3"] = Insulator3;
    matMap["Insulator4"] = Insulator4;
    matMap["Liner"] = Liner;
    matMap["Coolant"] = Coolant;
    matMap["OutSheathe"] = OutSheathe;
    matMap["InSheathe"] = InSheathe;
    matMap["OuterFuel1"] = OuterFuel1;
    matMap["OuterFuel2"] = OuterFuel2;
    matMap["OuterFuel3"] = OuterFuel3;
    matMap["OuterFuel4"] = OuterFuel4;
    matMap["InnerFuel1"] = InnerFuel1;
    matMap["InnerFuel2"] = InnerFuel2;
    matMap["InnerFuel3"] = InnerFuel3;
    matMap["InnerFuel4"] = InnerFuel4;
//    matMap["OutFlowTube"] = OutFlowTube;
//    matMap["FlowTube1"] = FlowTube1;
//    matMap["FlowTube2"] = FlowTube2;
    matMap["FlowTube"] = FlowTube;
//    matMap["InFlowTube"] = InFlowTube;
    matMap["CentralCoolant"] = CentralCoolant;

    matChanged = false;

    return;
}
