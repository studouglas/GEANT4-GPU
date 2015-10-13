/*
SCWRDopplerConstructor.cc

Created by:		Wesley Ford
Date:			23-05-2012
Modified:       11-03-2013

Source code for the CANDU 6 lattice geometry and materials

*/

#include "SCWRDopplerConstructor.hh"

// Constructor
SCWRDopplerConstructor::SCWRDopplerConstructor()
: StorkVWorldConstructor(), cellLogical(0), pressTubeLogical(0), outLinerLogical(0), insulatorLogical(0),
  linerLogical(0), coolantLogical(0), outSheatheLogical(0), outSheatheLogicalH1(0), outSheatheLogicalH2(0),
  inSheatheLogical(0), inSheatheLogicalH1(0), inSheatheLogicalH2(0), outFuelLogical(0), outFuelLogicalH1(0),
  outFuelLogicalH2(0), inFuelLogical(0), inFuelLogicalH1(0), inFuelLogicalH2(0), centralCoolantLogical(0)
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

    latticePitch= 25.*cm;

    /*
    moderatorTemp=342.00*kelvin;
    moderatorDensity=1.0851*g/cm3;

    pressTubeTemp=416.74*kelvin;
    pressTubeDensity=6.52*g/cm3;

    outLinerTemp=470.5200*kelvin;
    outLinerDensity=6.52*g/cm3;

    insulatorTemp=557.17*kelvin;
    insulatorDensity=5.83*g/cm3;

    linerTemp=671.8*kelvin;
    linerDensity=7.9*g/cm3;

    coolantTemp=681.79*kelvin;
    //coolantDensity=0.14933*g/cm3;
    coolantDensity=0.001*g/cm3;

    inSheatheTemp=756.30*kelvin;
    inSheatheDensity=7.9*g/cm3;

    outSheatheTemp=756.30*kelvin;
    outSheatheDensity=7.9*g/cm3;

    innerFuelTemp=1420.62*kelvin;
    innerFuelDensity=9.91*g/cm3;

    outerFuelTemp=1420.62*kelvin;
    outerFuelDensity=9.87*g/cm3;

    flowTubeTemp=657.79*kelvin;
    flowTubeDensity=7.9*g/cm3;

    centralCoolantTemp=633.79*kelvin;
    //centralCoolantDensity=0.58756*g/cm3;
    centralCoolantDensity=0.001*g/cm3;
    */

    // update temperature and density data at h=1.75cm
    moderatorTemp=300.00*kelvin;
    moderatorDensity=1.0851*g/cm3;

    pressTubeTemp=414.12*kelvin;
    pressTubeDensity=6.52*g/cm3;

    outLinerTemp=658.95*kelvin;
    outLinerDensity=6.52*g/cm3;

    insulatorTemp=549.11*kelvin;
    insulatorDensity=5.83*g/cm3;

    linerTemp=658.95*kelvin;
    linerDensity=7.9*g/cm3;

    coolantTemp=666.64*kelvin;
    //coolantDensity=0.21395*g/cm3;
    coolantDensity=0.001*g/cm3;

    inSheatheTemp=706.93*kelvin;
    inSheatheDensity=7.9*g/cm3;

    outSheatheTemp=706.93*kelvin;
    outSheatheDensity=7.9*g/cm3;

    innerFuelTemp=1410.09*kelvin;
    innerFuelDensity=9.91*g/cm3;

    outerFuelTemp=1410.09*kelvin;
    outerFuelDensity=9.87*g/cm3;

    flowTubeTemp=658.95*kelvin;
    flowTubeDensity=7.9*g/cm3;

    centralCoolantTemp=635.32*kelvin;
    centralCoolantDensity=0.58374*g/cm3;
    //centralCoolantDensity=0.001*g/cm3;
}


// Desturctor
SCWRDopplerConstructor::~SCWRDopplerConstructor()
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
//    delete outFlowTubeVisAtt;
    if(flowTubeVisAtt)
        delete flowTubeVisAtt;
//    delete inFlowTubeVisAtt;
    if(centralCoolantVisAtt)
        delete centralCoolantVisAtt;
}


//Checked geometry dimensions against Jasons

// ConstructWorld()
// Construct the geometry and materials of the CANDU 6 lattice cell.
G4VPhysicalVolume* SCWRDopplerConstructor::ConstructWorld()
{
	// Lattic cell dimensions
	G4double buffer = 1.0*cm;
	reactorDim = G4ThreeVector(latticePitch/4.0,latticePitch/4.0,latticePitch/2.0);

	encWorldDim = 2.0 * G4ThreeVector(reactorDim[0]+buffer,reactorDim[1]+buffer,
									  reactorDim[2]+buffer);

	G4SolidStore* theSolids = G4SolidStore::GetInstance();


	// Set static dimensions

    // Pressure Tube dimension
    G4double pressTubeRadmax = 9.05*cm;
    G4double pressTubeLen = reactorDim[2];

    // outer liner dimensions
    G4double outLinerRadmax = 7.85*cm;
    G4double outLinerLen = reactorDim[2];

    // insulator dimensions
    G4double insulatorRadmax = 7.8*cm;
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
    G4double outFuelRadmax = 0.44*cm;
    G4double outFuelLen = reactorDim[2];

    // inner fuel dimensions
    G4double inFuelRadmax = 0.415*cm;
    G4double inFuelLen = reactorDim[2];

    G4double flowTubeRadmax = 4.7*cm;
    G4double flowTubeLen = reactorDim[2];

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
		new G4Box("worldBox", encWorldDim[0]/2, encWorldDim[1]/2,
				  encWorldDim[2]/2);

		// Create the lattice cell solid
		new G4Box("cellBox", reactorDim[0], reactorDim[1], reactorDim[2]);

        // Create Pressure Tube
        new G4Tubs("pressTube", 0., pressTubeRadmax, pressTubeLen, 0.,
					0.5*CLHEP::pi);

		// Create outer liner tube
		new G4Tubs("outLinerTube", 0., outLinerRadmax, outLinerLen, 0., 0.5*CLHEP::pi);

		// Create insulator
        new G4Tubs("insulatorTube", 0., insulatorRadmax, insulatorLen, 0.,
					0.5*CLHEP::pi);

		// Create liner tube
		new G4Tubs("linerTube", 0., linerRadmax, linerLen, 0.,
					0.5*CLHEP::pi);

		// Create coolant solid
		new G4Tubs("coolantTube", 0., coolantRadmax, coolantLen, 0.,
					0.5*CLHEP::pi);

		// Create the sheathe for fuel pins
		new G4Tubs("outSheatheTube", 0., outSheatheRadmax, outSheatheLen, 0.,2.0*CLHEP::pi);

        new G4Tubs("outSheatheTubeH1", 0., outSheatheRadmax, outSheatheLen, 0.,1.0*CLHEP::pi);

        new G4Tubs("outSheatheTubeH2", 0., outSheatheRadmax, outSheatheLen, -0.5*CLHEP::pi, 1.0*CLHEP::pi);

        // Create the sheathe for fuel pins
		new G4Tubs("inSheatheTube", 0., inSheatheRadmax, inSheatheLen, 0.,2.0*CLHEP::pi);

        new G4Tubs("inSheatheTubeH1", 0., inSheatheRadmax, inSheatheLen, 0.,1.0*CLHEP::pi);

        new G4Tubs("inSheatheTubeH2", 0., inSheatheRadmax, inSheatheLen, -0.5*CLHEP::pi, 1.0*CLHEP::pi);

		// Create outer fuel pins
		new G4Tubs("outFuelCyl", 0., outFuelRadmax, outFuelLen, 0.,2.0*CLHEP::pi);

		new G4Tubs("outFuelCylH1", 0., outFuelRadmax, outFuelLen, 0.,1.0*CLHEP::pi);

		new G4Tubs("outFuelCylH2", 0., outFuelRadmax, outFuelLen, -0.5*CLHEP::pi, 1.0*CLHEP::pi);

		// Create inner fuel pins
		new G4Tubs("inFuelCyl", 0., inFuelRadmax, inFuelLen, 0., 2.0*CLHEP::pi);

		new G4Tubs("inFuelCylH1", 0., inFuelRadmax, inFuelLen, 0., 1.0*CLHEP::pi);

		new G4Tubs("inFuelCylH2", 0., inFuelRadmax, inFuelLen, -0.5*CLHEP::pi, 1.0*CLHEP::pi);

        new G4Tubs("flowTube", 0., flowTubeRadmax, flowTubeLen, 0., 0.5*CLHEP::pi);

		// Create the central coolant
		new G4Tubs("centralCoolantCyl", 0., centralCoolantRadmax, centralCoolantLen, 0., 0.5*CLHEP::pi);

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
    pressTubeLogical = new G4LogicalVolume(theSolids->GetSolid("pressTube"),
                        matMap["PressTube"], "pressTubeLogical",0,0,0);


    new G4PVPlacement(0, G4ThreeVector(-reactorDim[0],-reactorDim[1],0.), pressTubeLogical,
					  "pressTubePhysical",cellLogical,0,0);

    // Create the Pressure Tube
    outLinerLogical = new G4LogicalVolume(theSolids->GetSolid("outLinerTube"),
                        matMap["OutLiner"], "outLinerLogical",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), outLinerLogical,
					  "outLinerPhysical",pressTubeLogical,0,0);

    // Create the insulator
    insulatorLogical = new G4LogicalVolume(theSolids->GetSolid("insulatorTube"),
                        matMap["Insulator"], "insulatorLogical",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), insulatorLogical,
					  "insulatorPhysical",outLinerLogical,0,0);

	// Create the Pressure Tube
    linerLogical = new G4LogicalVolume(theSolids->GetSolid("linerTube"),
                        matMap["Liner"], "linerLogical",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), linerLogical,
					  "linerPhysical",insulatorLogical,0,0);

    // Create the Coolant
    coolantLogical = new G4LogicalVolume(theSolids->GetSolid("coolantTube"),
                        matMap["Coolant"], "coolantLogical",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), coolantLogical,
					  "coolantPhysical",linerLogical,0,0);

    // Create the Sheathe
    outSheatheLogical = new G4LogicalVolume(theSolids->GetSolid("outSheatheTube"),
                        matMap["OutSheathe"], "outSheatheLogical",0,0,0);

    outSheatheLogicalH1 = new G4LogicalVolume(theSolids->GetSolid("outSheatheTubeH1"),
                        matMap["OutSheathe"], "outSheatheLogicalH1",0,0,0);

    outSheatheLogicalH2 = new G4LogicalVolume(theSolids->GetSolid("outSheatheTubeH2"),
                        matMap["OutSheathe"], "outSheatheLogicalH2",0,0,0);


    inSheatheLogical = new G4LogicalVolume(theSolids->GetSolid("inSheatheTube"),
                        matMap["InSheathe"], "inSheatheLogical",0,0,0);

    inSheatheLogicalH1 = new G4LogicalVolume(theSolids->GetSolid("inSheatheTubeH1"),
                        matMap["InSheathe"], "inSheatheLogicalH1",0,0,0);

    inSheatheLogicalH2 = new G4LogicalVolume(theSolids->GetSolid("inSheatheTubeH2"),
                        matMap["InSheathe"], "inSheatheLogicalH2",0,0,0);

    // Create a fuel
    outFuelLogical = new G4LogicalVolume(theSolids->GetSolid("outFuelCyl"),
                        matMap["OuterFuel"], "outFuelLogical",0,0,0);

    outFuelLogicalH1 = new G4LogicalVolume(theSolids->GetSolid("outFuelCylH1"),
                        matMap["OuterFuel"], "outFuelLogicalH1",0,0,0);

    outFuelLogicalH2 = new G4LogicalVolume(theSolids->GetSolid("outFuelCylH2"),
                        matMap["OuterFuel"], "outFuelLogicalH2",0,0,0);

    // Create a fuel
    inFuelLogical = new G4LogicalVolume(theSolids->GetSolid("inFuelCyl"),
                        matMap["InnerFuel"], "inFuelLogical",0,0,0);

    inFuelLogicalH1 = new G4LogicalVolume(theSolids->GetSolid("inFuelCylH1"),
                        matMap["InnerFuel"], "inFuelLogicalH1",0,0,0);

    inFuelLogicalH2 = new G4LogicalVolume(theSolids->GetSolid("inFuelCylH2"),
                        matMap["InnerFuel"], "inFuelLogicalH2",0,0,0);


    flowTubeLogical = new G4LogicalVolume(theSolids->GetSolid("flowTube"),
                        matMap["FlowTube"], "flowTubeLogical",0,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), flowTubeLogical,
					  "flowTubePhysical",coolantLogical,0,0);

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

    new G4PVPlacement(0, G4ThreeVector(ringRad[0],0.,0.), outSheatheLogicalH1, "outSheathePhysicalH1",coolantLogical,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), outFuelLogicalH1,"outFuelPhysicalH1",
					  outSheatheLogicalH1,0,0);

    // Place pins for outer ring
    for( G4int i = 1; i < 8; i++ )
    {
        // Reset string stream
        volName.str("");

        volName << i;

             G4ThreeVector Tm(ringRad[0]*cos(2.0*i*CLHEP::pi/32),
                              ringRad[0]*sin(2.0*i*CLHEP::pi/32), 0.);

            new G4PVPlacement(0, Tm, outSheatheLogical,"outSheathePhysical " +
                                volName.str(),coolantLogical,0,0);

    }

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), outFuelLogical,"outFuelPhysical",
					  outSheatheLogical,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,ringRad[0],0.), outSheatheLogicalH2, "outSheathePhysicalH2",coolantLogical,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), outFuelLogicalH2,"outFuelPhysicalH2",
					  outSheatheLogicalH2,0,0);


    new G4PVPlacement(0, G4ThreeVector(ringRad[1],0.,0.), inSheatheLogicalH1, "inSheathePhysicalH1",coolantLogical,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), inFuelLogicalH1,"inFuelPhysicalH1",
					  inSheatheLogicalH1,0,0);

	// Place pins for inner ring
    for( G4int i = 1; i < 8; i++ )
    {
        // Reset string stream
        volName.str("");

        volName << i;

             G4ThreeVector Tm(ringRad[1]*cos(2.0*i*CLHEP::pi/32),
                              ringRad[1]*sin(2.0*i*CLHEP::pi/32), 0.);

            new G4PVPlacement(0, Tm, inSheatheLogical,"inSheathePhysical " +
                                volName.str(),coolantLogical,0,0);

    }

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), inFuelLogical,"inFuelPhysical",
					  inSheatheLogical,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,ringRad[1],0.), inSheatheLogicalH2, "inSheathePhysicalH2",coolantLogical,0,0);

    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), inFuelLogicalH2,"inFuelPhysicalH2",
					  inSheatheLogicalH2,0,0);


	// Add sensitive detector to ALL logical volumes
	worldLogical->SetSensitiveDetector( sDReactor );
	cellLogical->SetSensitiveDetector( sDReactor );
	pressTubeLogical->SetSensitiveDetector( sDReactor );
	outLinerLogical->SetSensitiveDetector( sDReactor );
	insulatorLogical->SetSensitiveDetector( sDReactor );
	linerLogical->SetSensitiveDetector( sDReactor );
	coolantLogical->SetSensitiveDetector( sDReactor );
	outSheatheLogical->SetSensitiveDetector( sDReactor );
	outSheatheLogicalH1->SetSensitiveDetector( sDReactor );
	outSheatheLogicalH2->SetSensitiveDetector( sDReactor );
	inSheatheLogical->SetSensitiveDetector( sDReactor );
	inSheatheLogicalH1->SetSensitiveDetector( sDReactor );
	inSheatheLogicalH2->SetSensitiveDetector( sDReactor );
	outFuelLogical->SetSensitiveDetector( sDReactor );
	outFuelLogicalH1->SetSensitiveDetector( sDReactor );
	outFuelLogicalH2->SetSensitiveDetector( sDReactor );
	inFuelLogical->SetSensitiveDetector( sDReactor );
	inFuelLogicalH1->SetSensitiveDetector( sDReactor );
	inFuelLogicalH2->SetSensitiveDetector( sDReactor );
	flowTubeLogical->SetSensitiveDetector( sDReactor );
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
//    delete outFlowTubeVisAtt;
    if(flowTubeVisAtt)
        delete flowTubeVisAtt;
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
    pressTubeLogical->SetVisAttributes(pressTubeVisAtt);

    outLinerVisAtt = new G4VisAttributes(G4Colour(1,0.5,0));
    outLinerVisAtt->SetVisibility(1);
    outLinerLogical->SetVisAttributes(outLinerVisAtt);

    insulatorVisAtt = new G4VisAttributes(G4Colour(1,1,0.5));
    insulatorVisAtt->SetVisibility(1);
    insulatorLogical->SetVisAttributes(insulatorVisAtt);

    linerVisAtt = new G4VisAttributes(G4Colour(0,1,0));
    linerVisAtt->SetVisibility(1);
    linerLogical->SetVisAttributes(linerVisAtt);

    coolantVisAtt = new G4VisAttributes(G4Colour(0,1,1));
    coolantVisAtt->SetVisibility(1);
    coolantLogical->SetVisAttributes(coolantVisAtt);

    outSheatheVisAtt = new G4VisAttributes(G4Colour(0.5,0,1));
    outSheatheVisAtt->SetVisibility(true);
    outSheatheLogical->SetVisAttributes(outSheatheVisAtt);
    outSheatheLogicalH1->SetVisAttributes(outSheatheVisAtt);
    outSheatheLogicalH2->SetVisAttributes(outSheatheVisAtt);

    inSheatheVisAtt = new G4VisAttributes(G4Colour(1,0,1));
    inSheatheVisAtt->SetVisibility(true);
    inSheatheLogical->SetVisAttributes(inSheatheVisAtt);
    inSheatheLogicalH1->SetVisAttributes(inSheatheVisAtt);
    inSheatheLogicalH2->SetVisAttributes(inSheatheVisAtt);

    outFuelVisAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    outFuelVisAtt->SetVisibility(true);
    outFuelLogical->SetVisAttributes(outFuelVisAtt);
    outFuelLogicalH1->SetVisAttributes(outFuelVisAtt);
    outFuelLogicalH2->SetVisAttributes(outFuelVisAtt);

    inFuelVisAtt = new G4VisAttributes(G4Colour(1.,0.5,0.));
    inFuelVisAtt->SetVisibility(true);
    inFuelLogical->SetVisAttributes(inFuelVisAtt);
    inFuelLogicalH1->SetVisAttributes(inFuelVisAtt);
    inFuelLogicalH2->SetVisAttributes(inFuelVisAtt);

    flowTubeVisAtt = new G4VisAttributes(G4Colour(0,1,0));
    flowTubeVisAtt->SetVisibility(1);
    flowTubeLogical->SetVisAttributes(flowTubeVisAtt);

    centralCoolantVisAtt = new G4VisAttributes(G4Colour(0.,1.,1.));
    centralCoolantVisAtt->SetVisibility(1);
    centralCoolantLogical->SetVisAttributes(centralCoolantVisAtt);



    return worldPhysical;
}


// ConstructMaterials()
// Define and build the materials in the C6 lattice cell.
void SCWRDopplerConstructor::ConstructMaterials()
{
    // Elements, isotopes and materials
    G4Isotope *H1, *H2, *C12, *C13, *O16, *Si28, *Si29, *Si30, *P31, *S32, *S33, *S34, *S36,
              *Cr50, *Cr52, *Cr53, *Cr54, *Mn55, *Fe54, *Fe56, *Fe57, *Fe58, *Ni58, *Ni60, *Ni61,
              *Ni62, *Ni64, *Nb93, *Y89, *Mo92, *Mo94, *Mo95, *Mo96, *Mo97, *Mo98, *Mo100,
              *Sn112, *Sn114, *Sn115, *Sn116, *Sn117, *Sn118, *Sn119, *Sn120, *Sn122, *Sn124,
              *Zr90, *Zr91, *Zr92, *Zr94, *Zr96, *Th232, *Pu238, *Pu239, *Pu240, *Pu241, *Pu242;

    StorkElement *H, *D, *C, *Oxygen, *Si, *P, *S, *Cr, *Mn, *Fe, *Ni, *Nb, *Y, *Mo, *Sn, *Zr, *Th, *Pu;

    StorkMaterial *World, *Moderator, *PressTube, *OutLiner, *Insulator, *Liner, *OutSheathe, *InSheathe,
    *OuterFuel, *InnerFuel, *FlowTube, *CentralCoolant ,*ExelLiner,
    *ZircSteel, /*ZircHydrid,*/ *H2O, *D2O, *Coolant;

    H1 = new G4Isotope("H1", 1, 1, 1.008*g/mole);
    H = new StorkElement("Hydrogen", "H", 1);
    H->AddIsotope(H1, 1.0);

    H2 = new G4Isotope("H2", 1, 2, 2.014*g/mole);
    D = new StorkElement("Deterium", "D", 1);
    D->AddIsotope(H2, 1.0);

    C12 = new G4Isotope("C12", 6, 12, 12.0*g/mole);
    C13 = new G4Isotope("C13", 6, 13, 13.0033548378*g/mole);

    C = new StorkElement("Carbon", "C", 2);
    C->AddIsotope(C12, 98.93*perCent);
    C->AddIsotope(C13,  1.07*perCent);


    // Make oxygen isotope and element
    O16 = new G4Isotope("O16", 8, 16, 15.995*g/mole);

    Oxygen = new StorkElement("Oxygen", "O", 1);
    Oxygen->AddIsotope(O16, 100.*perCent);

    Si28 = new G4Isotope("Si28", 14, 28, 27.9769*g/mole);
    Si29 = new G4Isotope("Si29", 14, 29, 28.9765*g/mole);
    Si30 = new G4Isotope("Si30", 14, 28, 29.9738*g/mole);

    Si = new StorkElement("Silicon", "Si", 3);
    Si->AddIsotope(Si28, 92.223*perCent);
    Si->AddIsotope(Si29,  4.685*perCent);
    Si->AddIsotope(Si30,  3.092*perCent);

    // Make oxygen isotope and element
    P31 = new G4Isotope("P31", 15, 31, 30.97376*g/mole);

    P = new StorkElement("Phosphorus", "P", 1);
    P->AddIsotope(P31, 100.*perCent);

    S32 = new G4Isotope("S32", 16, 32, 31.9721*g/mole);
    S33 = new G4Isotope("S33", 16, 33, 32.9715*g/mole);
    S34 = new G4Isotope("S34", 16, 34, 33.9679*g/mole);
    S36 = new G4Isotope("S36", 16, 36, 35.9679*g/mole);

    S = new StorkElement("Sulphur", "S", 4);
    S->AddIsotope(S32, 94.93*perCent);
    S->AddIsotope(S33,  0.76*perCent);
    S->AddIsotope(S34,  4.29*perCent);
    S->AddIsotope(S36,  0.02*perCent);

    //make chromium isotopes and element
    Cr50 = new G4Isotope("Cr50", 24, 50, 49.9460422*g/mole);
    Cr52 = new G4Isotope("Cr52", 24, 52, 51.9405075*g/mole);
    Cr53 = new G4Isotope("Cr53", 24, 53, 52.9406494*g/mole);
    Cr54 = new G4Isotope("Cr54", 24, 54, 53.9388804*g/mole);

    Cr = new StorkElement("Chromium", "Cr", 4);
    Cr->AddIsotope(Cr50,  4.345*perCent);
    Cr->AddIsotope(Cr52, 83.789*perCent);
    Cr->AddIsotope(Cr53,  9.501*perCent);
    Cr->AddIsotope(Cr54,  2.365*perCent);

    //make chromium isotopes and element
    Mn55 = new G4Isotope("Mn55", 25, 55, 54.9380*g/mole);

    Mn = new StorkElement("Manganese", "Mn", 1);
    Mn->AddIsotope(Mn55,  100.*perCent);

    //make iron isotopes and element
    Fe54 = new G4Isotope("Fe54", 26, 54, 53.9396105*g/mole);
    Fe56 = new G4Isotope("Fe56", 26, 56, 55.9349375*g/mole);
    Fe57 = new G4Isotope("Fe57", 26, 57, 56.9353940*g/mole);
    Fe58 = new G4Isotope("Fe58", 26, 58, 57.9332756*g/mole);

    Fe = new StorkElement("Iron", "Fe", 4);
    Fe->AddIsotope(Fe54,  5.845*perCent);
    Fe->AddIsotope(Fe56, 91.754*perCent);
    Fe->AddIsotope(Fe57,  2.119*perCent);
    Fe->AddIsotope(Fe58,  0.282*perCent);

    //make nickel isotopes and element
    Ni58 = new G4Isotope("Ni58", 28, 58, 57.9353429*g/mole);
    Ni60 = new G4Isotope("Ni60", 28, 60, 59.9307864*g/mole);
    Ni61 = new G4Isotope("Ni61", 28, 61, 60.9310560*g/mole);
    Ni62 = new G4Isotope("Ni62", 28, 62, 61.9283451*g/mole);
    Ni64 = new G4Isotope("Ni64", 28, 64, 63.9279660*g/mole);

    Ni = new StorkElement("Nickel", "Ni", 5);
    Ni->AddIsotope(Ni58, 68.0769*perCent);
    Ni->AddIsotope(Ni60, 26.2231*perCent);
    Ni->AddIsotope(Ni61, 1.1399*perCent);
    Ni->AddIsotope(Ni62, 3.6345*perCent);
    Ni->AddIsotope(Ni64,  0.9256*perCent);

    //make niobium isotopes and element
    Nb93 = new G4Isotope("Nb93", 41, 93, 92.9063781*g/mole);

    Nb = new StorkElement("Niobium", "Nb", 1);
    Nb->AddIsotope(Nb93, 100*perCent);

    Y89 = new G4Isotope("Y89", 39, 89, 99.9058*g/mole);

    Y = new StorkElement("Yttrium", "Y", 1);
    Y->AddIsotope(Y89,  100.*perCent);

        //make Zirconium isotopes and element
    Mo92 = new G4Isotope("Mo92", 42, 92, 91.9068*g/mole);
    Mo94 = new G4Isotope("Mo94", 42, 94, 93.9051*g/mole);
    Mo95 = new G4Isotope("Mo95", 42, 95, 94.9058*g/mole);
    Mo96 = new G4Isotope("Mo96", 42, 96, 95.9047*g/mole);
    Mo97 = new G4Isotope("Mo97", 42, 97, 96.9060*g/mole);
    Mo98 = new G4Isotope("Mo98", 42, 98, 97.9054*g/mole);
    Mo100 = new G4Isotope("Mo100", 42, 100, 99.9075*g/mole);

    Mo = new StorkElement("Molybdenum", "Mo", 7);
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

    Sn = new StorkElement("Tin", "Sn", 10);
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

    Zr = new StorkElement("Zirconium", "Zr", 5);
    Zr->AddIsotope(Zr90, 51.45*perCent);
    Zr->AddIsotope(Zr91, 11.22*perCent);
    Zr->AddIsotope(Zr92, 17.15*perCent);
    Zr->AddIsotope(Zr94, 17.38*perCent);
    Zr->AddIsotope(Zr96,  2.80*perCent);

    Th232 = new G4Isotope("Th232", 90, 232, 232.0381*g/mole);

    Th = new StorkElement("Thorium", "Th", 1);
    Th->AddIsotope(Th232, 100.*perCent);

    //make Plutonium isotopes and element
    Pu238 = new G4Isotope("Pu238", 94, 238, 238.0496*g/mole);
    Pu239 = new G4Isotope("Pu239", 94, 239, 239.0522*g/mole);
    Pu240 = new G4Isotope("Pu240", 94, 240, 240.0538*g/mole);
    Pu241 = new G4Isotope("Pu241", 94, 241, 241.0569*g/mole);
    Pu242 = new G4Isotope("Pu242", 94, 242, 242.0587*g/mole);

// fixed the isotope compositon so that it is interms of abundance instead of weight percentage
    Pu = new StorkElement("Plutonium", "Pu", 5);
    Pu->AddIsotope(Pu238, 2.77*perCent);
    Pu->AddIsotope(Pu239, 52.11*perCent);
    Pu->AddIsotope(Pu240, 22.93*perCent);
    Pu->AddIsotope(Pu241, 15.15*perCent);
    Pu->AddIsotope(Pu242, 7.03*perCent);
/*
    G4ElementTable *tempTab = Pu->GetElementTable();
    G4IsotopeVector *tempVec;
    G4double *tempVecAbun;
    for(int i=0; i<tempTab->size(); i++)
    {
        G4cout << "Element :" << ((*tempTab)[i])->GetName() << " contains " << ((*tempTab)[i])->GetNumberOfIsotopes() << " isotopes" << G4endl;
        tempVec = ((*tempTab)[i])->GetIsotopeVector();
        tempVecAbun = ((*tempTab)[i])->GetRelativeAbundanceVector();
        for(int j=0; j<((*tempTab)[i])->GetNumberOfIsotopes(); j++)
        {
            G4cout << "Isotope :" << ((*tempVec)[j])->GetName() << " has abundance" << tempVecAbun[j] << G4endl;
        }
    }

    G4cout << "\n### Element properties after they have been added to the materials ###\n" << G4endl;
*/
    // Create the world material
    World = new StorkMaterial("Galactic", 1, 1, 1.e-25*g/cm3, kStateGas,
						   2.73*kelvin, 3.e-18*pascal);

    // Create H20 material
    H2O = new StorkMaterial("LightWater", 1.*g/cm3, 2, kStateLiquid, coolantTemp);
    H2O->AddElement(H,2);
    H2O->AddElement(Oxygen,1);

    // Create D20 material
    D2O = new StorkMaterial("HeavyWater", 1.1*g/cm3, 2, kStateLiquid, moderatorTemp);
    D2O->AddElement(D,2);
    D2O->AddElement(Oxygen,1);


    // Create Coolant
	Coolant = new StorkMaterial("Coolant", coolantDensity, 1, kStateLiquid,
							 coolantTemp);
    Coolant->AddMaterial(H2O,  100*perCent);

    //Create Moderator
    Moderator = new StorkMaterial("Moderator", moderatorDensity, 2, kStateLiquid,
							   moderatorTemp);
    Moderator->AddMaterial(D2O,99.833*perCent);
    Moderator->AddMaterial(H2O,0.167*perCent);

    //Create Moderator
    ExelLiner = new StorkMaterial("ExelLiner", 1, 4, kStateSolid, pressTubeTemp);
    ExelLiner->AddElement(Sn,3.5*perCent);
    ExelLiner->AddElement(Mo,0.8*perCent);
    ExelLiner->AddElement(Nb,0.8*perCent);
    ExelLiner->AddElement(Zr,94.9*perCent);

    ZircSteel = new StorkMaterial("ZircSteel", 1, 10, kStateSolid, outSheatheTemp);
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

    /*ZircHydrid = new StorkMaterial("ZircHydrid", 1, 2, kStateSolid, 1);
    ZircHydrid->AddElement(Zr,98.26*perCent);
    ZircHydrid->AddElement(H,1.74*perCent);*/

    Insulator = new StorkMaterial("Insulator", insulatorDensity, 3, kStateSolid, insulatorTemp);
    Insulator->AddElement(Zr,66.63*perCent);
    Insulator->AddElement(Y,7.87*perCent);
    Insulator->AddElement(Oxygen,25.5*perCent);

    //Create Fuel
    OuterFuel = new StorkMaterial("OuterFuel", outerFuelDensity, 3, kStateSolid, outerFuelTemp);
    OuterFuel->AddElement(Oxygen,12.08*perCent);
    OuterFuel->AddElement(Pu,10.59*perCent);
    OuterFuel->AddElement(Th,77.34*perCent);

    InnerFuel = new StorkMaterial("InnerFuel", innerFuelDensity, 3, kStateSolid, innerFuelTemp);
    InnerFuel->AddElement(Oxygen,12.07*perCent);
    InnerFuel->AddElement(Pu,13.23*perCent);
    InnerFuel->AddElement(Th,74.7*perCent);

    OutSheathe = new StorkMaterial("OutSheathe", outSheatheDensity, 1, kStateSolid, outSheatheTemp);
    OutSheathe->AddMaterial(ZircSteel,100*perCent);

    InSheathe = new StorkMaterial("InSheathe", inSheatheDensity, 1, kStateSolid, inSheatheTemp);
    InSheathe->AddMaterial(ZircSteel,100*perCent);

    Liner = new StorkMaterial("Liner", linerDensity, 1, kStateSolid, linerTemp);
    Liner->AddMaterial(ZircSteel,100*perCent);

    FlowTube = new StorkMaterial("FlowTube", flowTubeDensity, 1, kStateSolid, flowTubeTemp);
    FlowTube->AddMaterial(ZircSteel,100*perCent);

    PressTube = new StorkMaterial("PressTube", pressTubeDensity, 1, kStateSolid, pressTubeTemp);
    PressTube->AddMaterial(ExelLiner,100*perCent);

    OutLiner = new StorkMaterial("OutLiner", outLinerDensity, 1, kStateSolid, outLinerTemp);
    OutLiner->AddMaterial(ExelLiner,100*perCent);

    CentralCoolant = new StorkMaterial("CentralCoolant", centralCoolantDensity, 1, kStateLiquid, centralCoolantTemp);
    CentralCoolant->AddMaterial(Coolant,  100*perCent);

//// Adjusted Material Data
//    StorkElement *HMod;
//
//    H1 = new G4Isotope("H1", 1, 1, 1.008*g/mole);
//    H = new StorkElement("Hydrogen", "H", 1);
//    H->AddIsotope(H1, 1.0);
//
//    H2 = new G4Isotope("H2", 1, 2, 2.014*g/mole);
//    D = new StorkElement("Deterium", "D", 1);
//    D->AddIsotope(H2, 1.0);
//
//    HMod = new StorkElement("HMod", "D", 2);
//    HMod->AddIsotope(H1, 0.178808*perCent);
//    HMod->AddIsotope(H2, 99.8212*perCent);
//
//    C12 = new G4Isotope("C12", 6, 12, 12.0*g/mole);
//    C13 = new G4Isotope("C13", 6, 13, 13.0033548378*g/mole);
//
//    C = new StorkElement("Carbon", "C", 2);
//    C->AddIsotope(C12, 98.93*perCent);
//    C->AddIsotope(C13,  1.07*perCent);
//
//
//    // Make oxygen isotope and element
//    O16 = new G4Isotope("O16", 8, 16, 15.995*g/mole);
//
//    Oxygen = new StorkElement("Oxygen", "O", 1);
//    Oxygen->AddIsotope(O16, 100.*perCent);
//
//    Si28 = new G4Isotope("Si28", 14, 28, 27.9769*g/mole);
//    Si29 = new G4Isotope("Si29", 14, 29, 28.9765*g/mole);
//    Si30 = new G4Isotope("Si30", 14, 28, 29.9738*g/mole);
//
//    Si = new StorkElement("Silicon", "Si", 3);
//    Si->AddIsotope(Si28, 92.5003*perCent);
//    Si->AddIsotope(Si29,  4.5605*perCent);
//    Si->AddIsotope(Si30,  2.94*perCent);
//
//    // Make oxygen isotope and element
//    P31 = new G4Isotope("P31", 15, 31, 30.97376*g/mole);
//
//    P = new StorkElement("Phosphorus", "P", 1);
//    P->AddIsotope(P31, 100.*perCent);
//
//    S32 = new G4Isotope("S32", 16, 32, 31.9721*g/mole);
//    /*S33 = new G4Isotope("S33", 16, 33, 32.9715*g/mole);
//    S34 = new G4Isotope("S34", 16, 34, 33.9679*g/mole);
//    S36 = new G4Isotope("S36", 16, 36, 35.9679*g/mole);*/
//
//    S = new StorkElement("Sulphur", "S", 1);
//    S->AddIsotope(S32, 94.93*perCent);
//    /*S->AddIsotope(S33,  0.76*perCent);
//    S->AddIsotope(S34,  4.29*perCent);
//    S->AddIsotope(S36,  0.02*perCent);*/
//
//    //make chromium isotopes and element
//    Cr50 = new G4Isotope("Cr50", 24, 50, 49.9460422*g/mole);
//    Cr52 = new G4Isotope("Cr52", 24, 52, 51.9405075*g/mole);
//    Cr53 = new G4Isotope("Cr53", 24, 53, 52.9406494*g/mole);
//    Cr54 = new G4Isotope("Cr54", 24, 54, 53.9388804*g/mole);
//
//    Cr = new StorkElement("Chromium", "Cr", 4);
//    Cr->AddIsotope(Cr50,  4.52276*perCent);
//    Cr->AddIsotope(Cr52, 83.86853*perCent);
//    Cr->AddIsotope(Cr53,  9.338*perCent);
//    Cr->AddIsotope(Cr54,  2.27872*perCent);
//
//    //make chromium isotopes and element
//    Mn55 = new G4Isotope("Mn55", 25, 55, 54.9380*g/mole);
//
//    Mn = new StorkElement("Manganese", "Mn", 1);
//    Mn->AddIsotope(Mn55,  100.*perCent);
//
//    //make iron isotopes and element
//    Fe54 = new G4Isotope("Fe54", 26, 54, 53.9396105*g/mole);
//    Fe56 = new G4Isotope("Fe56", 26, 56, 55.9349375*g/mole);
//    Fe57 = new G4Isotope("Fe57", 26, 57, 56.9353940*g/mole);
//    Fe58 = new G4Isotope("Fe58", 26, 58, 57.9332756*g/mole);
//
//    Fe = new StorkElement("Iron", "Fe", 4);
//    Fe->AddIsotope(Fe54,  6.05079*perCent);
//    Fe->AddIsotope(Fe56, 91.5996*perCent);
//    Fe->AddIsotope(Fe57,  2.07762*perCent);
//    Fe->AddIsotope(Fe58,  0.271994*perCent);
//
//    //make nickel isotopes and element
//    Ni58 = new G4Isotope("Ni58", 28, 58, 57.9353429*g/mole);
//    Ni60 = new G4Isotope("Ni60", 28, 60, 59.9307864*g/mole);
//    Ni61 = new G4Isotope("Ni61", 28, 61, 60.9310560*g/mole);
//    Ni62 = new G4Isotope("Ni62", 28, 62, 61.9283451*g/mole);
//    Ni64 = new G4Isotope("Ni64", 28, 64, 63.9279660*g/mole);
//
//    Ni = new StorkElement("Nickel", "Ni", 5);
//    Ni->AddIsotope(Ni58, 68.93742*perCent);
//    Ni->AddIsotope(Ni60, 25.6715*perCent);
//    Ni->AddIsotope(Ni61, 1.09601*perCent);
//    Ni->AddIsotope(Ni62, 3.4444*perCent);
//    Ni->AddIsotope(Ni64,  0.850692*perCent);
//
//    //make niobium isotopes and element
//    Nb93 = new G4Isotope("Nb93", 41, 93, 92.9063781*g/mole);
//
//    Nb = new StorkElement("Niobium", "Nb", 1);
//    Nb->AddIsotope(Nb93, 100*perCent);
//
//    Y89 = new G4Isotope("Y89", 39, 89, 99.9058*g/mole);
//
//    Y = new StorkElement("Yttrium", "Y", 1);
//    Y->AddIsotope(Y89,  100.*perCent);
//
//        //make Zirconium isotopes and element
//    Mo92 = new G4Isotope("Mo92", 42, 92, 91.9068*g/mole);
//    Mo94 = new G4Isotope("Mo94", 42, 94, 93.9051*g/mole);
//    Mo95 = new G4Isotope("Mo95", 42, 95, 94.9058*g/mole);
//    Mo96 = new G4Isotope("Mo96", 42, 96, 95.9047*g/mole);
//    Mo97 = new G4Isotope("Mo97", 42, 97, 96.9060*g/mole);
//    Mo98 = new G4Isotope("Mo98", 42, 98, 97.9054*g/mole);
//    Mo100 = new G4Isotope("Mo100", 42, 100, 99.9075*g/mole);
//
//    Mo = new StorkElement("Molybdenum", "Mo", 7);
//    Mo->AddIsotope(Mo92, 9.5471*perCent);
//    Mo->AddIsotope(Mo94, 5.85553*perCent);
//    Mo->AddIsotope(Mo95, 9.98504*perCent);
//    Mo->AddIsotope(Mo96, 62.09203*perCent);
//    Mo->AddIsotope(Mo97,  2.77674*perCent);
//    Mo->AddIsotope(Mo98, 7.05023*perCent);
//    Mo->AddIsotope(Mo100,  2.69333*perCent);
//
//    Sn112 = new G4Isotope("Sn112", 50, 112, 111.9048*g/mole);
//    Sn114 = new G4Isotope("Sn114", 50, 114, 113.9028*g/mole);
//    Sn115 = new G4Isotope("Sn115", 50, 115, 114.9033*g/mole);
//    Sn116 = new G4Isotope("Sn116", 50, 116, 115.9017*g/mole);
//    Sn117 = new G4Isotope("Sn117", 50, 117, 116.9030*g/mole);
//    Sn118 = new G4Isotope("Sn118", 50, 118, 117.9016*g/mole);
//    Sn119 = new G4Isotope("Sn119", 50, 119, 118.9033*g/mole);
//    Sn120 = new G4Isotope("Sn120", 50, 120, 119.9022*g/mole);
//    Sn122 = new G4Isotope("Sn122", 50, 122, 121.9034*g/mole);
//    Sn124 = new G4Isotope("Sn124", 50, 124, 123.9053*g/mole);
//
//    Sn = new StorkElement("Tin", "Sn", 10);
//    Sn->AddIsotope(Sn112, 1.02987*perCent);
//    Sn->AddIsotope(Sn114, 0.684453*perCent);
//    Sn->AddIsotope(Sn115, 0.353998*perCent);
//    Sn->AddIsotope(Sn116, 14.886*perCent);
//    Sn->AddIsotope(Sn117, 7.7997*perCent);
//    Sn->AddIsotope(Sn118, 24.3796*perCent);
//    Sn->AddIsotope(Sn119, 8.58074*perCent);
//    Sn->AddIsotope(Sn120, 32.2277*perCent);
//    Sn->AddIsotope(Sn122, 4.50456*perCent);
//    Sn->AddIsotope(Sn124, 5.5534*perCent);
//
//    //make Zirconium isotopes and element
//    Zr90 = new G4Isotope("Zr90", 40, 90, 89.9047044*g/mole);
//    Zr91 = new G4Isotope("Zr91", 40, 91, 90.9056458*g/mole);
//    Zr92 = new G4Isotope("Zr92", 40, 92, 91.9050408*g/mole);
//    Zr94 = new G4Isotope("Zr94", 40, 94, 93.9063152*g/mole);
//    Zr96 = new G4Isotope("Zr96", 40, 96, 95.9082734*g/mole);
//
//    Zr = new StorkElement("Zirconium", "Zr", 5);
//    Zr->AddIsotope(Zr90, 52.32717*perCent);
//    Zr->AddIsotope(Zr91, 11.34266*perCent);
//    Zr->AddIsotope(Zr92, 16.82896*perCent);
//    Zr->AddIsotope(Zr94, 16.81345*perCent);
//    Zr->AddIsotope(Zr96,  2.68775*perCent);
//
//    Th232 = new G4Isotope("Th232", 90, 232, 232.0381*g/mole);
//
//    Th = new StorkElement("Thorium", "Th", 1);
//    Th->AddIsotope(Th232, 100.*perCent);
//
//    //make Plutonium isotopes and element
//    Pu238 = new G4Isotope("Pu238", 94, 238, 238.0496*g/mole);
//    Pu239 = new G4Isotope("Pu239", 94, 239, 239.0522*g/mole);
//    Pu240 = new G4Isotope("Pu240", 94, 240, 240.0538*g/mole);
//    Pu241 = new G4Isotope("Pu241", 94, 241, 241.0569*g/mole);
//    Pu242 = new G4Isotope("Pu242", 94, 242, 242.0587*g/mole);
//
//// fixed the isotope compositon so that it is interms of abundance instead of weight percentage
//    Pu = new StorkElement("Plutonium", "Pu", 5);
//    Pu->AddIsotope(Pu238, 2.77*perCent);
//    Pu->AddIsotope(Pu239, 52.11*perCent);
//    Pu->AddIsotope(Pu240, 22.93*perCent);
//    Pu->AddIsotope(Pu241, 15.15*perCent);
//    Pu->AddIsotope(Pu242, 7.03*perCent);
///*
//    G4ElementTable *tempTab = Pu->GetElementTable();
//    G4IsotopeVector *tempVec;
//    G4double *tempVecAbun;
//    for(int i=0; i<tempTab->size(); i++)
//    {
//        G4cout << "Element :" << ((*tempTab)[i])->GetName() << " contains " << ((*tempTab)[i])->GetNumberOfIsotopes() << " isotopes" << G4endl;
//        tempVec = ((*tempTab)[i])->GetIsotopeVector();
//        tempVecAbun = ((*tempTab)[i])->GetRelativeAbundanceVector();
//        for(int j=0; j<((*tempTab)[i])->GetNumberOfIsotopes(); j++)
//        {
//            G4cout << "Isotope :" << ((*tempVec)[j])->GetName() << " has abundance" << tempVecAbun[j] << G4endl;
//        }
//    }
//
//    G4cout << "\n### Element properties after they have been added to the materials ###\n" << G4endl;
//*/
//    // Create the world material
//    World = new StorkMaterial("Galactic", 1, 1, 1.e-25*g/cm3, kStateGas,
//						   2.73*kelvin, 3.e-18*pascal);
//
//    // Create H20 material
//    H2O = new StorkMaterial("LightWater", 1.*g/cm3, 2, kStateLiquid, coolantTemp);
//    H2O->AddElement(H,11.191*perCent);
//    H2O->AddElement(Oxygen,88.809*perCent);
//
//    // Create D20 material
//    D2O = new StorkMaterial("HeavyWater", 1.1*g/cm3, 2, kStateLiquid, moderatorTemp);
//    D2O->AddElement(D,2);
//    D2O->AddElement(Oxygen,1);
//
//
//    // Create Coolant
//	Coolant = new StorkMaterial("Coolant", coolantDensity, 1, kStateLiquid,
//							 coolantTemp);
//    Coolant->AddMaterial(H2O,  100*perCent);
//
//    //Create Moderator
//    Moderator = new StorkMaterial("Moderator", moderatorDensity, 2, kStateLiquid,
//							   moderatorTemp);
//    Moderator->AddElement(HMod,20.1*perCent);
//    Moderator->AddElement(Oxygen,79.9*perCent);
//
//    //Create Moderator
//    ExelLiner = new StorkMaterial("ExelLiner", 1, 4, kStateSolid, pressTubeTemp);
//    ExelLiner->AddElement(Sn,3.5*perCent);
//    ExelLiner->AddElement(Mo,0.8*perCent);
//    ExelLiner->AddElement(Nb,0.8*perCent);
//    ExelLiner->AddElement(Zr,94.9*perCent);
//
//    ZircSteel = new StorkMaterial("ZircSteel", 1, 10, kStateSolid, outSheatheTemp);
//    ZircSteel->AddElement(C,0.034*perCent);
//    ZircSteel->AddElement(Si,0.51*perCent);
//    ZircSteel->AddElement(Mn,0.74*perCent);
//    ZircSteel->AddElement(P,0.016*perCent);
//    ZircSteel->AddElement(S,0.002*perCent);
//    ZircSteel->AddElement(Ni,20.82*perCent);
//    ZircSteel->AddElement(Cr,25.04*perCent);
//    ZircSteel->AddElement(Fe,51.738*perCent);
//    ZircSteel->AddElement(Mo,0.817*perCent);
//    ZircSteel->AddElement(Zr,0.283*perCent);
//
//    /*ZircHydrid = new StorkMaterial("ZircHydrid", 1, 2, kStateSolid, 1);
//    ZircHydrid->AddElement(Zr,98.26*perCent);
//    ZircHydrid->AddElement(H,1.74*perCent);*/
//
//    Insulator = new StorkMaterial("Insulator", insulatorDensity, 3, kStateSolid, insulatorTemp);
//    Insulator->AddElement(Zr,66.63*perCent);
//    Insulator->AddElement(Y,7.87*perCent);
//    Insulator->AddElement(Oxygen,25.5*perCent);
//
//    //Create Fuel
//    OuterFuel = new StorkMaterial("OuterFuel", outerFuelDensity, 3, kStateSolid, outerFuelTemp);
//    OuterFuel->AddElement(Oxygen,12.08*perCent);
//    OuterFuel->AddElement(Pu,10.59*perCent);
//    OuterFuel->AddElement(Th,77.34*perCent);
//
//    InnerFuel = new StorkMaterial("InnerFuel", innerFuelDensity, 3, kStateSolid, innerFuelTemp);
//    InnerFuel->AddElement(Oxygen,12.07*perCent);
//    InnerFuel->AddElement(Pu,13.23*perCent);
//    InnerFuel->AddElement(Th,74.7*perCent);
//
//    OutSheathe = new StorkMaterial("OutSheathe", outSheatheDensity, 1, kStateSolid, outSheatheTemp);
//    OutSheathe->AddMaterial(ZircSteel,100*perCent);
//
//    InSheathe = new StorkMaterial("InSheathe", inSheatheDensity, 1, kStateSolid, inSheatheTemp);
//    InSheathe->AddMaterial(ZircSteel,100*perCent);
//
//    Liner = new StorkMaterial("Liner", linerDensity, 1, kStateSolid, linerTemp);
//    Liner->AddMaterial(ZircSteel,100*perCent);
//
//    FlowTube = new StorkMaterial("FlowTube", flowTubeDensity, 1, kStateSolid, flowTubeTemp);
//    FlowTube->AddMaterial(ZircSteel,100*perCent);
//
//    PressTube = new StorkMaterial("PressTube", pressTubeDensity, 1, kStateSolid, pressTubeTemp);
//    PressTube->AddMaterial(ExelLiner,100*perCent);
//
//    OutLiner = new StorkMaterial("OutLiner", outLinerDensity, 1, kStateSolid, outLinerTemp);
//    OutLiner->AddMaterial(ExelLiner,100*perCent);
//
//    CentralCoolant = new StorkMaterial("CentralCoolant", centralCoolantDensity, 1, kStateLiquid, centralCoolantTemp);
//    CentralCoolant->AddMaterial(Coolant,  100*perCent);

    // Add materials to the map indexed by either ZA (format ZZAAA or ZZ)
    // For composite materials:  world is 0, heavy water is 1, UHW is 2
    matMap["Galactic"] = World;
    matMap["Moderator"] = Moderator;
    matMap["PressTube"] = PressTube;
    matMap["OutLiner"] = OutLiner;
    matMap["Insulator"] = Insulator;
    matMap["Liner"] = Liner;
    matMap["Coolant"] = Coolant;
    matMap["OutSheathe"] = OutSheathe;
    matMap["InSheathe"] = InSheathe;
    matMap["OuterFuel"] = OuterFuel;
    matMap["InnerFuel"] = InnerFuel;
    matMap["FlowTube"] = FlowTube;
    matMap["CentralCoolant"] = CentralCoolant;

/*
    G4ElementVector *elemVec;
    G4double *fracVec;

    for(std::map<G4String,G4Material*>::iterator it=matMap.begin(); it!=matMap.end(); ++it)
    {
        G4cout << "\nMaterial :" << (it->second)->GetName() << " contains " << (it->second)->GetNumberOfElements() << " elements" << " and has a temperature " << (it->second)->GetTemperature() << G4endl;
        elemVec=(it->second)->GetElementVector();
        fracVec=(it->second)->GetFractionVector();
        for(int j=0; j<(it->second)->GetNumberOfElements(); j++)
        {
            G4cout << "Element :" << ((*elemVec)[j])->GetName() << " has wt%" << fracVec[j] << G4endl;
            G4cout << "Element :" << ((*elemVec)[j])->GetName() << " contains " << ((*elemVec)[j])->GetNumberOfIsotopes() << " isotopes" << " and has a temperature " << (dynamic_cast<StorkElement*>((*elemVec)[j]))->GetTemperature() << G4endl;
            tempVec = ((*elemVec)[j])->GetIsotopeVector();
            tempVecAbun = ((*elemVec)[j])->GetRelativeAbundanceVector();
            for(int k=0; k<((*elemVec)[j])->GetNumberOfIsotopes(); k++)
            {
                G4cout << "Isotope :" << ((*tempVec)[k])->GetName() << " has abundance " << tempVecAbun[k] << G4endl;
            }
        }
    }

    G4cout << "\n### Element properties for the entire table ###\n" << G4endl;

    for(int i=0; i<tempTab->size(); i++)
    {
        G4cout << "Element :" << ((*tempTab)[i])->GetName() << " contains " << ((*tempTab)[i])->GetNumberOfIsotopes() << " isotopes" << G4endl;
        tempVec = ((*tempTab)[i])->GetIsotopeVector();
        tempVecAbun = ((*tempTab)[i])->GetRelativeAbundanceVector();
        for(int j=0; j<((*tempTab)[i])->GetNumberOfIsotopes(); j++)
        {
            G4cout << "Isotope :" << ((*tempVec)[j])->GetName() << " has abundance" << tempVecAbun[j] << G4endl;
        }
    }
*/
    matChanged = false;

    return;
}
