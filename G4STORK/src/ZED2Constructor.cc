
/*
ZED2Constructor.cc

Created by:		Salma Mahzooni
Date:			26-07-2013
Modified:               06-12-2014

Source code for the ZED2 geometry and materials

*/

#include "ZED2Constructor.hh"


// Constructor
ZED2Constructor::ZED2Constructor()
: StorkVWorldConstructor()
{
    vesselVisAtt=NULL;
    tank1VisATT=NULL;
    ModVisAtt=NULL;
    fuelA1VisATT=NULL;
    fuelB1VisATT=NULL;
    sheathA1VisATT=NULL;
    sheathB1VisATT=NULL;
    Air1VisAtt=NULL;
    Coolant1VisAtt=NULL;
    Pressure1VisAtt=NULL;
    GasAnn1VisAtt=NULL;
    Calandria1VisAtt=NULL;
    EndPlate2VisATT=NULL;
    airTubeVisAtt=NULL;
    DumplineAlVisAtt=NULL;
    DumplineHWVisAtt=NULL;

	// Set default values for member variables
   	matTemp = 299.51/*293.6*/*kelvin;

}


// Destructor
ZED2Constructor::~ZED2Constructor()
{
    // Delete visualization attributes
    if(vesselVisAtt)
        delete vesselVisAtt;
    if(tank1VisATT)
        delete tank1VisATT;
    if(ModVisAtt)
        delete ModVisAtt;
    if(fuelA1VisATT)
        delete fuelA1VisATT;
    if(fuelB1VisATT)
        delete fuelB1VisATT;
    if(sheathA1VisATT)
        delete sheathA1VisATT;
    if(sheathB1VisATT)
        delete sheathB1VisATT;
    if(Air1VisAtt)
        delete Air1VisAtt;
    if(Coolant1VisAtt)
        delete Coolant1VisAtt;
    if(Pressure1VisAtt)
        delete Pressure1VisAtt;
    if(GasAnn1VisAtt)
        delete GasAnn1VisAtt;
    if(Calandria1VisAtt)
        delete Calandria1VisAtt;
    if(EndPlate2VisATT)
        delete EndPlate2VisATT;
    if(airTubeVisAtt)
        delete airTubeVisAtt;
    if(DumplineAlVisAtt)
        delete DumplineAlVisAtt;
    if(DumplineHWVisAtt)
        delete DumplineHWVisAtt;

}


// ConstructNewWorld()
// Build bare sphere world for the first time.  Set default values and user
// inputs.  Also set up the variable property map.
G4VPhysicalVolume*
ZED2Constructor::ConstructNewWorld(const StorkParseInput* infile)
{

    // Call base class ConstructNewWorld() to complete construction
    return StorkVWorldConstructor::ConstructNewWorld(infile);
}


// ConstructWorld
// Construct the geometry and materials of the spheres given the inputs.
G4VPhysicalVolume* ZED2Constructor::ConstructWorld()
{



	// Set local variables and enclosed world dimensions
	reactorDim = G4ThreeVector(0.*cm,  231.806*cm ,550.*cm/2);
	G4double buffer = 1.0*cm;
	encWorldDim = G4ThreeVector(reactorDim[1]+buffer, reactorDim[1]+buffer, 2*reactorDim[2]+buffer);
	G4SolidStore* theSolids = G4SolidStore::GetInstance();

    //Defining the graphite wall and bottom
    G4double Graphitewall[3] ={0.*cm, 231.806*cm, 315.4*cm/2.};
    G4double Graphitebott[3] = {0., 231.806*cm,90.0*cm/2.};


    //Defining the shielding wall
    //G4double Shieldingwall[3] = {168.6306*cm,231.7*cm,17.6*cm/2.};

	// Create Dimensions of Calandria Tank
    G4double CalandriaDim1[3] = { 0.*cm, 168.635*cm, 315.4*cm/2.-2.69*cm/2.};
    G4double BotReacTankDim[3] = {0.*cm, 168.635*cm, 2.69*cm/2.};

    // Defining the dimensions of the moderator
    G4double ModHeight = 132.707*cm;
    G4double distbtwflrtofuel = 10.1124*cm;
    G4double RodHeight = 2.0*CalandriaDim1[2]-distbtwflrtofuel;
    G4double MTankDim[3] = {0.*cm, 168.0*cm, ModHeight};
    G4double TubeAirFuel[3] = {0.0*cm, 168.0*cm, (2*CalandriaDim1[2]-ModHeight)/2.};


	// Create Dimensions of Fuel Assembly
    G4double CalendriaT1Dim[3] = {0.0*cm, 12.74*cm, RodHeight};
    G4double GasAnn1Dim[3] = {0.0*cm, 12.46*cm, RodHeight};
    G4double PressureT1Dim[3] = {0.0*cm, 10.78*cm, RodHeight};
    G4double Coolant1Dim[3] = {0.0*cm, 10.19*cm, (5.*(49.51*cm))};
    G4double Air1Dim[3] = {0.0*cm, 10.19*cm, RodHeight-(5.*(49.51*cm))};
    G4double EndPlate2[3] = { 0.0*cm, 4.585*cm, 0.16*cm/2.0};
    G4double FuelRodADim1[3] = {0.0*cm, 1.264*cm,48.25*cm/2.};
    G4double FuelRodBDim1[3] = {0.0*cm, 1.070*cm,48.0*cm/2.};
    G4double SheathADim1[3] = {0.0*cm, 1.350*cm, 49.19*cm/2.};
    G4double SheathBDim1[3] = {0.0*cm, 1.150*cm, 49.19*cm/2.};
	// Create the ring for fuel pins placement
	G4int rings = 4;
    G4double ringRad[3] = {1.734*cm,3.075*cm,4.384*cm};
    G4double secondRingOffset = 0.261799*radian;

	// Create Dimensions of dump lines in graphite
    //G4double DumpLineAlDim[3] = {0.0*cm, 22.86*cm, 90.*cm/2.};
    //G4double DumplineHWDim[3] = {0.0*cm, 22.066*cm, 90.*cm/2.};

	// Create Dimensions of dump lines in Al calandria
    //G4double DumpLineAlDimC[3] = {0.0*cm, 22.86*cm, 2.69*cm/2.};
    //G4double DumplineHWDimC[3] = {0.0*cm, 22.066*cm, 2.69*cm/2.};

    G4double topCalandriatoModH = 2.*CalandriaDim1[2]-ModHeight;
    G4double AirinCT = RodHeight-Coolant1Dim[2];
    G4double topFueltoModH = topCalandriatoModH-AirinCT;
    G4double FuelinModH = Coolant1Dim[2]-(topFueltoModH);
    G4int ModFuelIntersectPin = floor((((topFueltoModH)/10.)/49.51*cm)/10.);
    G4int NumOfFuelBunInMod = floor((((FuelinModH)/10.)/49.51*cm)/10.);
    G4double FullFuelBunInAir = ModFuelIntersectPin*49.51*cm;
    G4double ModFuelIntersectPos = (topFueltoModH-FullFuelBunInAir-0.16*cm);
    G4double CutFuelBunInMod = 49.19*cm-ModFuelIntersectPos;

// Positions of the fuel bundles
 G4double Pich[2] = {24.5*cm, 24.5*cm};
 G4double XPos[] = {
		    Pich[0]/2,Pich[0]/2,Pich[0]/2,Pich[0]/2,
		    3*Pich[0]/2, 3*Pich[0]/2, 3*Pich[0]/2, 3*Pich[0]/2,
		    5*Pich[0]/2,5*Pich[0]/2,5*Pich[0]/2,
		    7*Pich[0]/2,7*Pich[0]/2};

  G4double YPos[] = {
		     Pich[1]/2, 3.*Pich[1]/2, 5.*Pich[1]/2, 7.*Pich[1]/2,
		     Pich[1]/2, 3.*Pich[1]/2, 5.*Pich[1]/2, 7.*Pich[1]/2,
		     Pich[1]/2, 3.*Pich[1]/2, 5.*Pich[1]/2,
		     Pich[1]/2, 3.*Pich[1]/2
                    };


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
		new G4Box("ZED2World", encWorldDim[1]/2, encWorldDim[1]/2 , encWorldDim[2]/2);

		// Create the reactor tube
        //new G4Tubs("RecDimTube", reactorDim[0], reactorDim[1], reactorDim[2], 0., CLHEP::pi/2.0);
		//new G4Box("RecDimTube", reactorDim[1]/2, reactorDim[1]/2 , reactorDim[2]);

		// Create the air above the moderator
        new G4Tubs("AirTube", TubeAirFuel[0], TubeAirFuel[1], TubeAirFuel[2], 0., CLHEP::pi/2.0);


		// Create Graphite Reflector solid
        new G4Tubs("graphitewall", reactorDim[0], reactorDim[1], Graphitewall[2], 0., CLHEP::pi/2.0);
        new G4Tubs("graphitebott", reactorDim[0], reactorDim[1], Graphitebott[2], 0., CLHEP::pi/2.0);
        new G4UnionSolid("graphitewall+graphitebott", theSolids->GetSolid("graphitewall"), theSolids->GetSolid("graphitebott"), 0, G4ThreeVector(0.,0.,-Graphitewall[2]-Graphitebott[2]));

        // Create Sheilding walls
        //new G4Tubs("sheildingwall", Shieldingwall[0], Shieldingwall[1], Shieldingwall[2], 0., 2.0*CLHEP::pi);

        // Create the Calandria solids 1
		new G4Tubs("calandriashell", CalandriaDim1[0], CalandriaDim1[1], CalandriaDim1[2], 0., CLHEP::pi/2.0);
		new G4Tubs("calandriabott", BotReacTankDim[0], BotReacTankDim[1], BotReacTankDim[2], 0., CLHEP::pi/2.0);
        new G4UnionSolid("calandriashell+calandriabott", theSolids->GetSolid("calandriashell"), theSolids->GetSolid("calandriabott"), 0, G4ThreeVector(0,0,(-CalandriaDim1[2]-BotReacTankDim[2])));

         // Create Moderator solid
		new G4Tubs("ModSphere", MTankDim[0], MTankDim[1], MTankDim[2]/2., 0., CLHEP::pi/2.0);

    // Create the air above the coolant tube solid
		new G4Tubs("AirTube1", Air1Dim[0]/2, Air1Dim[1]/2, Air1Dim[2]/2., 0., 2.0*CLHEP::pi);

	 // Create the Calandria tube
		new G4Tubs("CalandriaTubedwnCut1", CalendriaT1Dim[0]/2, CalendriaT1Dim[1]/2, (CalendriaT1Dim[2]-topCalandriatoModH)/2., 0., 2.0*CLHEP::pi);
		new G4Tubs("CalandriaTubedwnCut2", CalendriaT1Dim[0]/2, CalendriaT1Dim[1]/2, topCalandriatoModH/2., 0., 2.0*CLHEP::pi);

    // Create the GasAnn tube solid
		new G4Tubs("GasAnnTube1Cut1", GasAnn1Dim[0]/2, GasAnn1Dim[1]/2, (GasAnn1Dim[2]-topCalandriatoModH)/2., 0., 2.0*CLHEP::pi);
		new G4Tubs("GasAnnTube1Cut2", GasAnn1Dim[0]/2, GasAnn1Dim[1]/2, topCalandriatoModH/2., 0., 2.0*CLHEP::pi);

    // Create the pressure tube solid
		new G4Tubs("PressureTubedwnCut1", PressureT1Dim[0]/2, PressureT1Dim[1]/2, (GasAnn1Dim[2]-topCalandriatoModH)/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("PressureTubedwnCut2", PressureT1Dim[0]/2, PressureT1Dim[1]/2, topCalandriatoModH/2., 0., 2.0*CLHEP::pi);

    // Create the coolant tube solid
		new G4Tubs("CoolantTube1Cut1", Coolant1Dim[0]/2, Coolant1Dim[1]/2, FuelinModH/2., 0., 2.0*CLHEP::pi);
		new G4Tubs("CoolantTube1Cut2", Coolant1Dim[0]/2, Coolant1Dim[1]/2, topFueltoModH/2., 0., 2.0*CLHEP::pi);

    // Create  outer fuel bunndles solid
		new G4Tubs("FuelTubeB1", FuelRodBDim1[0]/2, FuelRodBDim1[1]/2, FuelRodBDim1[2], 0., 2.0*CLHEP::pi);
		new G4Tubs("FuelTubeB1Cut1", FuelRodBDim1[0]/2, FuelRodBDim1[1]/2, CutFuelBunInMod/2., 0., 2.0*CLHEP::pi);
		new G4Tubs("FuelTubeB1Cut2", FuelRodBDim1[0]/2, FuelRodBDim1[1]/2, ModFuelIntersectPos/2., 0., 2.0*CLHEP::pi);


    // Create  inner fuel bunndles solid
		new G4Tubs("FuelTubeA1", FuelRodADim1[0]/2, FuelRodADim1[1]/2, FuelRodADim1[2], 0., 2.0*CLHEP::pi);
		new G4Tubs("FuelTubeA1Cut1", FuelRodADim1[0]/2, FuelRodADim1[1]/2, CutFuelBunInMod/2., 0., 2.0*CLHEP::pi);
		new G4Tubs("FuelTubeA1Cut2", FuelRodADim1[0]/2, FuelRodADim1[1]/2, ModFuelIntersectPos/2., 0., 2.0*CLHEP::pi);


    // Create  outer Zr-4 sheath  solid
		new G4Tubs("SheathB1", SheathBDim1[0]/2, SheathBDim1[1]/2, SheathBDim1[2], 0., 2.0*CLHEP::pi);
		new G4Tubs("SheathB1Cut1", SheathBDim1[0]/2, SheathBDim1[1]/2, CutFuelBunInMod/2., 0., 2.0*CLHEP::pi);
		new G4Tubs("SheathB1Cut2", SheathBDim1[0]/2, SheathBDim1[1]/2, ModFuelIntersectPos/2., 0., 2.0*CLHEP::pi);

    // Create  inner Zr-4 sheath  solid
		new G4Tubs("SheathA1", SheathADim1[0]/2, SheathADim1[1]/2, SheathADim1[2], 0., 2.0*CLHEP::pi);
		new G4Tubs("SheathA1Cut1", SheathADim1[0]/2, SheathADim1[1]/2, CutFuelBunInMod/2., 0., 2.0*CLHEP::pi);
		new G4Tubs("SheathA1Cut2", SheathADim1[0]/2, SheathADim1[1]/2, ModFuelIntersectPos/2., 0., 2.0*CLHEP::pi);

    // Create the end plate
		new G4Tubs("EndPlate2", EndPlate2[0], EndPlate2[1], EndPlate2[2], 0., 2.0*CLHEP::pi);
		new G4Tubs("EndPlate1", EndPlate2[0], EndPlate2[1], EndPlate2[2], 0., 2.0*CLHEP::pi);



		geomChanged = false;
	}


        // Create world volume
        worldLogical = new G4LogicalVolume(theSolids->GetSolid("ZED2World"),matMap["World"], "worldLogical",0,0,0);
        worldPhysical = new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), worldLogical,"worldPhysical",0,0,0);

        //Create the reactor dimension tube
        //reactDimTubeLogical = new G4LogicalVolume(theSolids->GetSolid("RecDimTube"),matMap["World"], "reactDimTubeLogical",0,0,0);
        //new G4PVPlacement(0, G4ThreeVector(/*-reactorDim[1]/2,-reactorDim[1]/2*/ 0,0,0.), reactDimTubeLogical, "reactDimTubePhysical", worldLogical,0,0);

        // Create Reflector volume
        vesselLogical = new G4LogicalVolume(theSolids->GetSolid("graphitewall+graphitebott"), matMap["Graphite"], "VesselLogical", 0, 0, 0);
        new G4PVPlacement(0,G4ThreeVector(-reactorDim[1]/2,-reactorDim[1]/2,-reactorDim[2]+2.*Graphitebott[2]+Graphitewall[2]), vesselLogical, "VesselPhysical",worldLogical , 0, 0);

        // Create Sheilding walls
        //ShieldingwalLogical = new G4LogicalVolume(theSolids->GetSolid("sheildingwall"),matMap["TShield"], "ShieldingwalLogical", 0,0,0);
        //new G4PVPlacement(0, G4ThreeVector(0.,0.,Graphitewall[2]+Graphitebott[2]+Shieldingwall[2]), ShieldingwalLogical, "ShieldingwallPhysical", airboxLogical,0,0,true);


        // Create Calandrai volume in mother air volume
        tankLogical1 = new G4LogicalVolume(theSolids->GetSolid("calandriashell+calandriabott"), matMap["Al57S"], "VesselLogical1", 0, 0, 0);
        new G4PVPlacement(0, G4ThreeVector(0.,0.,BotReacTankDim[2]), tankLogical1,"CalandriaPhysical1", vesselLogical,0,0);

        // Create Moderator volume
        ModLogical = new G4LogicalVolume(theSolids->GetSolid("ModSphere"),matMap["Moderator"], "ModLogical",0,0,0);
        new G4PVPlacement(0, G4ThreeVector(0.,0.,MTankDim[2]/2-CalandriaDim1[2]), ModLogical, "ModPhysical", tankLogical1, false,0);

        //Create Air above the moderator
        airTubeLogical = new G4LogicalVolume(theSolids->GetSolid("AirTube"),matMap["Air"], "airTubeLogical",0,0,0);
        new G4PVPlacement(0, G4ThreeVector(0.,0., TubeAirFuel[2]+MTankDim[2]-CalandriaDim1[2]), airTubeLogical, "airTubePhysical", tankLogical1,0,0);

        std::stringstream volName;

        logicCalandria1 = new G4LogicalVolume(theSolids->GetSolid("CalandriaTubedwnCut2"), matMap["AlCalT"], "CalandriaTube1LogicalCut2", 0, 0, 0);
        for (G4int i=0; i<13; i++)
        {
        // Create calandria tubes in mother air volume
        volName.str("");
        volName.clear();
        volName << i;
        new G4PVPlacement (0, G4ThreeVector(XPos[i],YPos[i],0.), logicCalandria1, "CalandriaTube1PhysicalCut2"+volName.str(),  airTubeLogical, false, 0);
        }
        // Create gas annulus tubes in mother air volume
        logicGasAnn1 = new G4LogicalVolume(theSolids->GetSolid("GasAnnTube1Cut2"), matMap["Air"], "GasAnnTube1Logical", 0, 0, 0);
        new G4PVPlacement (0, G4ThreeVector(0,0,0), logicGasAnn1, "GasAnnTube1PhysicalCut2",   logicCalandria1, false, 0);

        // Create pressure tubes in mother air volume
        logicPressure1 = new G4LogicalVolume(theSolids->GetSolid("PressureTubedwnCut2"), matMap["AlPresT"], "PressureTube1Logical", 0, 0, 0);
        new G4PVPlacement (0, G4ThreeVector(0,0,0), logicPressure1, "PressureTube1PhysicalCut2",  logicGasAnn1, false, 0);

       // Create  lower coolant in mother air volume
        logicCoolant1 = new G4LogicalVolume(theSolids->GetSolid("CoolantTube1Cut2"), matMap["Coolant"], "Coolant1Logical", 0, 0, 0);
        new G4PVPlacement (0, G4ThreeVector(0,0,-topCalandriatoModH/2.+topFueltoModH/2.), logicCoolant1, "Coolant1PhysicalCut2",  logicPressure1, false, 0);

        // Create  air
        logicAir1 = new G4LogicalVolume(theSolids->GetSolid("AirTube1"), matMap["Air"], "Air1Logical", 0, 0, 0);
        new G4PVPlacement (0, G4ThreeVector(0,0,-topCalandriatoModH/2.+topFueltoModH+Air1Dim[2]/2.0), logicAir1, "Air1Physical",  logicPressure1, false, 0);

        // Create inner/outer FULL fuel bunndles and sheath in air volume
        logicRodA1 = new G4LogicalVolume(theSolids->GetSolid("FuelTubeA1"), matMap["LEUMat"], "FuelRodA1Logical", 0, 0, 0);
        logicRodB1 = new G4LogicalVolume(theSolids->GetSolid("FuelTubeB1"), matMap["LEUMat"], "FuelRodB1Logical", 0, 0, 0);
        logicSheathA1 = new G4LogicalVolume(theSolids->GetSolid("SheathA1"), matMap["Zr4"], "SheathA1Logical", 0, 0, 0);
        logicSheathB1 = new G4LogicalVolume(theSolids->GetSolid("SheathB1"), matMap["Zr4"], "SheathB1Logical", 0, 0, 0);
        logicEndPlate1 = new G4LogicalVolume(theSolids->GetSolid("EndPlate1"), matMap["Zr4"], "EndPlate1", 0, 0, 0);
        logicEndPlate2 = new G4LogicalVolume(theSolids->GetSolid("EndPlate2"), matMap["Zr4"], "EndPlate2", 0, 0, 0);

                for (G4int l=0; l<ModFuelIntersectPin; l++)
                {
                // Rotation and translation of the rod and sheathe

                // place the center pin in air
                    new G4PVPlacement(0, G4ThreeVector(0,0, (topFueltoModH/2.-(l+1)*(SheathADim1[2]+2.*EndPlate2[2])-l*(2.*EndPlate2[2]+SheathADim1[2])) ), logicSheathA1,"sheathePhysical " + volName.str(), logicCoolant1,0,0);
                    new G4PVPlacement(0, G4ThreeVector(0,0,0), logicRodA1,"fuelPhysicalA", logicSheathA1,0,0);
                    new G4PVPlacement(0, G4ThreeVector(0,0,0), logicRodB1,"fuelPhysicalB", logicSheathB1,0,0);




                            for( G4int j = 1; j < rings; j++ )
                                {
                                    for( G4int k = 0; k < j*6; k++ )
                                    {
                                        // Reset string stream
                                        volName.str("");

                                        volName << j << "-" << k;

                                            if(j == 2)
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)+secondRingOffset), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)+secondRingOffset),(topFueltoModH/2.-(l+1)*(SheathADim1[2]+2.*EndPlate2[2])-l*(2.*EndPlate2[2]+SheathADim1[2])));
                                                        new G4PVPlacement(0, Tm, logicSheathB1,"sheathePhysical " +volName.str(),logicCoolant1,0,0,0);

                                                    }
                                            else if (j == 1)
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)),(topFueltoModH/2.-(l+1)*(SheathADim1[2]+2.*EndPlate2[2])-l*(2.*EndPlate2[2]+SheathADim1[2])));
                                                        new G4PVPlacement(0, Tm, logicSheathA1,"sheathePhysical " +volName.str(),logicCoolant1,0,0,0);
                                                    }
                                            else
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)),(topFueltoModH/2.-(l+1)*(SheathADim1[2]+2.*EndPlate2[2])-l*(2.*EndPlate2[2]+SheathADim1[2])));
                                                        new G4PVPlacement(0, Tm, logicSheathB1,"sheathePhysical " +volName.str(),logicCoolant1,0,0,0);
                                                    }
                                        }
                                    }


                    // Make the end plates 1
                    G4ThreeVector EP1(0,0,(topFueltoModH/2.-EndPlate2[2])-l*(49.51*cm));
                    new G4PVPlacement(0, EP1, logicEndPlate1,"EndPlate1Physical1",logicCoolant1,0,0);
                    // Make the end plates 2
                    G4ThreeVector EP2(0,0,(topFueltoModH/2.-EndPlate2[2])-l*(2.*EndPlate2[2])-(l+1)*(2.*SheathADim1[2]+2.*EndPlate2[2]));
                    new G4PVPlacement(0, EP2, logicEndPlate2,"EndPlate2Physical1",logicCoolant1,0,0);

                    }


        // Create inner/outer cut fuel bunndl in air volume
        logicRodA1Cut2 = new G4LogicalVolume(theSolids->GetSolid("FuelTubeA1Cut2"), matMap["LEUMat"], "FuelRodA1LogicalCut2", 0, 0, 0);
        logicRodB1Cut2 = new G4LogicalVolume(theSolids->GetSolid("FuelTubeB1Cut2"), matMap["LEUMat"], "FuelRodB1LogicalCut2", 0, 0, 0);
        logicSheathA1Cut2 = new G4LogicalVolume(theSolids->GetSolid("SheathA1Cut2"), matMap["Zr4"], "SheathA1Logicalcut2", 0, 0, 0);
        logicSheathB1Cut2 = new G4LogicalVolume(theSolids->GetSolid("SheathB1Cut2"), matMap["Zr4"], "SheathB1LogicalCut2", 0, 0, 0);
        logicEndPlate2Cut2 = new G4LogicalVolume(theSolids->GetSolid("EndPlate2"), matMap["Zr4"], "EndPlate2Cut2", 0, 0, 0);
                // Rotation and translation of the rod and sheathe

                // Set name for sheathe physical volume

                    volName.str("");
                    volName << 0;

                // place the center pin in air
                    new G4PVPlacement(0, G4ThreeVector(0,0,-topFueltoModH/2.+ModFuelIntersectPos/2.), logicSheathA1Cut2,"sheathePhysicalCut2 " + volName.str(), logicCoolant1,0,0);
                    new G4PVPlacement(0, G4ThreeVector(0,0,0), logicRodA1Cut2,"fuelPhysicalCut2A", logicSheathA1Cut2,0,0);
                    new G4PVPlacement(0, G4ThreeVector(0,0,0), logicRodB1Cut2,"fuelPhysicalCut2B", logicSheathB1Cut2,0,0);




                            for( G4int j = 1; j < rings; j++ )
                                {
                                    for( G4int k = 0; k < j*6; k++ )
                                    {
                                        // Reset string stream
                                        volName.str("");

                                        volName << j << "-" << k;

                                            if(j == 2)
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)+secondRingOffset), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)+secondRingOffset),-topFueltoModH/2.+ModFuelIntersectPos/2.);
                                                        new G4PVPlacement(0, Tm, logicSheathB1Cut2,"sheathePhysicalCut2 " +volName.str(),logicCoolant1,0,0);

                                                    }
                                            else if (j == 1)
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)),-topFueltoModH/2.+ModFuelIntersectPos/2.);
                                                        new G4PVPlacement(0, Tm, logicSheathA1Cut2,"sheathePhysicalCut2 " +volName.str(),logicCoolant1,0,0);
                                                    }
                                            else
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)),-topFueltoModH/2.+ModFuelIntersectPos/2.);
                                                        new G4PVPlacement(0, Tm, logicSheathB1Cut2,"sheathePhysicalCut2 " +volName.str(),logicCoolant1,0,0);
                                                    }
                                        }
                                    }


                    // Make the end plates 2
                    G4ThreeVector EP2(0,0,-topFueltoModH/2.+ModFuelIntersectPos+0.08*cm);
                    new G4PVPlacement(0, EP2, logicEndPlate2Cut2,"EndPlate2Physical1Cut2",logicCoolant1,0,0);





        // *********Create Calandrai volume in moderator volume*************
        logicCalandria1Mod = new G4LogicalVolume(theSolids->GetSolid("CalandriaTubedwnCut1"), matMap["AlCalT"], "CalandriaTube1ModLogical", 0, 0, 0);
        for (G4int i=0; i<13; i++)
        {
        // Create calandria tubes in moderator volume -reactorDim[2]+2.*Graphitebott[2]+2.*BotReacTankDim[2]+distbtwflrtofuel+(CalendriaT1Dim[2]-topFueltoModH)/2.MTankDim[2]
        volName.str("");
        volName.clear();
        volName << i;
        new G4PVPlacement (0, G4ThreeVector(XPos[i],YPos[i],distbtwflrtofuel/2.), logicCalandria1Mod, "CalandriaTube1ModPhysicalCut1"+volName.str(),  ModLogical, false, 0);
        }
        // Create gas annulus tubes in moderator volume
        logicGasAnn1Mod = new G4LogicalVolume(theSolids->GetSolid("GasAnnTube1Cut1"), matMap["Air"], "GasAnnTube1Logical", 0, 0, 0);
        new G4PVPlacement (0, G4ThreeVector(0,0,0), logicGasAnn1Mod, "GasAnnTube1PhysicalCut1",   logicCalandria1Mod, false, 0);

        // Create pressure tubes in moderator volume
        logicPressure1Mod = new G4LogicalVolume(theSolids->GetSolid("PressureTubedwnCut1"), matMap["AlPresT"], "PressureTube1Logical", 0, 0, 0);
        new G4PVPlacement (0, G4ThreeVector(0,0,0), logicPressure1Mod, "PressureTube1PhysicalCut1",  logicGasAnn1Mod, false, 0);

       // Create  lower coolant in moderator volume
        logicCoolant1Mod = new G4LogicalVolume(theSolids->GetSolid("CoolantTube1Cut1"), matMap["Coolant"], "Coolant1Logical", 0, 0, 0);
        new G4PVPlacement (0, G4ThreeVector(0,0,0), logicCoolant1Mod, "Coolant1PhysicalCut1",  logicPressure1Mod, false, 0);

        // Create inner/outer fuel bunndle and sheath in Moderator volume
        logicRodA1Mod = new G4LogicalVolume(theSolids->GetSolid("FuelTubeA1"), matMap["LEUMat"], "FuelRodA1LogicalMod", 0, 0, 0);
        logicRodB1Mod = new G4LogicalVolume(theSolids->GetSolid("FuelTubeB1"), matMap["LEUMat"], "FuelRodB1LogicalMod", 0, 0, 0);
        logicSheathA1Mod = new G4LogicalVolume(theSolids->GetSolid("SheathA1"), matMap["Zr4"], "SheathA1LogicalMod", 0, 0, 0);
        logicSheathB1Mod = new G4LogicalVolume(theSolids->GetSolid("SheathB1"), matMap["Zr4"], "SheathB1LogicalMod", 0, 0, 0);
        logicEndPlate1Mod = new G4LogicalVolume(theSolids->GetSolid("EndPlate1"), matMap["Zr4"], "EndPlate1Mod", 0, 0, 0);
        logicEndPlate2Mod = new G4LogicalVolume(theSolids->GetSolid("EndPlate2"), matMap["Zr4"], "EndPlate2Mod", 0, 0, 0);

                for (G4int l=0; l<NumOfFuelBunInMod; l++)
                {
                // Rotation and translation of the rod and sheathe

                // Set name for sheathe physical volume
                // place the center pin
                    new G4PVPlacement(0, G4ThreeVector(0,0, (-FuelinModH/2.+(l+1)*(SheathADim1[2]+2.*EndPlate2[2])+l*(2.*EndPlate2[2]+SheathADim1[2]) )), logicSheathA1Mod,"sheathePhysicalMod " + volName.str(), logicCoolant1Mod,0,0);
                    new G4PVPlacement(0, G4ThreeVector(0,0,0), logicRodA1Mod,"fuelPhysicalModA ", logicSheathA1Mod,0,0);
                    new G4PVPlacement(0, G4ThreeVector(0,0,0), logicRodB1Mod,"fuelPhysicalModB ", logicSheathB1Mod,0,0);




                            for( G4int j = 1; j < rings; j++ )
                                {
                                    for( G4int k = 0; k < j*6; k++ )
                                    {
                                        // Reset string stream
                                        volName.str("");

                                        volName << j << "-" << k;

                                            if(j == 2)
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)+secondRingOffset), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)+secondRingOffset),(-FuelinModH/2.+(l+1)*(SheathADim1[2]+2.*EndPlate2[2])+l*(2.*EndPlate2[2]+SheathADim1[2])));
                                                        new G4PVPlacement(0, Tm, logicSheathB1Mod,"sheathePhysicalMod " +volName.str(),logicCoolant1Mod,0,0);



                                                    }
                                            else if (j == 1)
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)),(-FuelinModH/2.+(l+1)*(SheathADim1[2]+2.*EndPlate2[2])+l*(2.*EndPlate2[2]+SheathADim1[2])));
                                                        new G4PVPlacement(0, Tm, logicSheathA1Mod,"sheathePhysicalMod " +volName.str(),logicCoolant1Mod,0,0);

                                                    }
                                            else
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)),(-FuelinModH/2.+(l+1)*(SheathADim1[2]+2.*EndPlate2[2])+l*(2.*EndPlate2[2]+SheathADim1[2])));
                                                        new G4PVPlacement(0, Tm, logicSheathB1Mod,"sheathePhysicalMod " +volName.str(),logicCoolant1Mod,0,0);

                                                    }
                                        }
                                    }
                   // Make the end plates 1
                    G4ThreeVector EP1(0,0,(-FuelinModH/2.+EndPlate2[2])+l*(49.51*cm));
                    //G4cout << "Endplate1  "<<(-Coolant1Dim[2]+EndPlate2[2])+l*(49.51*cm)<< G4endl;
                    new G4PVPlacement(0, EP1, logicEndPlate1Mod,"EndPlate1Physical1Mod ",logicCoolant1Mod,0,0);
                    // Make the end plates 2
                    G4ThreeVector EPP2(0,0,(-FuelinModH/2.+EndPlate2[2])+l*(2.*EndPlate2[2])+(l+1)*(2.*SheathADim1[2]+2.*EndPlate2[2]));
                    new G4PVPlacement(0, EP2, logicEndPlate2Mod,"EndPlate2Physical1Mod ",logicCoolant1Mod,0,0);
                    }


    // Create inner/outer cut fuel bunndle in Moderator volume
        logicRodA1Cut1 = new G4LogicalVolume(theSolids->GetSolid("FuelTubeA1Cut1"), matMap["LEUMat"], "FuelRodA1LogicalCut1", 0, 0, 0);
        logicRodB1Cut1 = new G4LogicalVolume(theSolids->GetSolid("FuelTubeB1Cut1"), matMap["LEUMat"], "FuelRodB1LogicalCut1", 0, 0, 0);
        logicSheathA1Cut1 = new G4LogicalVolume(theSolids->GetSolid("SheathA1Cut1"), matMap["Zr4"], "SheathA1LogicalCut1", 0, 0, 0);
        logicSheathB1Cut1 = new G4LogicalVolume(theSolids->GetSolid("SheathB1Cut1"), matMap["Zr4"], "SheathB1LogicalCut1", 0, 0, 0);
        logicEndPlate2Cut1 = new G4LogicalVolume(theSolids->GetSolid("EndPlate2"), matMap["Zr4"], "EndPlate2Cut1", 0, 0, 0);

                // place the center pin for the cut fuel bundle in the moderator
                    new G4PVPlacement(0, G4ThreeVector(0,0, (-FuelinModH/2.+(NumOfFuelBunInMod*49.51*cm)+2.*EndPlate2[2]+CutFuelBunInMod/2. )), logicSheathA1Cut1,"sheathePhysicalCut1 " + volName.str(), logicCoolant1Mod,0,0);
                    new G4PVPlacement(0, G4ThreeVector(0,0,0), logicRodA1Cut1,"fuelPhysicalCut1A ", logicSheathA1Cut1,0,0);
                    new G4PVPlacement(0, G4ThreeVector(0,0,0), logicRodB1Cut1,"fuelPhysicalCut1B ", logicSheathB1Cut1,0,0);




                    // Make the end plates 2
                    G4ThreeVector EP(0,0,(-FuelinModH/2.+(NumOfFuelBunInMod*49.51*cm)+0.08*cm));
                    new G4PVPlacement(0, EP, logicEndPlate2Cut1,"EndPlate2Physical1Cut1 ",logicCoolant1Mod,0,0);

                            for( G4int j = 1; j < rings; j++ )
                                {
                                    for( G4int k = 0; k < j*6; k++ )
                                    {
                                        // Reset string stream
                                        volName.str("");

                                        volName << j << "-" << k;

                                            if(j == 2)
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)+secondRingOffset), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)+secondRingOffset),(-FuelinModH/2.+(NumOfFuelBunInMod*49.51*cm)+2.*EndPlate2[2]+CutFuelBunInMod/2. ));
                                                        // place the fuel for the cut fuel bundle in the moderator
                                                        new G4PVPlacement(0, Tm, logicSheathB1Cut1,"sheathePhysicalCut1 " +volName.str(),logicCoolant1Mod,0,0);

                                                    }
                                            else if (j == 1)
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)),(-FuelinModH/2.+(NumOfFuelBunInMod*49.51*cm)+2.*EndPlate2[2]+CutFuelBunInMod/2. ));
                                                        new G4PVPlacement(0, Tm, logicSheathA1Cut1,"sheathePhysicalCut1 " +volName.str(),logicCoolant1Mod,0,0);
                                                    }
                                            else
                                                    {
                                                        G4ThreeVector Tm(ringRad[j-1]*cos(2.0*CLHEP::pi*G4double(k)/G4double(j*6)), ringRad[j-1]*sin(2.0*CLHEP::pi*G4double(k)/G4double(j*6)),(-FuelinModH/2.+(NumOfFuelBunInMod*49.51*cm)+2.*EndPlate2[2]+CutFuelBunInMod/2. ));
                                                        // place the fuel for the cut fuel bundle in the moderator
                                                        new G4PVPlacement(0, Tm, logicSheathB1Cut1,"sheathePhysicalCut1 " +volName.str(),logicCoolant1Mod,0,0);
                                                    }
                                        }
                                    }





    // Set reactor as sensitive detector
	worldLogical->SetSensitiveDetector( sDReactor );
	//reactDimTubeLogical->SetSensitiveDetector( sDReactor);
    airTubeLogical->SetSensitiveDetector( sDReactor );
	vesselLogical->SetSensitiveDetector( sDReactor );
    tankLogical1->SetSensitiveDetector( sDReactor );
    ModLogical->SetSensitiveDetector( sDReactor );
	logicCalandria1->SetSensitiveDetector( sDReactor );
    logicGasAnn1->SetSensitiveDetector( sDReactor );
	logicPressure1->SetSensitiveDetector( sDReactor );
    logicCoolant1->SetSensitiveDetector( sDReactor );
    logicAir1->SetSensitiveDetector( sDReactor );
	logicRodA1->SetSensitiveDetector( sDReactor );
	logicRodB1->SetSensitiveDetector( sDReactor );
	logicSheathA1->SetSensitiveDetector( sDReactor );
	logicSheathB1->SetSensitiveDetector( sDReactor );
    logicEndPlate2->SetSensitiveDetector( sDReactor );
    logicEndPlate1->SetSensitiveDetector( sDReactor );
    logicCalandria1Mod->SetSensitiveDetector( sDReactor );
    logicGasAnn1Mod->SetSensitiveDetector( sDReactor );
	logicPressure1Mod->SetSensitiveDetector( sDReactor );
	logicCoolant1Mod->SetSensitiveDetector( sDReactor );
    logicRodA1Cut2->SetSensitiveDetector( sDReactor );
	logicRodB1Cut2->SetSensitiveDetector( sDReactor );
	logicSheathA1Cut2->SetSensitiveDetector( sDReactor );
	logicSheathB1Cut2->SetSensitiveDetector( sDReactor );
    logicEndPlate2Cut2->SetSensitiveDetector( sDReactor );
    logicRodA1Mod->SetSensitiveDetector( sDReactor );
	logicRodB1Mod->SetSensitiveDetector( sDReactor );
	logicSheathA1Mod->SetSensitiveDetector( sDReactor );
	logicSheathB1Mod->SetSensitiveDetector( sDReactor );
    logicEndPlate2Mod->SetSensitiveDetector( sDReactor );
    logicEndPlate1Mod->SetSensitiveDetector( sDReactor );
    logicRodA1Cut1->SetSensitiveDetector( sDReactor );
	logicRodB1Cut1->SetSensitiveDetector( sDReactor );
	logicSheathA1Cut1->SetSensitiveDetector( sDReactor );
	logicSheathB1Cut1->SetSensitiveDetector( sDReactor );
    logicEndPlate2Cut1->SetSensitiveDetector( sDReactor );



    // Set visualization attributes

    if(worldVisAtt)
        delete worldVisAtt;
    if(vesselVisAtt)
        delete vesselVisAtt;
    if(tank1VisATT)
        delete tank1VisATT;
    if(ModVisAtt)
        delete ModVisAtt;
    if(fuelA1VisATT)
        delete fuelA1VisATT;
    if(fuelB1VisATT)
        delete fuelB1VisATT;
    if(sheathA1VisATT)
        delete sheathA1VisATT;
    if(sheathB1VisATT)
        delete sheathB1VisATT;
    if(Air1VisAtt)
        delete Air1VisAtt;
    if(Coolant1VisAtt)
        delete Coolant1VisAtt;
    if(Pressure1VisAtt)
        delete Pressure1VisAtt;
    if(GasAnn1VisAtt)
        delete GasAnn1VisAtt;
    if(Calandria1VisAtt)
        delete Calandria1VisAtt;
    if(EndPlate2VisATT)
        delete EndPlate2VisATT;
    if(airTubeVisAtt)
        delete airTubeVisAtt;
    if(DumplineAlVisAtt)
        delete DumplineAlVisAtt;
    if(DumplineHWVisAtt)
        delete DumplineHWVisAtt;

    worldVisAtt = new G4VisAttributes(G4Colour(0.5, 1., 0.5));
    worldVisAtt->SetVisibility(true);
    worldLogical->SetVisAttributes(worldVisAtt);
    //reactDimTubeLogical->SetVisAttributes(worldVisAtt);

    airTubeVisAtt = new G4VisAttributes(G4Colour(0., 1., 0.5));
    airTubeVisAtt->SetVisibility(true);
    airTubeLogical->SetVisAttributes(airTubeVisAtt);

    vesselVisAtt= new G4VisAttributes(G4Colour(1.0,0.0,0.0));
    vesselVisAtt->SetForceSolid(false);
    vesselVisAtt->SetVisibility(true);
    vesselLogical->SetVisAttributes(vesselVisAtt);

    tank1VisATT= new G4VisAttributes(G4Colour(1.0,1.0,0.0));
    tank1VisATT->SetForceSolid(false);
    tank1VisATT->SetVisibility(true);
    tankLogical1->SetVisAttributes(tank1VisATT);

    ModVisAtt = new G4VisAttributes(G4Colour(0.,1.,0.));
    ModVisAtt->SetVisibility(true);
    ModVisAtt->SetForceSolid(false);
    ModLogical->SetVisAttributes(ModVisAtt);

    Calandria1VisAtt = new G4VisAttributes(G4Colour(1., 0., 1.));
    Calandria1VisAtt->SetForceSolid(false);
    Calandria1VisAtt->SetVisibility(false);
    logicCalandria1->SetVisAttributes(Calandria1VisAtt);
    logicCalandria1Mod->SetVisAttributes(Calandria1VisAtt);


    GasAnn1VisAtt = new G4VisAttributes(G4Colour(1., 0., 0.));
    GasAnn1VisAtt->SetForceSolid(false);
    GasAnn1VisAtt->SetVisibility(false);
    logicGasAnn1->SetVisAttributes(GasAnn1VisAtt);
    logicGasAnn1Mod->SetVisAttributes(GasAnn1VisAtt);



    Pressure1VisAtt = new G4VisAttributes(G4Colour(0., 1., 0.));
    Pressure1VisAtt->SetForceSolid(false);
    Pressure1VisAtt->SetVisibility(false);
    logicPressure1->SetVisAttributes(Pressure1VisAtt);
    logicPressure1Mod->SetVisAttributes(Pressure1VisAtt);


    Coolant1VisAtt = new G4VisAttributes(G4Colour(0.53, 0.81, 0.92));
    Coolant1VisAtt->SetForceSolid(false);
    Coolant1VisAtt->SetVisibility(false);
    logicCoolant1->SetVisAttributes(Coolant1VisAtt);
    logicCoolant1Mod->SetVisAttributes(Coolant1VisAtt);

    Air1VisAtt = new G4VisAttributes(G4Colour(0., 1., 0.5));
    Air1VisAtt->SetForceSolid(false);
    Air1VisAtt->SetVisibility(true);
    logicAir1->SetVisAttributes(Air1VisAtt);
    //logicAir1RU->SetVisAttributes(Air1VisAtt);

    fuelA1VisATT = new G4VisAttributes(G4Colour(0.0, 0.0 ,1.0));
    fuelA1VisATT->SetForceSolid(false);
    fuelA1VisATT->SetVisibility(true);
    logicRodA1->SetVisAttributes(fuelA1VisATT);
    logicRodA1Cut2->SetVisAttributes(fuelA1VisATT);
    logicRodA1Mod->SetVisAttributes(fuelA1VisATT);
    logicRodA1Cut1->SetVisAttributes(fuelA1VisATT);

    fuelB1VisATT = new G4VisAttributes(G4Colour(0,0.5,0.92));
    fuelB1VisATT->SetForceSolid(false);
    fuelB1VisATT->SetVisibility(true);
    logicRodB1->SetVisAttributes(fuelB1VisATT);
    logicRodB1Cut2->SetVisAttributes(fuelB1VisATT);
    logicRodB1Mod->SetVisAttributes(fuelB1VisATT);
    logicRodB1Cut1->SetVisAttributes(fuelB1VisATT);

    sheathA1VisATT = new G4VisAttributes(G4Colour(0.5, 0.0 ,1.0));
    sheathA1VisATT->SetForceSolid(false);
    sheathA1VisATT->SetVisibility(false);
    logicSheathA1->SetVisAttributes(sheathA1VisATT);
    logicSheathA1Cut2->SetVisAttributes(sheathA1VisATT);
    logicSheathA1Mod->SetVisAttributes(sheathA1VisATT);
    logicSheathA1Cut1->SetVisAttributes(sheathA1VisATT);

    sheathB1VisATT = new G4VisAttributes(G4Colour(1.0, 0.5 ,1.0));
    sheathB1VisATT->SetForceSolid(false);
    sheathB1VisATT->SetVisibility(false);
    logicSheathB1->SetVisAttributes(sheathB1VisATT);
    logicSheathB1Cut2->SetVisAttributes(sheathB1VisATT);
    logicSheathB1Mod->SetVisAttributes(sheathB1VisATT);
    logicSheathB1Cut1->SetVisAttributes(sheathB1VisATT);


    EndPlate2VisATT = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5));
    EndPlate2VisATT->SetForceSolid(false);
    EndPlate2VisATT->SetVisibility(true);
    logicEndPlate2->SetVisAttributes(EndPlate2VisATT);
    logicEndPlate1->SetVisAttributes(EndPlate2VisATT);
    logicEndPlate2Cut2->SetVisAttributes(EndPlate2VisATT);
    logicEndPlate2Mod->SetVisAttributes(EndPlate2VisATT);
    logicEndPlate1Mod->SetVisAttributes(EndPlate2VisATT);
    logicEndPlate2Cut1->SetVisAttributes(EndPlate2VisATT);

    return worldPhysical;
}


// ConstructMaterials()
// Construct all the materials needed for the ZED2Constructor.
void ZED2Constructor::ConstructMaterials()
{
 // Elements, isotopes and materials
     G4Isotope *U234, *U235, *U238, *U236, *D2,   *O16, *O17,
              *Fe54, *Fe56, *Fe57, *Fe58, *Cr50, *Cr52, *Cr53, *Cr54,
              *Si28, *Si29, *Si30, *Cu63, *Cu65, *Mn55, *Mg24,
              *Mg25, *Mg26, *Zn64, *Zn66, *Zn67, *Zn68, *Zn70,
	          *Al27, *Ti46, *Ti47, *Ti48, *Ti49, *Ti50, *Na23,
	          *Ga69, *Ga71, *H1,   *C12,  *C13,  *Zr90, *Zr91,
	          *Zr92, *Zr94, *Zr96, *Sn112, *Sn114, *Sn115, *Sn116,
	          *Sn117, *Sn118, *Sn119, *Sn120, *Sn122, *Sn124,
	          *Ca40, *Ca42, *Ca43, *Ca44, *Ca46, *Ca48, *B10, *B11,
	          *Li6, *Li7, *Gd152,*Gd154, *Gd155, *Gd156, *Gd157,
              *Gd158, *Gd160,*V50, *V51;
    G4Element *Oxygen, *Deuterium, *LEU,
              *Cr, *Fe, *Si, *Cu, *Mn, *Mg, *Zn, *Al,
              *Ti, *Na, *Ga, *Hydrogen, *C, *Zr, *Sn, *Ca, /*RU,*/
              *B, *Li, *Gd, *V, /*OxygenMod, *HydrogenMod, *OxygenRU,*/
              *FeAl, *CuAl,*FeZr, *CrZr, *OxygenZr,
              *OxygenLEU, /*HydrogenLW,*/ *OxygenLW;
    G4Material *World, *LEUMat, /*HeavyWater,*/
	           *Aluminum57S, *AlPresT, *AlCalT, *H2O, *D2O,
	           /*Coolant,*/ *AnnulusGas, *Zr4, *Air, /*RUMat,*/ *Moderator, *Graphite;

    // Create the world environment
    World = new G4Material("Galactic", 1, 1, 1.e-25*g/cm3, kStateGas,2.73*kelvin, 3.e-18*pascal);


    //make Calcium isotopes and element
    Ca40 = new G4Isotope("Ca40", 20, 40, 39.9625906*g/mole);
    Ca42 = new G4Isotope("Ca42", 20, 42, 41.9586176*g/mole);
    Ca43 = new G4Isotope("Ca43", 20, 43, 42.9587662*g/mole);
    Ca44 = new G4Isotope("Ca44", 20, 44, 43.9554806*g/mole);
    Ca46 = new G4Isotope("Ca46", 20, 46, 45.953689*g/mole);
    Ca48 = new G4Isotope("Ca48", 20, 48, 47.952533*g/mole);

    Ca = new G4Element("Calcium", "Ca", 6);
    Ca->AddIsotope(Ca40,  96.941*perCent);
    Ca->AddIsotope(Ca42,  0.647*perCent);
    Ca->AddIsotope(Ca43,  0.135*perCent);
    Ca->AddIsotope(Ca44,  2.086*perCent);
    Ca->AddIsotope(Ca46,  0.004*perCent);
    Ca->AddIsotope(Ca48,  0.187*perCent);

    //make Boron isotopes and element
    B10 = new G4Isotope("B10", 5, 10, 10.012937*g/mole);
    B11 = new G4Isotope("B11", 5, 11, 11.009305*g/mole);

    B = new G4Element("Boron", "B", 2);
    B->AddIsotope(B10,  19.9*perCent);
    B->AddIsotope(B11,  80.1*perCent);

    //make Lithium isotopes and element
    Li6 = new G4Isotope("Li6", 3, 6, 6.0151223*g/mole);
    Li7 = new G4Isotope("Li7", 3, 7, 7.0160040*g/mole);

    Li = new G4Element("Lithium", "Li", 2);
    Li->AddIsotope(Li6,  7.59 *perCent);
    Li->AddIsotope(Li7,  92.41*perCent);

    //make Vanadium isotopes and element
    V50 = new G4Isotope("V50", 23, 50, 49.9471609 *g/mole);
    V51 = new G4Isotope("V51", 23, 51, 50.9439617 *g/mole);

    V = new G4Element("Vanadium", "V", 2);
    V->AddIsotope(V50,  0.250 *perCent);
    V->AddIsotope(V51,  99.750*perCent);




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

    CrZr = new G4Element("Chromium", "Cr", 4);
    CrZr->AddIsotope(Cr50,  4.10399884*perCent);
    CrZr->AddIsotope(Cr52,  82.20818453*perCent);
    CrZr->AddIsotope(Cr53,  9.50012786*perCent);
    CrZr->AddIsotope(Cr54,  4.18768878*perCent);


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

    //make iron element for Aluminium material in ZED-2
    FeAl = new G4Element("Iron", "Fe", 4);
    FeAl->AddIsotope(Fe54,  0.02340*perCent);
    FeAl->AddIsotope(Fe56,  0.36700*perCent);
    FeAl->AddIsotope(Fe57,  0.00848*perCent);
    FeAl->AddIsotope(Fe58,  0.00112*perCent);

    //make iron element for Aluminium material in ZED-2
    FeZr = new G4Element("Iron", "Fe", 4);
    FeZr->AddIsotope(Fe54,  5.60198907*perCent);
    FeZr->AddIsotope(Fe56,  91.9458541*perCent);
    FeZr->AddIsotope(Fe57,  2.14094671*perCent);
    FeZr->AddIsotope(Fe58,  0.31121012*perCent);

    //make Silicon isotopes and element
    Si28 = new G4Isotope("Si28", 14, 28, 27.9769271*g/mole);
    Si29 = new G4Isotope("Si29", 14, 29, 28.9764949*g/mole);
    Si30 = new G4Isotope("Si30", 14, 30, 29.9737707*g/mole);

    Si = new G4Element("Silicon", "Si", 3);
    Si->AddIsotope(Si28,  92.23*perCent);
    Si->AddIsotope(Si29,  4.67*perCent);
    Si->AddIsotope(Si30,  3.1*perCent);

    //make Magnesium isotopes and element
    Mg24 = new G4Isotope("Mg24", 12, 24, 23.9850423*g/mole);
    Mg25 = new G4Isotope("Mg25", 12, 25, 24.9858374*g/mole);
    Mg26 = new G4Isotope("Mg26", 12, 26, 25.9825937 *g/mole);

    Mg = new G4Element("Magnesium", "Mg", 3);
    Mg->AddIsotope(Mg24,  78.99*perCent);
    Mg->AddIsotope(Mg25,  10.00*perCent);
    Mg->AddIsotope(Mg26,  11.01*perCent);

    //make Manganese isotopes and element
    Mn55 = new G4Isotope("Mn55", 25, 55, 54.9380471*g/mole);

    Mn = new G4Element("Manganese", "Mn", 1);
    Mn->AddIsotope(Mn55,  100.00*perCent);

    //make Copper isotopes and element
    Cu63 = new G4Isotope("Cu63", 29, 63, 62.9295989*g/mole);
    Cu65 = new G4Isotope("Cu65", 29, 65, 64.9277929 *g/mole);

    Cu = new G4Element("Copper", "Cu", 2);
    Cu->AddIsotope(Cu63,  69.17*perCent);
    Cu->AddIsotope(Cu65,  30.83*perCent);

    //make copper for Al
    CuAl = new G4Element("Copper", "Cu", 2);
    CuAl->AddIsotope(Cu63,  0.01383*perCent);
    CuAl->AddIsotope(Cu65,  0.00617*perCent);

    //make Aluminum isotopes and element
    Al27 = new G4Isotope("Al27", 13, 27, 26.9815386 *g/mole);

    Al = new G4Element("Aluminum", "Al", 1);
    Al->AddIsotope(Al27,  100.00*perCent);

    //make Zirconium isotopes and element
    Zr90 = new G4Isotope("Zr90", 40, 90, 89.9047026*g/mole);
    Zr91 = new G4Isotope("Zr91", 40, 91, 90.9056439*g/mole);
    Zr92 = new G4Isotope("Zr92", 40, 92, 91.9050386*g/mole);
    Zr94 = new G4Isotope("Zr94", 40, 94, 93.9063148*g/mole);
    Zr96 = new G4Isotope("Zr96", 40, 96, 95.908275*g/mole);



    Zr = new G4Element("Zirconium", "Zr", 5);
    Zr->AddIsotope(Zr90,  0.5075558873*perCent); //ATM%
    Zr->AddIsotope(Zr91,  0.1116101232*perCent);
    Zr->AddIsotope(Zr92,  0.1722780975*perCent);
    Zr->AddIsotope(Zr94,  0.1791179604*perCent);
    Zr->AddIsotope(Zr96,  0.0294379317*perCent);


    //make Zinc isotopes and element
    Zn64 = new G4Isotope("Zn64", 30, 64, 63.9291448*g/mole);
    Zn66 = new G4Isotope("Zn66", 30, 66, 65.9260347*g/mole);
    Zn67 = new G4Isotope("Zn67", 30, 67, 66.9271291*g/mole);
    Zn68 = new G4Isotope("Zn68", 30, 68, 67.9248459*g/mole);
    Zn70 = new G4Isotope("Zn70", 30, 70, 69.925325*g/mole);

    Zn = new G4Element("Zinc", "Zn", 5);
    Zn->AddIsotope(Zn64,  48.63*perCent);
    Zn->AddIsotope(Zn66, 27.90*perCent);
    Zn->AddIsotope(Zn67,  4.10*perCent);
    Zn->AddIsotope(Zn68,  18.75*perCent);
    Zn->AddIsotope(Zn70,  0.62*perCent);

    //make Tin isotopes and element
    Sn112 = new G4Isotope("Sn112", 50, 112, 111.904826*g/mole);
    Sn114 = new G4Isotope("Sn114", 50, 114, 113.902784*g/mole);
    Sn115 = new G4Isotope("Sn115", 50, 115, 114.903348*g/mole);
    Sn116 = new G4Isotope("Sn116", 50, 116, 115.901747*g/mole);
    Sn117 = new G4Isotope("Sn117", 50, 117, 116.902956*g/mole);
    Sn118 = new G4Isotope("Sn118", 50, 118, 117.901609*g/mole);
    Sn119 = new G4Isotope("Sn119", 50, 119, 118.903311*g/mole);
    Sn120 = new G4Isotope("Sn120", 50, 120, 119.9021991*g/mole);
    Sn122 = new G4Isotope("Sn122", 50, 122, 121.9034404*g/mole);
    Sn124 = new G4Isotope("Sn124", 50, 124, 123.9052743*g/mole);

    Sn = new G4Element("Tin", "Sn", 10);
    Sn->AddIsotope(Sn112,  0.97*perCent);
    Sn->AddIsotope(Sn114,  0.66*perCent);
    Sn->AddIsotope(Sn115,  0.34*perCent);
    Sn->AddIsotope(Sn116,  14.54*perCent);
    Sn->AddIsotope(Sn117,  7.68*perCent);
    Sn->AddIsotope(Sn118,  24.22*perCent);
    Sn->AddIsotope(Sn119,  8.59*perCent);
    Sn->AddIsotope(Sn120,  32.58*perCent);
    Sn->AddIsotope(Sn122,  4.63*perCent);
    Sn->AddIsotope(Sn124,  0.0*perCent);

    // Soudium Isotopes
    Na23 = new G4Isotope("Na23", 11, 23, 22.9897677*g/mole);

    // Naturally occuring Soudiium
    Na = new G4Element("Soudium", "Na", 1);
    Na->AddIsotope(Na23, 1.);

    // Gallium Isotopes
    Ga69 = new G4Isotope("Ga69", 31, 69, 68.9255809*g/mole);
    Ga71 = new G4Isotope("Ga71", 31, 71, 70.9247005*g/mole);

    // Naturally Occuring Gallium
    Ga = new G4Element("Gallium", "Ga", 2);
    Ga->AddIsotope(Ga69, 60.108*perCent);
    Ga->AddIsotope(Ga71, 39.892*perCent);


       //make Gadolinium isotopes and element
    Gd152 = new G4Isotope("Gd152", 64, 152, 151.919786*g/mole);
    Gd154 = new G4Isotope("Gd154", 64, 154, 153.920861*g/mole);
    Gd155 = new G4Isotope("Gd155", 64, 155, 154.922618*g/mole);
    Gd156 = new G4Isotope("Gd156", 64, 156, 155.922118*g/mole);
    Gd157 = new G4Isotope("Gd157", 64, 157, 156.923956*g/mole);
    Gd158 = new G4Isotope("Gd158", 64, 158, 157.924019*g/mole);
    Gd160 = new G4Isotope("Gd160", 64, 160, 159.927049*g/mole);


    Gd = new G4Element("Gadolinium", "Gd", 7);
    Gd->AddIsotope(Gd152,  0.20*perCent);
    Gd->AddIsotope(Gd154,  2.18*perCent);
    Gd->AddIsotope(Gd155,  14.80*perCent);
    Gd->AddIsotope(Gd156,  20.47*perCent);
    Gd->AddIsotope(Gd157,  15.65*perCent);
    Gd->AddIsotope(Gd158,  24.84*perCent);
    Gd->AddIsotope(Gd160,  21.86*perCent);


    //make titanium isotopes and element
    Ti46 = new G4Isotope("Ti46", 22, 46, 45.9526294*g/mole);
    Ti47 = new G4Isotope("Ti47", 22, 47, 46.9517640*g/mole);
    Ti48 = new G4Isotope("Ti48", 22, 48, 47.9479473*g/mole);
    Ti49 = new G4Isotope("Ti49", 22, 49, 48.9478711*g/mole);
    Ti50 = new G4Isotope("Ti50", 22, 50, 49.9447921*g/mole);

    Ti = new G4Element("Titanium", "Zn", 5);
    Ti->AddIsotope(Ti46,  8.25*perCent);
    Ti->AddIsotope(Ti47,  7.44*perCent);
    Ti->AddIsotope(Ti48,  73.72*perCent);
    Ti->AddIsotope(Ti49,  5.41*perCent);
    Ti->AddIsotope(Ti50,  5.18*perCent);

    //make Carbon isotopes and element
    C12 = new G4Isotope("C12", 6, 12, 12.0*g/mole);
    C13 = new G4Isotope("C13", 6, 13, 13.00335*g/mole);

    C = new G4Element("Carbon", "C", 2);
    C->AddIsotope(C12, 98.83*perCent);
    C->AddIsotope(C13,  1.07*perCent);


    // Make the uranium isotopes and element
    U234 = new G4Isotope("U234", 92, 234, 234.0410*g/mole);
    U235 = new G4Isotope("U235", 92, 235, 235.0439*g/mole);
    U236 = new G4Isotope("U236", 92, 236, 236.0456*g/mole);
    U238 = new G4Isotope("U238", 92, 238, 238.0508*g/mole);



    // Make hydrogen isotopes and elements
    H1 = new G4Isotope("H1", 1, 1, 1.0078*g/mole);
    Hydrogen = new G4Element("Hydrogen", "H", 1);
    Hydrogen->AddIsotope(H1, 100*perCent);


    D2 = new G4Isotope("D2", 1, 2, 2.014*g/mole);
    Deuterium = new G4Element("Deuterium", "D", 1);
    Deuterium->AddIsotope(D2, 100*perCent);

    // Make Oxygen isotopes and elements
    O16 = new G4Isotope("O16", 8, 16, 15.9949146*g/mole);
    O17 = new G4Isotope("O17", 8, 17, 16.9991312*g/mole);
   // O18 = new G4Isotope("O18", 8, 18, 17.9991603*g/mole);
    Oxygen = new G4Element("Oxygen", "O", 2);
    Oxygen->AddIsotope(O16, 99.963868927*perCent);
    Oxygen->AddIsotope(O17, 0.036131072*perCent);

    OxygenZr = new G4Element("Oxygen", "O", 1);
    OxygenZr->AddIsotope(O16, 0.688463*perCent);

    OxygenLEU = new G4Element("Oxygen", "O", 1);
    OxygenLEU->AddIsotope(O16, 100.0*perCent);

     // Making Oxygen for the heavy water
    /*OxygenMod = new G4Element("OxygenMod", "OM", 2);
    OxygenMod->AddIsotope(O16, 33.313111651*perCent);
    OxygenMod->AddIsotope(O17, 0.020000116*perCent);*/

    // Making Oxygen for the light water
    OxygenLW = new G4Element("OxygenLW", "OLW", 2);
    OxygenLW->AddIsotope(O16, 99.995998592*perCent);;
    OxygenLW->AddIsotope(O17, 0.004001407*perCent);


    // Making hydrogen for the lightwater
    Hydrogen = new G4Element("HydrogenLW", "HLW", 1);
    Hydrogen->AddIsotope(H1, 100*perCent);


    LEU = new G4Element("Low Enriched Uranium","LEU",4);
    LEU->AddIsotope(U234, 0.007432*perCent);
    LEU->AddIsotope(U235, 0.9583*perCent);
    LEU->AddIsotope(U236, 0.000239*perCent);
    LEU->AddIsotope(U238, 99.0341*perCent);


    // Make the LEU material
    LEUMat = new G4Material("U235 Material", 10.52*g/cm3, 2,kStateSolid, 299.51*kelvin);
    LEUMat->AddElement(LEU,88.146875681*perCent);
    LEUMat->AddElement(OxygenLEU,11.853119788*perCent);



    // Create H20 material
    H2O = new G4Material("Light Water", 0.99745642056*g/cm3, 2, kStateLiquid);
    H2O->AddElement(OxygenLW, 1);
    H2O->AddElement(Hydrogen, 2);

    D2O = new G4Material("Heavy Water", 1.10480511492*g/cm3, 2, kStateLiquid);
    D2O->AddElement(Oxygen, 1);
    D2O->AddElement(Deuterium, 2);

//    Graphite = new G4Material("Graphite", 6., 12.0107*g/mole, 1.64*g/cm3);
    Graphite = new G4Material("Graphite", 1.64*g/cm3, 5, kStateSolid);
    Graphite->AddElement(Li, 1.7e-5*perCent);
    Graphite->AddElement(B, 3.e-5*perCent);
    Graphite->AddElement(C, 99.99697797*perCent);
    Graphite->AddElement(V, 0.00300031*perCent);
    Graphite->AddElement(Gd, 2.e-5*perCent);



    // Make Argon
    G4Element* Ar = new G4Element("Argon", "Ar", 18., 39.948*g/mole);
    // Make Argon
    G4Element* N = new G4Element("Nitrogen", "N", 7., 14.01*g/mole);




    //Create Aluminum57S (Reactor Calandria)
    Aluminum57S = new G4Material("Aluminuum 57S", 2.7*g/cm3, 8, kStateSolid);
    Aluminum57S->AddElement(Al, 96.7*perCent);
    Aluminum57S->AddElement(Si, 0.25*perCent);
    Aluminum57S->AddElement(Fe, 0.4*perCent);
    Aluminum57S->AddElement(Cu, 0.1*perCent);
    Aluminum57S->AddElement(Mn, 0.1*perCent);
    Aluminum57S->AddElement(Mg, 2.2*perCent);
    Aluminum57S->AddElement(Cr, 0.15*perCent);
    Aluminum57S->AddElement(Zn, 0.1*perCent);

    //Create AlPresT (pressure Tube)
//    AlPresT = new G4Material("Aluminuum 6061", 2.712631*g/cm3, 8, kStateSolid);
    AlPresT = new G4Material("Aluminuum 6061", 2.712631*g/cm3, 8, kStateSolid);

    AlPresT->AddElement(Al, 99.1244424*perCent);
    AlPresT->AddElement(Si, 0.5922414*perCent);
    AlPresT->AddElement(Fe, 0.1211379*perCent);
    AlPresT->AddElement(Cu, 0.0018171*perCent);
    AlPresT->AddElement(Mn, 0.0383626*perCent);
    //AlPresT->AddElement(Mg, 0.7000*perCent);
    AlPresT->AddElement(Cr, 0.1211405*perCent);
    AlPresT->AddElement(Li, 0.00075712*perCent);
    AlPresT->AddElement(B, 0.00010095*perCent);
    //AlPresT->AddElement(Zn, 0.0230*perCent);
    //AlPresT->AddElement(Na, 0.0090*perCent);
    //AlPresT->AddElement(Ga, 0.0120*perCent);
    //AlPresT->AddElement(Ti, 0.0110*perCent);

    //Create AlCalT (calandria Tube)
//    AlCalT = new G4Material("Aluminuum 6063", 2.684951*g/cm3, 8, kStateSolid);
    AlCalT = new G4Material("Aluminuum 6063", 2.684951*g/cm3, 8, kStateSolid);
    AlCalT->AddElement(Al, 99.18675267*perCent);
    AlCalT->AddElement(Si, 0.509640251*perCent);
    AlCalT->AddElement(Fe, 0.241396625*perCent);
    AlCalT->AddElement(Li, 0.00754387*perCent);
    AlCalT->AddElement(B, 0.000100586*perCent);
    //AlCalT->AddElement(Cu, 0.0590*perCent);
    AlCalT->AddElement(Mn, 0.041228175*perCent);
    //AlCalT->AddElement(Mg, 0.5400*perCent);
    //AlCalT->AddElement(Cr, 0.0100*perCent);
    //AlCalT->AddElement(Zn, 0.0340*perCent);
    //AlCalT->AddElement(Na, 0.0170*perCent);
    AlCalT->AddElement(Gd, 0.000010059*perCent);
    AlCalT->AddElement(Ti, 0.041228175*perCent);


    Moderator = new G4Material("Moderator", 1.102597*g/cm3, 2, kStateLiquid, 299.51*kelvin);
    Moderator->AddMaterial(D2O, 98.705*perCent);
    Moderator->AddMaterial(H2O,  1.295*perCent);

    //Create Annulus Gas
    AnnulusGas = new G4Material("AnnulusGas", 0.0012*g/cm3, 2, kStateGas/*,
								448.72*kelvin*/);
    AnnulusGas->AddElement(C,27.11*perCent);
    AnnulusGas->AddElement(Oxygen,72.89*perCent);


    Zr4 = new G4Material("Zircaloy-4", 6.55*g/cm3, 4, kStateSolid);
    Zr4->AddElement(Oxygen, 0.12*perCent);
    Zr4->AddElement(CrZr, 0.11*perCent);
    Zr4->AddElement(FeZr, 0.22*perCent);
    Zr4->AddElement(Zr, 99.58*perCent);

    // Make Air
    Air = new G4Material("Air", 1.29*mg/cm3, 5, kStateGas);
    Air->AddElement(N, 74.74095914*perCent);
    Air->AddElement(Oxygen, 23.49454694*perCent);
    Air->AddElement(Ar, 1.274547311*perCent);
    Air->AddElement(Li, 0.474350981*perCent);
    Air->AddElement(C, 0.015595629*perCent);
    //Air->AddElement(Hydrogen, 0.009895657);





    // Add materials to the map indexed by either ZA (format ZZAAA or ZZ)
    // For composite materials:  world is 0, heavy water is 1, UHW is 2
    matMap["World"] = World;
    matMap["LEUMat"] = LEUMat;
    matMap["Graphite"] = Graphite;
    matMap["Al57S"] = Aluminum57S;
    matMap["AlPresT"] = AlPresT;
    matMap["AlCalT"] = AlCalT;
    matMap["Zr4"] = Zr4;
    matMap["Air"] = Air;
    //matMap["RUMat"] = RUMat;
    matMap["Moderator"] = Moderator;
    matMap["Coolant"] = H2O;
  /*  G4Isotope *U234, *U235, *U238, *U236, *D2,   *O16, *O17,
              *Fe54, *Fe56, *Fe57, *Fe58, *Cr50, *Cr52, *Cr53, *Cr54,
              *Si28, *Si29, *Si30, *Cu63, *Cu65, *Mn55, *Mg24,
              *Mg25, *Mg26, *Zn64, *Zn66, *Zn67, *Zn68, *Zn70,
	          *Al27, *Ti46, *Ti47, *Ti48, *Ti49, *Ti50, *Na23,
	          *Ga69, *Ga71, *H1,   *C12,  *C13,  *Zr90, *Zr91,
	          *Zr92, *Zr94, *Zr96, *Sn112, *Sn114, *Sn115, *Sn116,
	          *Sn117, *Sn118, *Sn119, *Sn120, *Sn122, *Sn124,
	          *Ca40, *Ca42, *Ca43, *Ca44, *Ca46, *Ca48, *B10, *B11,
	          *Li6, *Li7, *Gd152,*Gd154, *Gd155, *Gd156, *Gd157,
              *Gd158, *Gd160,*V50, *V51;
    G4Element *Oxygen, *Deuterium, *LEU,
              *Cr, *Fe, *Si, *Cu, *Mn, *Mg, *Zn, *Al,
              *Ti, *Na, *Ga, *Hydrogen, *C, *Zr, *Sn, *Ca, *RU,
              *B, *Li, *Gd, *V, *OxygenMod, *HydrogenMod, *OxygenRU,
              *FeAl, *CuAl,*FeZr, *CrZr, *OxygenZr,
              *OxygenLEU, *HydrogenLW, *OxygenLW;
    G4Material *World, *LEUMat, *HeavyWater,
	           *Aluminum57S, *AlPresT, *AlCalT, *H2O,
	           *Coolant, *AnnulusGas, *Zr4, *Air, *RUMat, *Moderator, *Graphite;

    // Create the world environment
    World = new G4Material("Galactic", 1, 1, 1.e-25*g/cm3, kStateGas,
                                      2.73*kelvin, 3.e-18*pascal);

    //make Calcium isotopes and element
    Ca40 = new G4Isotope("Ca40", 20, 40, 39.9625906*g/mole);
    Ca42 = new G4Isotope("Ca42", 20, 42, 41.9586176*g/mole);
    Ca43 = new G4Isotope("Ca43", 20, 43, 42.9587662*g/mole);
    Ca44 = new G4Isotope("Ca44", 20, 44, 43.9554806*g/mole);
    Ca46 = new G4Isotope("Ca46", 20, 46, 45.953689*g/mole);
    Ca48 = new G4Isotope("Ca48", 20, 48, 47.952533*g/mole);

    Ca = new G4Element("Calcium", "Ca", 6);
    Ca->AddIsotope(Ca40,  96.941*perCent);
    Ca->AddIsotope(Ca42,  0.647*perCent);
    Ca->AddIsotope(Ca43,  0.135*perCent);
    Ca->AddIsotope(Ca44,  2.086*perCent);
    Ca->AddIsotope(Ca46,  0.004*perCent);
    Ca->AddIsotope(Ca48,  0.187*perCent);

    //make Boron isotopes and element
    B10 = new G4Isotope("B10", 5, 10, 10.012937*g/mole);
    B11 = new G4Isotope("B11", 5, 11, 11.009305*g/mole);

    B = new G4Element("Boron", "B", 2);
    B->AddIsotope(B10,  19.9*perCent);
    B->AddIsotope(B11,  80.1*perCent);

    //make Lithium isotopes and element
    Li6 = new G4Isotope("Li6", 3, 6, 6.0151223*g/mole);
    Li7 = new G4Isotope("Li7", 3, 7, 7.0160040*g/mole);

    Li = new G4Element("Lithium", "Li", 2);
    Li->AddIsotope(Li6,  7.59 *perCent);
    Li->AddIsotope(Li7,  92.41*perCent);

    //make Vanadium isotopes and element
    V50 = new G4Isotope("V50", 23, 50, 49.9471609 *g/mole);
    V51 = new G4Isotope("V51", 23, 51, 50.9439617 *g/mole);

    V = new G4Element("Vanadium", "V", 2);
    V->AddIsotope(V50,  0.250 *perCent);
    V->AddIsotope(V51,  99.750*perCent);




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

    CrZr = new G4Element("Chromium", "Cr", 4);
    CrZr->AddIsotope(Cr50,  7.244845E-3*perCent);
    CrZr->AddIsotope(Cr52,  0.145123227*perCent);
    CrZr->AddIsotope(Cr53,  0.016770705*perCent);
    CrZr->AddIsotope(Cr54,  0.007392584*perCent);


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

    //make iron element for Aluminium material in ZED-2
    FeAl = new G4Element("Iron", "Fe", 4);
    FeAl->AddIsotope(Fe54,  0.02340*perCent);
    FeAl->AddIsotope(Fe56,  0.36700*perCent);
    FeAl->AddIsotope(Fe57,  0.00848*perCent);
    FeAl->AddIsotope(Fe58,  0.00112*perCent);

    //make iron element for Aluminium material in ZED-2
    FeZr = new G4Element("Iron", "Fe", 4);
    FeZr->AddIsotope(Fe54,  1.84154E-2*perCent);
    FeZr->AddIsotope(Fe56,  3.022533E-1*perCent);
    FeZr->AddIsotope(Fe57,  7.037927E-3*perCent);
    FeZr->AddIsotope(Fe58,  1.02304E-3*perCent);
    //make Silicon isotopes and element
    Si28 = new G4Isotope("Si28", 14, 28, 27.9769271*g/mole);
    Si29 = new G4Isotope("Si29", 14, 29, 28.9764949*g/mole);
    Si30 = new G4Isotope("Si30", 14, 30, 29.9737707*g/mole);

    Si = new G4Element("Silicon", "Si", 3);
    Si->AddIsotope(Si28,  92.23*perCent);
    Si->AddIsotope(Si29,  4.67*perCent);
    Si->AddIsotope(Si30,  3.1*perCent);

    //make Magnesium isotopes and element
    Mg24 = new G4Isotope("Mg24", 12, 24, 23.9850423*g/mole);
    Mg25 = new G4Isotope("Mg25", 12, 25, 24.9858374*g/mole);
    Mg26 = new G4Isotope("Mg26", 12, 26, 25.9825937 *g/mole);

    Mg = new G4Element("Magnesium", "Mg", 3);
    Mg->AddIsotope(Mg24,  78.99*perCent);
    Mg->AddIsotope(Mg25,  10.00*perCent);
    Mg->AddIsotope(Mg26,  11.01*perCent);

    //make Manganese isotopes and element
    Mn55 = new G4Isotope("Mn55", 25, 55, 54.9380471*g/mole);

    Mn = new G4Element("Manganese", "Mn", 1);
    Mn->AddIsotope(Mn55,  100.00*perCent);

    //make Copper isotopes and element
    Cu63 = new G4Isotope("Cu63", 29, 63, 62.9295989*g/mole);
    Cu65 = new G4Isotope("Cu65", 29, 65, 64.9277929 *g/mole);

    Cu = new G4Element("Copper", "Cu", 2);
    Cu->AddIsotope(Cu63,  69.17*perCent);
    Cu->AddIsotope(Cu65,  30.83*perCent);

    //make copper for Al
    CuAl = new G4Element("Copper", "Cu", 2);
    CuAl->AddIsotope(Cu63,  0.01383*perCent);
    CuAl->AddIsotope(Cu65,  0.00617*perCent);

    //make Aluminum isotopes and element
    Al27 = new G4Isotope("Al27", 13, 27, 26.9815386 *g/mole);

    Al = new G4Element("Aluminum", "Al", 1);
    Al->AddIsotope(Al27,  100.00*perCent);

    //make Zirconium isotopes and element
    Zr90 = new G4Isotope("Zr90", 40, 90, 89.9047026*g/mole);
    Zr91 = new G4Isotope("Zr91", 40, 91, 90.9056439*g/mole);
    Zr92 = new G4Isotope("Zr92", 40, 92, 91.9050386*g/mole);
    Zr94 = new G4Isotope("Zr94", 40, 94, 93.9063148*g/mole);
    Zr96 = new G4Isotope("Zr96", 40, 96, 95.908275*g/mole);



    Zr = new G4Element("Zirconium", "Zr", 5);
    Zr->AddIsotope(Zr90,  50.1407*perCent); //ATM%
    Zr->AddIsotope(Zr91,  11.0258*perCent);
    Zr->AddIsotope(Zr92,  17.0191*perCent);
    Zr->AddIsotope(Zr94,  17.6948*perCent);
    Zr->AddIsotope(Zr96,  2.90813*perCent);

    //make Zinc isotopes and element
    Zn64 = new G4Isotope("Zn64", 30, 64, 63.9291448*g/mole);
    Zn66 = new G4Isotope("Zn66", 30, 66, 65.9260347*g/mole);
    Zn67 = new G4Isotope("Zn67", 30, 67, 66.9271291*g/mole);
    Zn68 = new G4Isotope("Zn68", 30, 68, 67.9248459*g/mole);
    Zn70 = new G4Isotope("Zn70", 30, 70, 69.925325*g/mole);

    Zn = new G4Element("Zinc", "Zn", 5);
    Zn->AddIsotope(Zn64,  48.63*perCent);
    Zn->AddIsotope(Zn66, 27.90*perCent);
    Zn->AddIsotope(Zn67,  4.10*perCent);
    Zn->AddIsotope(Zn68,  18.75*perCent);
    Zn->AddIsotope(Zn70,  0.62*perCent);

    //make Tin isotopes and element
    Sn112 = new G4Isotope("Sn112", 50, 112, 111.904826*g/mole);
    Sn114 = new G4Isotope("Sn114", 50, 114, 113.902784*g/mole);
    Sn115 = new G4Isotope("Sn115", 50, 115, 114.903348*g/mole);
    Sn116 = new G4Isotope("Sn116", 50, 116, 115.901747*g/mole);
    Sn117 = new G4Isotope("Sn117", 50, 117, 116.902956*g/mole);
    Sn118 = new G4Isotope("Sn118", 50, 118, 117.901609*g/mole);
    Sn119 = new G4Isotope("Sn119", 50, 119, 118.903311*g/mole);
    Sn120 = new G4Isotope("Sn120", 50, 120, 119.9021991*g/mole);
    Sn122 = new G4Isotope("Sn122", 50, 122, 121.9034404*g/mole);
    Sn124 = new G4Isotope("Sn124", 50, 124, 123.9052743*g/mole);

    Sn = new G4Element("Tin", "Sn", 10);
    Sn->AddIsotope(Sn112,  0.97*perCent);
    Sn->AddIsotope(Sn114,  0.66*perCent);
    Sn->AddIsotope(Sn115,  0.34*perCent);
    Sn->AddIsotope(Sn116,  14.54*perCent);
    Sn->AddIsotope(Sn117,  7.68*perCent);
    Sn->AddIsotope(Sn118,  24.22*perCent);
    Sn->AddIsotope(Sn119,  8.59*perCent);
    Sn->AddIsotope(Sn120,  32.58*perCent);
    Sn->AddIsotope(Sn122,  4.63*perCent);
    Sn->AddIsotope(Sn124,  0.0*perCent);

    // Soudium Isotopes
    Na23 = new G4Isotope("Na23", 11, 23, 22.9897677*g/mole);

    // Naturally occuring Soudiium
    Na = new G4Element("Soudium", "Na", 1);
    Na->AddIsotope(Na23, 1.);

    // Gallium Isotopes
    Ga69 = new G4Isotope("Ga69", 31, 69, 68.9255809*g/mole);
    Ga71 = new G4Isotope("Ga71", 31, 71, 70.9247005*g/mole);

    // Naturally Occuring Gallium
    Ga = new G4Element("Gallium", "Ga", 2);
    Ga->AddIsotope(Ga69, 60.108*perCent);
    Ga->AddIsotope(Ga71, 39.892*perCent);


       //make Gadolinium isotopes and element
    Gd152 = new G4Isotope("Gd152", 64, 152, 151.919786*g/mole);
    Gd154 = new G4Isotope("Gd154", 64, 154, 153.920861*g/mole);
    Gd155 = new G4Isotope("Gd155", 64, 155, 154.922618*g/mole);
    Gd156 = new G4Isotope("Gd156", 64, 156, 155.922118*g/mole);
    Gd157 = new G4Isotope("Gd157", 64, 157, 156.923956*g/mole);
    Gd158 = new G4Isotope("Gd158", 64, 158, 157.924019*g/mole);
    Gd160 = new G4Isotope("Gd160", 64, 160, 159.927049*g/mole);


    Gd = new G4Element("Gadolinium", "Gd", 7);
    Gd->AddIsotope(Gd152,  0.20*perCent);
    Gd->AddIsotope(Gd154,  2.18*perCent);
    Gd->AddIsotope(Gd155,  14.80*perCent);
    Gd->AddIsotope(Gd156,  20.47*perCent);
    Gd->AddIsotope(Gd157,  15.65*perCent);
    Gd->AddIsotope(Gd158,  24.84*perCent);
    Gd->AddIsotope(Gd160,  21.86*perCent);


    //make titanium isotopes and element
    Ti46 = new G4Isotope("Ti46", 22, 46, 45.9526294*g/mole);
    Ti47 = new G4Isotope("Ti47", 22, 47, 46.9517640*g/mole);
    Ti48 = new G4Isotope("Ti48", 22, 48, 47.9479473*g/mole);
    Ti49 = new G4Isotope("Ti49", 22, 49, 48.9478711*g/mole);
    Ti50 = new G4Isotope("Ti50", 22, 50, 49.9447921*g/mole);

    Ti = new G4Element("Titanium", "Zn", 5);
    Ti->AddIsotope(Ti46,  8.25*perCent);
    Ti->AddIsotope(Ti47,  7.44*perCent);
    Ti->AddIsotope(Ti48,  73.72*perCent);
    Ti->AddIsotope(Ti49,  5.41*perCent);
    Ti->AddIsotope(Ti50,  5.18*perCent);

    //make Carbon isotopes and element
    C12 = new G4Isotope("C12", 6, 12, 12.0*g/mole);
    C13 = new G4Isotope("C13", 6, 13, 13.00335*g/mole);

    C = new G4Element("Carbon", "C", 2);
    C->AddIsotope(C12, 98.83*perCent);
    C->AddIsotope(C13,  1.07*perCent);


    // Make the uranium isotopes and element
    U234 = new G4Isotope("U234", 92, 234, 234.0410*g/mole);
    U235 = new G4Isotope("U235", 92, 235, 235.0439*g/mole);
    U236 = new G4Isotope("U236", 92, 236, 236.0456*g/mole);
    U238 = new G4Isotope("U238", 92, 238, 238.0508*g/mole);



    // Make heavy water isotopes and elements
    H1 = new G4Isotope("H1", 1, 1, 1.0078*g/mole);
    Hydrogen = new G4Element("Hydrogen", "H", 1);
    Hydrogen->AddIsotope(H1, 100*perCent);


    D2 = new G4Isotope("D2", 1, 2, 2.014*g/mole);
    Deuterium = new G4Element("Deuterium", "D", 1);
    Deuterium->AddIsotope(D2, 100*perCent);


    O16 = new G4Isotope("O16", 8, 16, 15.9949146*g/mole);
    O17 = new G4Isotope("O17", 8, 17, 16.9991312*g/mole);
   // O18 = new G4Isotope("O18", 8, 18, 17.9991603*g/mole);
    Oxygen = new G4Element("Oxygen", "O", 1);
    Oxygen->AddIsotope(O16, 100*perCent);

    OxygenZr = new G4Element("Oxygen", "O", 1);
    OxygenZr->AddIsotope(O16, 0.688463*perCent);


    OxygenRU = new G4Element("OxygenRU", "O", 2);
    OxygenRU->AddIsotope(O16, 11.843718*perCent);
    OxygenRU->AddIsotope(O17, 0.004502*perCent);


 // Making Oxygen for the heavy water
    OxygenMod = new G4Element("OxygenMod", "OM", 2);
    OxygenMod->AddIsotope(O16, 33.313111651*perCent);;
    OxygenMod->AddIsotope(O17, 0.020000116*perCent);


// Making hydrogen for the hwavy water
    HydrogenMod = new G4Element("HydrogenMod", "HM", 2);
    HydrogenMod->AddIsotope(H1, 0.958387035*perCent);
    HydrogenMod->AddIsotope(D2, 65.708501196*perCent);


     // Making Oxygen for the light water
    OxygenLW = new G4Element("OxygenLW", "OLW", 2);
    OxygenLW->AddIsotope(O16, 3.333194E+1*perCent);;
    OxygenLW->AddIsotope(O17, 1.3338E-3*perCent);


// Making hydrogen for the lightwater
    HydrogenLW = new G4Element("HydrogenLW", "HLW", 1);
    HydrogenLW->AddIsotope(H1, 6.669057E+1*perCent);


    LEU = new G4Element("Low Enriched Uranium","LEU",4);
    LEU->AddIsotope(U234, 0.007432*perCent);
    LEU->AddIsotope(U235, 0.9583*perCent);
    LEU->AddIsotope(U236, 0.000239*perCent);
    LEU->AddIsotope(U238, 99.0341*perCent);



    OxygenLEU = new G4Element("Oxygen", "O", 1);
    OxygenLEU->AddIsotope(O16, 100.0*perCent);

    // Make Recovered Uranium
    RU = new G4Element("Recovered Uranium","RU",4);
    RU->AddIsotope(U234, 0.01308*perCent);
    RU->AddIsotope(U235, 0.8476*perCent);
    RU->AddIsotope(U236, 0.2011*perCent);
    RU->AddIsotope(U238, 87.09*perCent);


    // Make the LEU material
    LEUMat = new G4Material("U235 Material", 10.52*g/cm3, 2,kStateSolid, 299.51*kelvin);
    LEUMat->AddElement(LEU,1);
    LEUMat->AddElement(OxygenLEU,2);


    // Make the RUfuel material
    RUMat = new G4Material("RU Material", 10.45*g/cm3, 2, kStateSolid, 298.55*kelvin);

    RUMat->AddElement(RU,1);
    RUMat->AddElement(OxygenRU,1);

    // Create H20 material
    H2O = new G4Material("Light Water", 0.99745642056*g/cm3, 2, kStateLiquid);
    H2O->AddElement(OxygenLW, 1);
    H2O->AddElement(HydrogenLW, 1);


    // Make the heavy water material
    HeavyWater = new G4Material("Heavy Water", 1.10480511492*g/cm3, 2, kStateLiquid);
    HeavyWater->AddElement(HydrogenMod, 1);
    HeavyWater->AddElement(OxygenMod, 1);



//    Graphite = new G4Material("Graphite", 6., 12.0107*g/mole, 1.64*g/cm3);
    Graphite = new G4Material("Graphite", 1.64*g/cm3, 5, kStateSolid);
    Graphite->AddElement(Li, 1.7e-5*perCent);
    Graphite->AddElement(B, 3.e-5*perCent);
    Graphite->AddElement(C, 99.99697797*perCent);
    Graphite->AddElement(V, 0.00300031*perCent);
    Graphite->AddElement(Gd, 2.e-5*perCent);



    // Make Argon
    G4Element* Ar = new G4Element("Argon", "Ar", 18., 39.948*g/mole);
    // Make Argon
    G4Element* N = new G4Element("Nitrogen", "N", 7., 14.01*g/mole);




    //Create Aluminum57S (Reactor Calandria)
    Aluminum57S = new G4Material("Aluminuum 57S", 2.7*g/cm3, 8, kStateSolid);
    Aluminum57S->AddElement(Al, 96.7*perCent);
    Aluminum57S->AddElement(Si, 0.25*perCent);
    Aluminum57S->AddElement(Fe, 0.4*perCent);
    Aluminum57S->AddElement(Cu, 0.1*perCent);
    Aluminum57S->AddElement(Mn, 0.1*perCent);
    Aluminum57S->AddElement(Mg, 2.2*perCent);
    Aluminum57S->AddElement(Cr, 0.15*perCent);
    Aluminum57S->AddElement(Zn, 0.1*perCent);

    //Create AlPresT (pressure Tube)
//    AlPresT = new G4Material("Aluminuum 6061", 2.712631*g/cm3, 8, kStateSolid);
    AlPresT = new G4Material("Aluminuum 6061", 2.712631*g/cm3, 8, kStateSolid);

    AlPresT->AddElement(Al, 99.1244424*perCent);
    AlPresT->AddElement(Si, 0.5922414*perCent);
    AlPresT->AddElement(Fe, 0.1211379*perCent);
    AlPresT->AddElement(Cu, 0.0018171*perCent);
    AlPresT->AddElement(Mn, 0.0383626*perCent);
    //AlPresT->AddElement(Mg, 0.7000*perCent);
    AlPresT->AddElement(Cr, 0.1211405*perCent);
    AlPresT->AddElement(Li, 0.00075712*perCent);
    AlPresT->AddElement(B, 0.00010095*perCent);
    //AlPresT->AddElement(Zn, 0.0230*perCent);
    //AlPresT->AddElement(Na, 0.0090*perCent);
    //AlPresT->AddElement(Ga, 0.0120*perCent);
    //AlPresT->AddElement(Ti, 0.0110*perCent);

    //Create AlCalT (calandria Tube)
//    AlCalT = new G4Material("Aluminuum 6063", 2.684951*g/cm3, 8, kStateSolid);
    AlCalT = new G4Material("Aluminuum 6063", 2.684951*g/cm3, 8, kStateSolid);
    AlCalT->AddElement(Al, 99.18675267*perCent);
    AlCalT->AddElement(Si, 0.509640251*perCent);
    AlCalT->AddElement(Fe, 0.241396625*perCent);
    AlCalT->AddElement(Li, 0.00754387*perCent);
    AlCalT->AddElement(B, 0.000100586*perCent);
    //AlCalT->AddElement(Cu, 0.0590*perCent);
    AlCalT->AddElement(Mn, 0.041228175*perCent);
    //AlCalT->AddElement(Mg, 0.5400*perCent);
    //AlCalT->AddElement(Cr, 0.0100*perCent);
    //AlCalT->AddElement(Zn, 0.0340*perCent);
    //AlCalT->AddElement(Na, 0.0170*perCent);
    AlCalT->AddElement(Gd, 0.000010059*perCent);
    AlCalT->AddElement(Ti, 0.041228175*perCent);


    // Create Coolant
    Coolant = new G4Material("Coolant", 0.8074*g/cm3, 2, kStateLiquid);
    Coolant->AddMaterial(HeavyWater, 99.3777*perCent);
    Coolant->AddMaterial(H2O,  0.6223*perCent);


    Moderator = new G4Material("Moderator", 1.102597*g/cm3, 2, kStateLiquid,299.51*kelvin);
    Moderator->AddMaterial(HeavyWater, 98.705*perCent);
    Moderator->AddMaterial(H2O,  1.295*perCent);

    //Create Annulus Gas
    AnnulusGas = new G4Material("AnnulusGas", 0.0012*g/cm3, 2, kStateGas);
    AnnulusGas->AddElement(C,27.11*perCent);
    AnnulusGas->AddElement(Oxygen,72.89*perCent);


    Zr4 = new G4Material("Zircaloy-4", 6.55*g/cm3, 4, kStateSolid);
    Zr4->AddElement(Oxygen, 0.12*perCent);
    Zr4->AddElement(CrZr, 0.11*perCent);
    Zr4->AddElement(FeZr, 0.22*perCent);
    Zr4->AddElement(Zr, 99.58*perCent);

    // Make Air
    Air = new G4Material("Air", 1.29*mg/cm3, 5, kStateGas);
    Air->AddElement(N, 74.74095914*perCent);
    Air->AddElement(Oxygen, 23.49454694*perCent);
    Air->AddElement(Ar, 1.274547311*perCent);
    Air->AddElement(Li, 0.474350981*perCent);
    Air->AddElement(C, 0.015595629*perCent);
    //Air->AddElement(Hydrogen, 0.009895657);





    // Add materials to the map indexed by either ZA (format ZZAAA or ZZ)
    // For composite materials:  world is 0, heavy water is 1, UHW is 2
    matMap["World"] = World;
    matMap["LEUMat"] = LEUMat;
    matMap["Heavy Water"] = HeavyWater;
    matMap["Graphite"] = Graphite;
    matMap["Al57S"] = Aluminum57S;
    matMap["AlPresT"] = AlPresT;
    matMap["AlCalT"] = AlCalT;
    matMap["Coolant"] = Coolant;
    matMap["AnnulusGas"] = AnnulusGas;
    matMap["Zr4"] = Zr4;
    matMap["Air"] = Air;
    matMap["RUMat"] = RUMat;
    matMap["Moderator"] = Moderator;*/


    matChanged = false;

    return;
}

