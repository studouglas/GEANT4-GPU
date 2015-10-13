
#include "SLOWPOKEConstructor.hh"

SLOWPOKEConstructor::SLOWPOKEConstructor()
: StorkVWorldConstructor(), ZirconiumLogical1(0), ZirconiumLogical2(0), ZirconiumLogical3(0), AirGapLogical(0),
FuelRodLogical(0), ReflectorLogical(0), D2OContainerLogical(0),D2OLogical(0),
contRodZirLogical(0), contRodAlumLogical(0), contRodCadLogical(0), contRodCentLogical(0),
insAlumLogical(0), insBeamLogical(0), outSmallAlumLogical(0), outLargeAlumLogical(0), cadLinLogical(0),
outSmallBeamLogical(0), outLargeBeamLogical(0), alumShellLogical(0), cellLogical(0)
{
	// Set default member variables (from file or default values)
    //Rod in (20.66cm - Rod In), (0.00cm - Rod Out)
    ControlRodPosition = 0.00*cm ;

  //  ControlRodPosition = 20.66*cm ;

    //Set initial T.H. properties of fuel and geometry.
    FuelRadius = 0.2064*cm;
	FuelTemp = (25 + 273.15)*kelvin;
    FuelDensity = 10.6*g/cm3;

    //Initialize property vectors.
    for(G4int i = 0; i<34; i++){
        FuelDensities[i] = FuelDensity;
        FuelTemperatures[i] = FuelTemp;
        FuelRadii[i] = FuelRadius;
    }


	// Set up variable property map
	variablePropMap[MatPropPair(controlrod,position)] = &ControlRodPosition;
    variablePropMap[MatPropPair(fuel,temperature)] = &FuelTemp;
    variablePropMap[MatPropPair(moderator,temperature)] = &moderatorTemp;

}

SLOWPOKEConstructor::~SLOWPOKEConstructor()
{
	// Delete visualization attributes
	delete ZirconiumAtt1;
	delete ZirconiumAtt2;
	delete ZirconiumAtt3;
	delete AirGapAtt;
	delete FuelRodAtt;
	delete ReflectorAtt;
	delete D2OContainerAtt;
	delete D2OAtt;
    delete contRodZirVisAtt;
    delete contRodAlumVisAtt;
    delete contRodCadVisAtt;
    delete contRodCentVisAtt;
    delete insAlumVisAtt;
    delete insBeamVisAtt;
    delete outSmallAlumVisAtt;
    delete outLargeAlumVisAtt;
    delete cadLinTubeVisAtt;
    delete outSmallBeamVisAtt;
    delete outLargeBeamVisAtt;
    delete alumShellVisAtt;
    delete cellVisAtt;

    //if(FissionMap) { delete FissionMap; }
}


// ConstructWorld()
// Construct the geometry and materials of the Guillaume Reactor.
G4VPhysicalVolume* SLOWPOKEConstructor::ConstructWorld()
{
	G4SolidStore* theSolids = G4SolidStore::GetInstance();

	// Set static dimensions of all the geometries
	// Note the format for cylinder dimensions is (inner radius, outer radius, height)

	// Reactor pool dimensions
	reactorDim = G4ThreeVector(0., 133.0*cm, 564.0*cm);

	// World dimensions
	G4double buffer = 1.0*cm;
	encWorldDim = 2.0*G4ThreeVector(reactorDim[1]+buffer,reactorDim[1]+buffer, reactorDim[2]/2+buffer);

    // Reflector dimensions
    G4double refAnnDim[3] = {11.049*cm, 21.2344*cm, 22.748*cm};
    G4double refBottomDim[3] = {0.0*cm, 16.113125*cm, 10.16*cm};
    G4double refTopDim[3] = {1.3890625*cm, 12.065*cm, 0.15875*cm};

    // D2O colum container and heavy water
    G4double D20ContainerDim[3] = {21.2344*cm, 30*cm, 22.748*cm};
    G4double D20Dim[3] = {22.2344*cm, 29*cm, 20.6975*cm};

    // Beamtube dimensions
    G4double smallBTubeDim[3] = {0.0*cm, 1.40208*cm, 515.332*cm};
    G4double smallLongBTubeDim[3] = {0.0*cm, 1.40208*cm, 522.332*cm};
    G4double largeBTubeDim[3] = {0.0*cm, 1.6*cm, 515.332*cm};

    // Aluminum tube dimensions
    G4double smallAlumTubeDim[3] = {0.0*cm, 1.56718*cm, 515.332*cm};
    G4double smallLongAlumTubeDim[3] = {0.0*cm, 1.56718*cm, 522.332*cm};
    G4double largeAlumTubeDim[3] = {0.0*cm, 1.905*cm, 515.332*cm};
    G4double alumTubePos[3]={14.56182*cm, 0.4*CLHEP::pi, 0.};
    G4double outAlumTubePos[3]={24.0*cm, 0.4*CLHEP::pi, 0.2*CLHEP::pi};

    // Cadmium lining
    G4double cadLinTubeDim[3] = {1.56718*cm, 1.61798*cm, 22.748*cm};

    // Aluminium Reactor Shell
    G4double alumShellTubeDim[3] = {30.0*cm, 31.0*cm, 541.0*cm};
    G4double alumShellPlateDim[3] = {0.0*cm, 31.0*cm, 1*cm};

    // Control Rod (Aluminum shell, cadmium rod and air gap)
    G4double contRodCentTubeDim[3] = {0.0*cm, 0.09652*cm, 24.76*cm};
    G4double contRodCadTubeDim[3] = {0.0*cm, 0.14732*cm, 24.76*cm};
    G4double contRodAlumTubeDim[3] = {0.0*cm, 0.62357*cm, 40.64*cm};
    G4double contRodZirTubeDim[3] = {1.229*cm, 1.331*cm, 23.2335*cm};

    /* Begining Of Reactor Core Dimensions */
	// Zirconium lower/upper plate dimensions
	G4double LowerZrDim[5] = {1.331*cm, 11.049*cm, 0.279*cm, CLHEP::pi/3, CLHEP::pi/3};
	G4double UpperZrDim[5] = {LowerZrDim[0], LowerZrDim[1], LowerZrDim[2], LowerZrDim[3], LowerZrDim[4]};

	// Zirconium holes lower/upper plate
	G4double WaterHolesLowerZrDim[3] = {0., 0.262*cm, LowerZrDim[2]+1*mm};
    G4double WaterHolesUpperZrDim[3] = {0., 0.19*cm, UpperZrDim[2]+1*mm};
	G4double PinHolesDim[3] = {0., 0.15*cm, LowerZrDim[2]+1*mm};

    // Hole Position lower/upper plate
	G4double holePat[3]= {0.637286*cm, CLHEP::pi/3, 5*CLHEP::pi/6};

	// Zirconium Rod/Air Gap/Fuel Dimensions
	G4double ZirconiumRodDim[3] = {0., 0.262*cm, 23.3805*cm};
    G4double AirGapDim[3] = {0., 0.212*cm, 23.1105*cm};
    G4double FuelRodDim[3] = {0., 0.2064*cm, 22.6975*cm};
    SetFuelDimensions(G4ThreeVector(FuelRodDim[0],FuelRodDim[1],FuelRodDim[2]));

    // Top Zirconium rod
    G4double Center1[2] = {(3-12)*0.551833696*cm, (12-3)*0.955804*cm};
    // Buttom Zirconium rod
    G4double Center2[2] = {(10-12)*0.551833696*cm, (12-10)*0.955804*cm};

    /* End Of Reactor Core Dimensions */
    // Lattice Matrix
    G4double latticeMat[25][25] ={{7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7},
        {7, 7,  7,  7,  7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	8,	8,	8,	8,	7,	7,	7,	7,	7},
        {7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	8,	8,	9,	8,	9,	8,	9,	8,	8,	7,	7,	7},
        {7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	8,	9,	8,	9,	9,	8,	8,	9,	9,	8,	9,	8,	7,	7},
        {7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	8,	8,  8,	9,	8,	9,	8,	9,	8,	9,	8,	8,	8,	7,	7},
        {7,	7,	7,	7,	7,	7,	7,	7,	8,	9,	9,	9,	9,	8,	8,	9,	9,	8,	8,	9,	9,	9,	9,	8,	7},
        {7,	7,	7,	7,	7,	7,	7,	8,	8,	9,	8,	8,	9,	8,	9,	8,	9,	8,	9,	8,	8,	9,	8,	8,	7},
        {7,	7,	7,	7,	7,	7,	8,	9,	8,	9,	8,	8,	8,	9,	8,	8,	9,	8,	8,	8,	9,	8,	9,	8,	7},
        {7,	7,	7,	7,	7,	8,	8,	8,	8,	9,	9,	9,	9,	8,	9,	8,	9,	9,	9,	9,	8,	8,	8,	8,	7},
        {7,	7,	7,	7,	7,	9,	9,	9,	9,	8,	8,	8,	8,	9,	9,	8,	8,	8,	8,	9,	9,	9,	9,	7,	7},
        {7,	7,	7,	7,	8,	9,	8,	8,	9,	8,	9,	9,	9,	8,	9,	9,	9,	8,	9,	8,	8,	9,	8,	7,	7},
        {7,	7,	7,	8,	8,	9,	8,	8,	9,	8,	9,	8,	9,	9,	8,	9,	8,	9,	8,	8,	9,	8,	8,	7,	7},
        {7,	7,	7,	8,	8,	9,	9,	8,	9,	8,	8,	9,	7,	9,	8,	8,	9,	8,	9,	9,	8,	8,	7,	7,	7},
        {7,	7,	8,	8,	9,	8,	8,	9,	8,	9,	8,	9,	9,	8,	9,	8,	9,	8,	8,	9,	8,	8,	7,	7,	7},
        {7,	7,	8,	9,	8,	8,	9,	8,	9,	9,	9,	8,	9,	9,	9,	8,	9,	8,	8,	9,	8,	7,	7,	7,	7},
        {7,	9,	9,	9,	9,	9,	8,	8,	8,	8,	9,	9,	8,	8,	8,	8,	9,	9,	9,	9,	7,	7,	7,	7,	7},
        {7,	8,	8,	8,	8,	9,	9,	9,	9,	8,	9,	8,	9,	9,	9,	9,	8,	8,	8,	8,	7,	7,	7,	7,	7},
        {7,	8,	9,	8,	9,	8,	8,	8,	9,	8,	8,	9,	8,	8,	8,	9,	8,	9,	8,	7,	7,	7,	7,	7,	7},
        {7,	8,	8,	9,	8,	8,	9,	8,	9,	8,	9,	8,	9,	8,	8,	9,	8,	8,	7,	7,	7,	7,	7,	7,	7},
        {7,	8,	9,	9,	9,	9,	8,	8,	9,	9,	8,	8,	9,	9,	9,	9,	8,	7,	7,	7,	7,	7,	7,	7,	7},
        {7,	7,	8,	8,	8,	9,	8,	9,	8,	9,	8,	9,	8,	8,	8,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7},
        {7,	7,	8,	8,	8,	9,	9,	8,	8,	9,	9,	8,	9,	8,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7},
        {7,	7,	7,	8,	8,	9,	8,	9,	8,	9,	8,	8,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7},
        {7,	7,	7,	7,	7,	8,	8,	8,	8,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7},
        {7,	7,	7,	7,	7,	7,	7,	7,	7,	7,  7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7}};

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
		G4ThreeVector HolePos[6];
        for(G4int i=0; i<6; i++)
        {
            HolePos[i] = G4ThreeVector(holePat[0]*cos(holePat[2]+i*holePat[1]), holePat[0]*sin(holePat[2]+i*holePat[1]), 0.);
        }

		// Create world solid
		new G4Box("worldBox", encWorldDim[0]/2, encWorldDim[1]/2, encWorldDim[2]/2);

        // Create water pool
		new G4Tubs("cellTube", 0., reactorDim[1], reactorDim[2]/2, 0., 2.0*CLHEP::pi);

        // Create aluminium shell
		new G4Tubs("alumShellTube", alumShellTubeDim[0], alumShellTubeDim[1], alumShellTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("alumShellPlate", alumShellPlateDim[0], alumShellPlateDim[1], alumShellPlateDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4UnionSolid("alumShell", theSolids->GetSolid("alumShellPlate"), theSolids->GetSolid("alumShellTube"), 0, G4ThreeVector(0., 0., 271.*cm));


        // Create reflector solids
        new G4Tubs("reflectTop", refTopDim[0], refTopDim[1], refTopDim[2]/2, 0., CLHEP::pi);
		new G4Tubs("reflectAnnulus", refAnnDim[0], refAnnDim[1], refAnnDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("reflectBottom", refBottomDim[0], refBottomDim[1], refBottomDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4UnionSolid("reflector0", theSolids->GetSolid("reflectAnnulus"), theSolids->GetSolid("reflectTop"), 0, G4ThreeVector(0., 0., 13.969375*cm));
		new G4UnionSolid("reflector", theSolids->GetSolid("reflector0"), theSolids->GetSolid("reflectBottom"), 0, G4ThreeVector(0., 0., -16.962*cm));

        // D20 container
        new G4Tubs("D2OContainer1", D20ContainerDim[0], D20ContainerDim[1], D20ContainerDim[2]/2, 1.570796327*rad, 4.712388980*rad);
        new G4Tubs("D2OContainer2", 0., D20ContainerDim[1]+1.082*cm, D20ContainerDim[2]/2, 2.751192606*rad, 0.7808000945*rad);
        new G4IntersectionSolid("D2OContainer", theSolids->GetSolid("D2OContainer1"), theSolids->GetSolid("D2OContainer2"), 0, G4ThreeVector(1.082*cm,0.,0.));
        new G4Tubs("D2O", D20Dim[0], D20Dim[1], D20Dim[2]/2, 2.751192606*rad, 0.7808000945*rad);

        // Create aluminium tube solids
		new G4Tubs("smallAlumTube", smallAlumTubeDim[0], smallAlumTubeDim[1], smallAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("smallLongAlumTube", smallLongAlumTubeDim[0], smallLongAlumTubeDim[1], smallLongAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("largeAlumTube", largeAlumTubeDim[0], largeAlumTubeDim[1], largeAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);

		// Create beam tube solids
		new G4Tubs("smallBeamTube", smallBTubeDim[0], smallBTubeDim[1], smallBTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("smallLongBeamTube", smallLongBTubeDim[0], smallLongBTubeDim[1], smallLongBTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("cadLinTube", cadLinTubeDim[0], cadLinTubeDim[1], cadLinTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("largeBeamTube", largeBTubeDim[0], largeBTubeDim[1], largeBTubeDim[2]/2, 0., 2.0*CLHEP::pi);

        // Create control rod solids
		new G4Tubs("contRodCentTube", contRodCentTubeDim[0], contRodCentTubeDim[1], contRodCentTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("contRodCadTube", contRodCadTubeDim[0], contRodCadTubeDim[1], contRodCadTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("contRodAlumTube", contRodAlumTubeDim[0], contRodAlumTubeDim[1], contRodAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("contRodZirTube", contRodZirTubeDim[0], contRodZirTubeDim[1], contRodZirTubeDim[2]/2, 0., 2.0*CLHEP::pi);

        // Create zirconium lower and upper plate
        new G4Tubs("LowerZrTub", LowerZrDim[0], LowerZrDim[1], LowerZrDim[2]/2, LowerZrDim[3], LowerZrDim[4]);
        new G4Tubs("UpperZrTub", UpperZrDim[0], UpperZrDim[1], UpperZrDim[2]/2, UpperZrDim[3], UpperZrDim[4]);

        // Water holes lower and upper Zr plate
        new G4Tubs("WaterHolesLower", WaterHolesLowerZrDim[0], WaterHolesLowerZrDim[1], WaterHolesLowerZrDim[2]/2, 0, 2.0*CLHEP::pi);
        new G4Tubs("WaterHolesUpper", WaterHolesUpperZrDim[0], WaterHolesUpperZrDim[1], WaterHolesUpperZrDim[2]/2, 0, 2.0*CLHEP::pi);

        // Pin holes lower Zr plate
        new G4Tubs("PinHolesLower", PinHolesDim[0], PinHolesDim[1], PinHolesDim[2]/2, 0, 2.0*CLHEP::pi);

        // Zirconium Rods
        new G4Tubs("ZirconiumRod", ZirconiumRodDim[0], ZirconiumRodDim[1], ZirconiumRodDim[2]/2+1*mm, 0, 2.0*CLHEP::pi);
        new G4Tubs("AirGapRod", AirGapDim[0], AirGapDim[1], AirGapDim[2]/2, 0, 2.0*CLHEP::pi);

        std::stringstream name;

        for(G4int i = 0; i<34; i++){
            name.str("");
            name << "FuelRod" << i ;
            new G4Tubs(name.str(), FuelRodDim[0], FuelRadii[i], FuelRodDim[2]/2, 0, 2.0*CLHEP::pi);
        }


        // These list will be used to store all the holes solid and position in
        // the upper grid, and lower grid. Also, the zirconium rods to be added
        // and removed are kept track of. In a list containing a vector pointing
        // to the center of the solid and a pointer to the solid itself.
        solidList *theUpperHoles = new solidList();
        solidList *theLowerHoles = new solidList();
        solidList *theZirconiumMinus = new solidList();
        solidList *theZirconiumRods = new solidList();

        // neighbour is used to keep track of the holes that were previously add so
        // that no overlap takes place
        G4bool *neighbour;
        for(G4int i=1; i<24; i++)
        {
            for(G4int j=1; j<24; j++)
            {
                if(latticeMat[i][j] != 7)
                {
                    // Center sotres the x and y coordinate of the lattice cell that is being tracked.
                    G4double Center[2] = {((i-12)*0.551833696+(j-12)*1.103632018)*cm, (12-i)*0.955804*cm};
                    if(latticeMat[i-1][j] != 7 && latticeMat[i][j-1] != 7)
                    {
                        G4bool DUMMY[6] = {0,0,1,1,1,0};
                        neighbour = DUMMY;
                    }
                    else if(latticeMat[i-1][j] != 7)
                    {
                        G4bool DUMMY[6] = {0,1,1,1,1,0};
                        neighbour = DUMMY;
                    }
                    else if(latticeMat[i][j-1] != 7)
                    {
                        G4bool DUMMY[6] = {0,0,1,1,1,1};
                        neighbour = DUMMY;
                    }
                    else
                    {
                        G4bool DUMMY[6] = {1,1,1,1,1,1};
                        neighbour = DUMMY;
                    }

                    // Once the holes not added have been determined the program goes through and
                    // adds only the holes that are inside the pie slice that we want to create
                    for(G4int k = 1; k<6; k++)
                    {
                        if(neighbour[k])
                        {
                            G4double x = HolePos[k].getX()+Center[0], y = HolePos[k].getY()+Center[1];
                            G4double radius = sqrt(x*x+y*y);
                            if(radius+WaterHolesLowerZrDim[1] < refAnnDim[0] && radius-WaterHolesLowerZrDim[1] > contRodZirTubeDim[1]
                               && x-WaterHolesLowerZrDim[1] < radius*cos(LowerZrDim[3])
                               && x+WaterHolesLowerZrDim[1] > radius*cos(LowerZrDim[3]+LowerZrDim[4]) && y > 0)
                            {
                                theLowerHoles->push_back(std::make_pair(theSolids->GetSolid("WaterHolesLower"), G4ThreeVector(x, y, 0.)));
                            }
                            if(radius+WaterHolesUpperZrDim[1] < refAnnDim[0] && radius-WaterHolesUpperZrDim[1] > contRodZirTubeDim[1]
                               && x-WaterHolesUpperZrDim[1] < radius*cos(LowerZrDim[3])
                               && x+WaterHolesUpperZrDim[1] > radius*cos(LowerZrDim[3]+LowerZrDim[4]) && y > 0)
                            {
                                theUpperHoles->push_back(std::make_pair(theSolids->GetSolid("WaterHolesUpper"), G4ThreeVector(x, y, 0.)));
                            }
                        }
                    }

                    // If the material number is 9 then no rod is added, but a pin hole is added to the lower grid
                    if(latticeMat[i][j] == 9)
                    {
                        G4double x = Center[0], y = Center[1];
                        G4double radius = sqrt(x*x+y*y);
                        if(radius > contRodZirTubeDim[1] && x-PinHolesDim[1] < radius*cos(LowerZrDim[3])
                           && x+PinHolesDim[1] > radius*cos(LowerZrDim[3]+LowerZrDim[4]) && y > 0)
                        {
                            //G4cout << "x:" << x << "y:" << y;
                            theLowerHoles->push_back(std::make_pair(theSolids->GetSolid("PinHolesLower"), G4ThreeVector(x, y, 0.)));
                        }
                    }

                    // If the material number is 8, then a zirconium rod is added
                    else
                    {
                        G4double x = Center[0], y = Center[1];
                        G4double radius = sqrt(x*x+y*y);
                        if(x+ZirconiumRodDim[1] < radius*cos(LowerZrDim[3]) && x+ZirconiumRodDim[1] > radius*cos(LowerZrDim[3]+LowerZrDim[4]) && y > 0)
                        {
                            theZirconiumRods->push_back(std::make_pair(theSolids->GetSolid("ZirconiumRod"), G4ThreeVector(x, y, 0)));
                        }
                        else if(x-ZirconiumRodDim[1] < radius*cos(LowerZrDim[3]) && x-ZirconiumRodDim[1] > radius*cos(LowerZrDim[3]+LowerZrDim[4]) && y > 0)
                        {
                            theZirconiumMinus->push_back(std::make_pair(theSolids->GetSolid("ZirconiumRod"), G4ThreeVector(x, y, 0)));
                        }
                    }
                }
            }
        }



        // First the union of the holes in the lower plate is formed.
        StorkUnion* TheSolid = new StorkUnion(theLowerHoles);
        solidPos Temp1 = TheSolid->GetUnionSolid("LowerZrHoles");
        // The holes are then substracted from the lower grid.
        new G4SubtractionSolid("LowerZirconiumPlate1/6", theSolids->GetSolid("LowerZrTub"), Temp1.first, 0, Temp1.second);
        delete TheSolid;

        // The union of the holes on the top grid is taken
        TheSolid = new StorkUnion(theUpperHoles);
        Temp1 = TheSolid->GetUnionSolid("UpperZrHoles");
        // The holes are then substracted from upper grid
        new G4SubtractionSolid("UpperZirconiumPlate1/6", theSolids->GetSolid("UpperZrTub"), Temp1.first, 0, Temp1.second);
        delete TheSolid;


        // Finaly the upper grid is added to the lower grid both allready have their holes
        new G4UnionSolid("ZirconiumWithoutRods1/6", theSolids->GetSolid("LowerZirconiumPlate1/6"), theSolids->GetSolid("UpperZirconiumPlate1/6"), 0, G4ThreeVector(0.,0., 22.5*cm));

        // The Zr rod to be substracted union is now created
        TheSolid = new StorkUnion(theZirconiumMinus);
        Temp1 = TheSolid->GetUnionSolid("UpperZrHoles");
        // The rods are then substracted from the upper and the lower plates
        new G4SubtractionSolid("ZirconiumWithoutRodsMinus1/6", theSolids->GetSolid("ZirconiumWithoutRods1/6"), Temp1.first, 0, Temp1.second+G4ThreeVector(0., 0., (ZirconiumRodDim[2]+LowerZrDim[2]-1*mm)/2));
        delete TheSolid;

        // The Zr rod to be added are now unionized
        TheSolid = new StorkUnion(theZirconiumRods);
        Temp1 = TheSolid->GetUnionSolid("ZrTubs");
        // The Zr union is added to the lower plate and upper plate
        new G4UnionSolid("Zirconium1/6-", theSolids->GetSolid("ZirconiumWithoutRodsMinus1/6"), Temp1.first, 0, Temp1.second+G4ThreeVector(0., 0., (ZirconiumRodDim[2]+LowerZrDim[2]-1*mm)/2));
        delete TheSolid;



        /*
         There are three pie slices to be created all with minor differences (one added rod, one missing rod).
         The Zirconium1/6- solid is the base for all of the grids, but the needed extra rods are added in the
         following section.
         */

        // First pie slice of the reactor core
        new G4SubtractionSolid("Zirconium1/6", theSolids->GetSolid("Zirconium1/6-"), theSolids->GetSolid("ZirconiumRod"), 0, G4ThreeVector(-Center1[0], Center1[1], (ZirconiumRodDim[2]+LowerZrDim[2]-1*mm)/2));

        // Second pie slice of the reactor core
        new G4UnionSolid("Zirconium2/6+", theSolids->GetSolid("Zirconium1/6-"), theSolids->GetSolid("ZirconiumRod"), 0, G4ThreeVector(Center1[0], Center1[1], (ZirconiumRodDim[2]+LowerZrDim[2]-1*mm)/2));
        new G4UnionSolid("Zirconium2/6", theSolids->GetSolid("Zirconium2/6+"), theSolids->GetSolid("ZirconiumRod"), 0, G4ThreeVector(Center2[0], Center2[1], (ZirconiumRodDim[2]+LowerZrDim[2]-1*mm)/2));

        // Third pie slice of the reactor core
        new G4UnionSolid("Zirconium3/6+", theSolids->GetSolid("Zirconium1/6-"), theSolids->GetSolid("ZirconiumRod"), 0, G4ThreeVector(Center1[0], Center1[1], (ZirconiumRodDim[2]+LowerZrDim[2]-1*mm)/2));
        new G4SubtractionSolid("Zirconium3/6-", theSolids->GetSolid("Zirconium3/6+"), theSolids->GetSolid("ZirconiumRod"), 0, G4ThreeVector(-Center1[0], Center1[1], (ZirconiumRodDim[2]+LowerZrDim[2]-1*mm)/2));
        new G4SubtractionSolid("Zirconium3/6", theSolids->GetSolid("Zirconium3/6-"), theSolids->GetSolid("ZirconiumRod"), 0, G4ThreeVector(-Center2[0], Center2[1], (ZirconiumRodDim[2]+LowerZrDim[2]-1*mm)/2));
		geomChanged = false;
	}
    // Position vector
    G4ThreeVector holePos;

    // Create world volume
    worldLogical = new G4LogicalVolume(theSolids->GetSolid("worldBox"), matMap["Galactic"],"worldLogical");
    worldPhysical = new G4PVPlacement(0, G4ThreeVector(0,0,0), worldLogical, "worldPhysical", 0, 0, 0);

    // Create the lattice cell (moderator) volume
    cellLogical = new G4LogicalVolume(theSolids->GetSolid("cellTube"),matMap["H2O"],"cellLogical");
    new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), cellLogical,"cellPhysical",worldLogical,0,0);

    // Create aluminium shell
    alumShellLogical = new G4LogicalVolume(theSolids->GetSolid("alumShell"),matMap["AlAlloy1"],"alumShellLogical");
    new G4PVPlacement(0, G4ThreeVector(-30*cm,-30*cm,22.5*cm-reactorDim[2]/2), alumShellLogical,"alumShellPhysical",cellLogical,0,0);

    // Reflector Logical Volume is being created
    ReflectorLogical = new G4LogicalVolume(theSolids->GetSolid("reflector"), matMap["Reflector"],"ReflectorLogical");
    new G4PVPlacement(0, G4ThreeVector(-30*cm,-30*cm, 53.042*cm-reactorDim[2]/2),  ReflectorLogical, "ReflectorPhysical", cellLogical, 0, 0);

    // Outter Tubes
    outSmallAlumLogical = new G4LogicalVolume(theSolids->GetSolid("smallLongAlumTube"),matMap["AlAlloy1"],"outAlumTubeLogical");
    holePos.set(outAlumTubePos[0]*cos(outAlumTubePos[1]*4+outAlumTubePos[2])-30*cm,-30*cm+outAlumTubePos[0]*sin(outAlumTubePos[1]*4+outAlumTubePos[2]),302.834*cm-reactorDim[2]/2);
    new G4PVPlacement(0, holePos, outSmallAlumLogical,"outSmallAlumTubePhysical1", cellLogical,0,0);
    holePos.set(-30*cm+outAlumTubePos[0]*cos(outAlumTubePos[1]*1+outAlumTubePos[2]),-30*cm+outAlumTubePos[0]*sin(outAlumTubePos[1]*1+outAlumTubePos[2]),302.834*cm-reactorDim[2]/2);
    new G4PVPlacement(0, holePos, outSmallAlumLogical,"outSmallAlumTubePhysical2", cellLogical,0,0);
    outLargeAlumLogical = new G4LogicalVolume(theSolids->GetSolid("largeAlumTube"),matMap["AlAlloy1"],"outAlumTubeLogical");
    holePos.set(-30*cm+outAlumTubePos[0]*cos(outAlumTubePos[2]),-30*cm+outAlumTubePos[0]*sin(outAlumTubePos[2]), 302.834*cm-reactorDim[2]/2);
    new G4PVPlacement(0, holePos, outLargeAlumLogical,"outLargeAlumTubePhysical", cellLogical,0,0);
    cadLinLogical = new G4LogicalVolume(theSolids->GetSolid("cadLinTube"),matMap["Cadmium"],"cadLinLogical");
    holePos.set(-30*cm+outAlumTubePos[0]*cos(outAlumTubePos[1]*3+outAlumTubePos[2]),-30*cm+outAlumTubePos[0]*sin(outAlumTubePos[1]*3+outAlumTubePos[2]),302.834*cm-reactorDim[2]/2);
    new G4PVPlacement(0, holePos, cadLinLogical,"cadLinTube", cellLogical,0,0);
    new G4PVPlacement(0, holePos, outSmallAlumLogical,"outSmallAlumTubePhysical3", cellLogical,0,0);
    outSmallBeamLogical = new G4LogicalVolume(theSolids->GetSolid("smallLongBeamTube"),matMap["Air"],"outSmallBeamLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), outSmallBeamLogical,"outSmallBeamTubePhysical",outSmallAlumLogical,0,0);
    outLargeBeamLogical = new G4LogicalVolume(theSolids->GetSolid("largeBeamTube"),matMap["Air"],"outLargeBeamLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), outLargeBeamLogical,"outLargeBeamTubePhysical",outLargeAlumLogical,0,0);

    // Inner Irradiation Tubes
    insAlumLogical = new G4LogicalVolume(theSolids->GetSolid("smallAlumTube"),matMap["AlAlloy3"],"insAlumLogical");
    G4int copyNum=0;
    for (G4int i=0; i<5; i++)
    {
        holePos.set((alumTubePos[0]*cos(alumTubePos[1]*i+alumTubePos[2])), (alumTubePos[0]*sin(alumTubePos[1]*i+alumTubePos[2])), 253.292*cm);
        new G4PVPlacement(0, holePos, insAlumLogical,"insAlumTubePhysical", ReflectorLogical, copyNum, 0);
        copyNum++;
    }

    // The air is placed inside the irradiation tubes
    insBeamLogical = new G4LogicalVolume(theSolids->GetSolid("smallBeamTube"),matMap["Air"],"insBeamLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), insBeamLogical,"insBeamTubePhysical",insAlumLogical,0,0);

    // D20 Container
    D2OContainerLogical = new G4LogicalVolume(theSolids->GetSolid("D2OContainer"), matMap["AlAlloy2"],"D2OContainerLogical");
    new G4PVPlacement(0, G4ThreeVector(0-30*cm, 0-30*cm, 53.042*cm-reactorDim[2]/2),  D2OContainerLogical, "D2OPhysical", cellLogical, 0, 0);
    D2OLogical = new G4LogicalVolume(theSolids->GetSolid("D2O"), matMap["D2O"],"D2OLogical");
    new G4PVPlacement(0, G4ThreeVector(0, 0, 0.25375*cm),  D2OLogical, "D2OPhysical", D2OContainerLogical, 0, 0);

    // Creates the zirconium guide
    contRodZirLogical = new G4LogicalVolume(theSolids->GetSolid("contRodZirTube"),matMap["Zirconium"], "contRodZirLogical");
    new G4PVPlacement(0, G4ThreeVector(0.-30*cm, 0.-30*cm, 53.02925*cm-reactorDim[2]/2), contRodZirLogical,"contRodZirPhysical",cellLogical,0,0);

    // Create the control rod
    contRodAlumLogical = new G4LogicalVolume(theSolids->GetSolid("contRodAlumTube"),matMap["AlAlloy4"], "contRodAlumLogical");
    new G4PVPlacement(0, G4ThreeVector(0.-30*cm, 0.-30*cm, 82.14*cm-reactorDim[2]/2-ControlRodPosition), contRodAlumLogical,"contRodAlumPhysical",cellLogical,0,0);
    contRodCadLogical = new G4LogicalVolume(theSolids->GetSolid("contRodCadTube"),matMap["Cadmium"], "contRodCadLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., -5.44*cm), contRodCadLogical,"contRodCadPhysical",contRodAlumLogical,0,0);
    contRodCentLogical = new G4LogicalVolume(theSolids->GetSolid("contRodCentTube"),matMap["Air"], "contRodCentLogical");

    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), contRodCentLogical,"contRodCentPhysical",contRodCadLogical,0,0);

    // Creates the three zirconium pie slice which together form half of the reactor core.
    ZirconiumLogical1 = new G4LogicalVolume(theSolids->GetSolid("Zirconium1/6"), matMap["Zirconium"], "ZirconiumLogical1");
    ZirconiumLogical2 = new G4LogicalVolume(theSolids->GetSolid("Zirconium2/6"), matMap["Zirconium"], "ZirconiumLogical2");
    ZirconiumLogical3 = new G4LogicalVolume(theSolids->GetSolid("Zirconium3/6"), matMap["Zirconium"], "ZirconiumLogical3");

    // Adding the fuel rods to all of the common rods of the three pie slices
    G4int AirGapNum = 0;
    std::stringstream Material;
    std::stringstream FuelName;
    for(G4int y=1; y<24; y++)
    {
        for(G4int x=1; x<24; x++)
        {
            G4double Center[2] = {((y-12)*0.551833696+(x-12)*1.103632018)*cm, (12-y)*0.955804*cm};

            G4double radius = sqrt(Center[0]*Center[0]+Center[1]*Center[1]);
            if(latticeMat[y][x] == 8 && Center[0]+ZirconiumRodDim[1] < radius*cos(LowerZrDim[3])
                                && Center[0]+ZirconiumRodDim[1] > radius*cos(LowerZrDim[3]+LowerZrDim[4]) && Center[1] > 0)
            {
                // Associating right material with volume
                Material.str("");
                Material << "Fuel" << AirGapNum;
                FuelName.str("");
                FuelName << "FuelRod" << AirGapNum;

                // Creating Air Gap in fuel Assemblie and fuel rod elements
                AirGapLogical = new G4LogicalVolume(theSolids->GetSolid("AirGapRod"), matMap["Air"], "AirGapLogical");
                FuelRodLogical = new G4LogicalVolume(theSolids->GetSolid(FuelName.str()), matMap[Material.str()], "FuelRodLogical");
                new G4PVPlacement(0, G4ThreeVector(0,0,-0.0875*cm),  FuelRodLogical, "FuelRodPhysical", AirGapLogical, 0, 0);

                new G4PVPlacement(0, G4ThreeVector(Center[0], Center[1], 11.82975*cm),  AirGapLogical, "AirGapPhysical", ZirconiumLogical1, 0, 0);
                new G4PVPlacement(0, G4ThreeVector(Center[0], Center[1], 11.82975*cm),  AirGapLogical, "AirGapPhysical", ZirconiumLogical2, 0, 0);
                new G4PVPlacement(0, G4ThreeVector(Center[0], Center[1], 11.82975*cm),  AirGapLogical, "AirGapPhysical", ZirconiumLogical3, 0, 0);
                AirGapNum++;

                AirGapLogical->SetSensitiveDetector( sDReactor );
                FuelRodLogical->SetSensitiveDetector( sDReactor );

                // Air Gap Visualization
                AirGapAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
                AirGapAtt->SetVisibility(false);
                AirGapLogical->SetVisAttributes(AirGapAtt);

                // Fuel Visualization
                FuelRodAtt = new G4VisAttributes(G4Colour(0.,1.,0.));
                FuelRodAtt->SetVisibility(false);
                FuelRodLogical->SetVisAttributes(FuelRodAtt);
            }
        }
    }

    // Creating the right material for the fuel rod
    Material.str("");
    Material << "Fuel" << AirGapNum;

    // Creating Air Gap in fuel Assemblie and fuel rod elements
    AirGapLogical = new G4LogicalVolume(theSolids->GetSolid("AirGapRod"), matMap["Air"], "AirGapLogical32");
    FuelRodLogical = new G4LogicalVolume(theSolids->GetSolid("FuelRod32"), matMap[Material.str()], "FuelRodLogical32");
    new G4PVPlacement(0, G4ThreeVector(0, 0, -0.0875*cm),  FuelRodLogical, "FuelRodPhysical32", AirGapLogical, 0, 0);



    // Adding one rod in the third pie slice
    new G4PVPlacement(0, G4ThreeVector(Center1[0], Center1[1], 11.82975*cm),  AirGapLogical, "AirGapPhysical32", ZirconiumLogical3, 0, 0);
    // Ading the fuel to only the pie slices that need extra.
    // Note no extra rods need to be added to the first grid.
    // Adding two rods in the second pie slice
    new G4PVPlacement(0, G4ThreeVector(Center1[0], Center1[1], 11.82975*cm),  AirGapLogical, "AirGapPhysical32", ZirconiumLogical2, 0, 0);

    AirGapLogical->SetSensitiveDetector( sDReactor );
    FuelRodLogical->SetSensitiveDetector( sDReactor );

    // Air Gap Visualization
    AirGapAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    AirGapAtt->SetVisibility(false);
    AirGapLogical->SetVisAttributes(AirGapAtt);

    // Fuel Visualization
    FuelRodAtt = new G4VisAttributes(G4Colour(0.,1.,0.));
    FuelRodAtt->SetVisibility(false);
    FuelRodLogical->SetVisAttributes(FuelRodAtt);

    // Creating the right material for the fuel rod
    AirGapNum++;
    Material.str("");
    Material << "Fuel" << AirGapNum;

    // Creating Air Gap in fuel Assemblie and fuel rod elements
    AirGapLogical = new G4LogicalVolume(theSolids->GetSolid("AirGapRod"), matMap["Air"], "AirGapLogical33");
    FuelRodLogical = new G4LogicalVolume(theSolids->GetSolid("FuelRod33"), matMap[Material.str()], "FuelRodLogical33");
    new G4PVPlacement(0, G4ThreeVector(0, 0, -0.0875*cm),  FuelRodLogical, "FuelRodPhysical33", AirGapLogical, 0, 0);

    new G4PVPlacement(0, G4ThreeVector(Center2[0], Center2[1], 11.82975*cm),  AirGapLogical, "AirGapPhysical33", ZirconiumLogical2, 0, AirGapNum+1);


    // Placing the pie slices where they belong
    std::stringstream PhysicalName;
    G4RotationMatrix* zRot;
    G4int copynum = 0;
    for(G4int i=0; i<6; i++)
    {
        PhysicalName.str("");
        zRot = new G4RotationMatrix;
        zRot->rotateZ(-i*CLHEP::pi/3);
        if(i == 0 || i == 3)
        {
            PhysicalName << "ZirconiumPhysical1-"<< copynum;
            new G4PVPlacement(zRot, G4ThreeVector(-30*cm, -30*cm, 41.5535*cm-reactorDim[2]/2),  ZirconiumLogical1, PhysicalName.str(), cellLogical, 0, copynum);
        }
        else if(i == 1 || i==4)
        {
            PhysicalName << "ZirconiumPhysical2-" << copynum;
            new G4PVPlacement(zRot, G4ThreeVector(-30*cm, -30*cm, 41.5535*cm-reactorDim[2]/2),  ZirconiumLogical2, PhysicalName.str(), cellLogical, 0, copynum);
        }
        else if(i == 2 || i == 5)
        {
            PhysicalName << "ZirconiumPhysical3-" << copynum;
            new G4PVPlacement(zRot, G4ThreeVector(-30*cm, -30*cm, 41.5535*cm-reactorDim[2]/2),  ZirconiumLogical3, PhysicalName.str(), cellLogical, 0, copynum);
            copynum++;
        }
    }

	// Add sensitive detector to ALL logical volumes
	worldLogical->SetSensitiveDetector( sDReactor );
    ZirconiumLogical1->SetSensitiveDetector( sDReactor );
    ZirconiumLogical2->SetSensitiveDetector( sDReactor );
    ZirconiumLogical3->SetSensitiveDetector( sDReactor );
    AirGapLogical->SetSensitiveDetector( sDReactor );
    FuelRodLogical->SetSensitiveDetector( sDReactor );
    ReflectorLogical->SetSensitiveDetector( sDReactor );
    D2OContainerLogical->SetSensitiveDetector( sDReactor );
    D2OLogical->SetSensitiveDetector( sDReactor );
    contRodZirLogical->SetSensitiveDetector( sDReactor );
	contRodAlumLogical->SetSensitiveDetector( sDReactor );
	contRodCadLogical->SetSensitiveDetector( sDReactor );
	contRodCentLogical->SetSensitiveDetector( sDReactor );
    insAlumLogical->SetSensitiveDetector( sDReactor );
	insBeamLogical->SetSensitiveDetector( sDReactor );
	outSmallAlumLogical->SetSensitiveDetector( sDReactor );
	outLargeAlumLogical->SetSensitiveDetector( sDReactor );
	cadLinLogical->SetSensitiveDetector( sDReactor );
	outSmallBeamLogical->SetSensitiveDetector( sDReactor );
	outLargeBeamLogical->SetSensitiveDetector( sDReactor );
    alumShellLogical->SetSensitiveDetector( sDReactor );
    cellLogical->SetSensitiveDetector( sDReactor );

    /* This is where all the visualizaion attributes are made */
    // World Visualization
    worldVisAtt = new G4VisAttributes(G4Colour(0.5,0.5,0.5));
    worldVisAtt->SetVisibility(false);
    worldLogical->SetVisAttributes(worldVisAtt);

    // Water Tub Visualization
    cellVisAtt = new G4VisAttributes(G4Colour(0., 0., 1.));
    cellVisAtt->SetVisibility(false);
    cellLogical->SetVisAttributes(cellVisAtt);

    // Aluminum Reactor Shell Visualization
    alumShellVisAtt = new G4VisAttributes(G4Colour(173./255,178./255,189./255));
    alumShellVisAtt->SetVisibility(false);
    alumShellLogical->SetVisAttributes(alumShellVisAtt);

    // Zirconium Visualization
    ZirconiumAtt1 = new G4VisAttributes(G4Colour(0.,0.,0.));
    ZirconiumAtt2 = new G4VisAttributes(G4Colour(0.,0.,0.));
    ZirconiumAtt3 = new G4VisAttributes(G4Colour(0.,0.,0.));
    ZirconiumAtt1->SetVisibility(false);
    ZirconiumAtt2->SetVisibility(false);
    ZirconiumAtt3->SetVisibility(false);
    ZirconiumLogical1->SetVisAttributes(ZirconiumAtt1);
    ZirconiumLogical2->SetVisAttributes(ZirconiumAtt2);
    ZirconiumLogical3->SetVisAttributes(ZirconiumAtt3);

    // Air Gap Visualization
    AirGapAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    AirGapAtt->SetVisibility(false);
    AirGapLogical->SetVisAttributes(AirGapAtt);

    // Fuel Visualization
    FuelRodAtt = new G4VisAttributes(G4Colour(0.,1.,0.));
    FuelRodAtt->SetVisibility(false);
    FuelRodLogical->SetVisAttributes(FuelRodAtt);

    // Reflector Visualization
    ReflectorAtt = new G4VisAttributes(G4Colour(205./255,127./255,50./255));
    ReflectorAtt->SetVisibility(false);
    ReflectorLogical->SetVisAttributes(ReflectorAtt);


    // D2O Column and Water Visualization
    D2OContainerAtt = new G4VisAttributes(G4Colour(173./255,178./255,189./255));
    D2OContainerAtt->SetVisibility(false);
    D2OContainerLogical->SetVisAttributes(D2OContainerAtt);
    D2OAtt = new G4VisAttributes(G4Colour(135./255,206./255,255./255));
    D2OAtt->SetVisibility(false);
    D2OLogical->SetVisAttributes(D2OAtt);

    // Control Rod Visualization
    contRodZirVisAtt = new G4VisAttributes(G4Colour(0.,0.,0.));
    contRodZirVisAtt->SetVisibility(false);
    contRodZirLogical->SetVisAttributes(contRodZirVisAtt);

    contRodAlumVisAtt = new G4VisAttributes(G4Colour(173./255,178./255,189./255));
    contRodAlumVisAtt->SetVisibility(false);
    contRodAlumLogical->SetVisAttributes(contRodAlumVisAtt);

    contRodCadVisAtt = new G4VisAttributes(G4Colour(237./255,135./255.0,45./255.0));
    contRodCadVisAtt->SetVisibility(false);
    contRodCadLogical->SetVisAttributes(contRodCadVisAtt);

    contRodCentVisAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    contRodCentVisAtt->SetVisibility(false);
    contRodCentLogical->SetVisAttributes(contRodCentVisAtt);

    // Irradiations Sites Visualization
    insAlumVisAtt = new G4VisAttributes(G4Colour(173./255,178./255,189./255));
    insAlumVisAtt->SetVisibility(false);
    insAlumLogical->SetVisAttributes(insAlumVisAtt);

    insBeamVisAtt = new G4VisAttributes(G4Colour(1., 0., 0.));
    insBeamVisAtt->SetVisibility(false);
    insBeamLogical->SetVisAttributes(insBeamVisAtt);

    outSmallAlumVisAtt = new G4VisAttributes(G4Colour(173./255,178./255,189./255));
    outSmallAlumVisAtt->SetVisibility(false);
    outSmallAlumLogical->SetVisAttributes(outSmallAlumVisAtt);

    outLargeAlumVisAtt = new G4VisAttributes(G4Colour(173./255,178./255,189./255));
    outLargeAlumVisAtt->SetVisibility(false);
    outLargeAlumLogical->SetVisAttributes(outLargeAlumVisAtt);

    cadLinTubeVisAtt = new G4VisAttributes(G4Colour(237./255,135./255.0,45./255.0));
    cadLinTubeVisAtt->SetVisibility(false);
    cadLinLogical->SetVisAttributes(cadLinTubeVisAtt);

    outSmallBeamVisAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    outSmallBeamVisAtt->SetVisibility(false);
    outSmallBeamLogical->SetVisAttributes(outSmallBeamVisAtt);

    outLargeBeamVisAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    outLargeBeamVisAtt->SetVisibility(false);
    outLargeBeamLogical->SetVisAttributes(outLargeBeamVisAtt);

    return worldPhysical;
}


// ConstructMaterials()
// Define and build the materials in the C6 lattice cell.
void SLOWPOKEConstructor::ConstructMaterials()
{
    // Density Of Defined Materials
    G4double ReflectorDensity = 1.85*g/cm3;
    G4double LWDensity = 0.998*g/cm3;
   // G4double FuelDensity = 10.6*g/cm3;
    G4double AirDensity = 5.0807e-5*g/cm3;
    G4double ZrDensity = 6.49*g/cm3;
    G4double AlAlloyDensity = 2.70*g/cm3;
    G4double CadmiumDensity = 8.65*g/cm3;
    G4double HWDensity = 1.105*g/cm3;

    // Temperature Of Defined Materials
    // using data from 20043405
    G4double ReflectorTemp=(22.5+273.15)*kelvin;
    G4double LWTemp=(30.6+273.15)*kelvin;
    G4double AirTemp=(18.0+273.15)*kelvin;
    G4double ZrTemp=(52.14+273.15)*kelvin;
    G4double AlAlloyTemp1=(20.0+273.15)*kelvin;
    G4double AlAlloyTemp2=(21.0+273.15)*kelvin;
    G4double AlAlloyTemp3=(22.0+273.15)*kelvin;
    G4double AlAlloyTemp4=(48.0+273.15)*kelvin;
    G4double CadmiumTemp=(50.0+273.15)*kelvin;
    G4double HWTemp=(20.5+273.15)*kelvin;

    // Defining all the pointers
    G4Isotope *C12, *C13, *N14, *N15, *O16, *O17, *O18, *Mg24,
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
    O18 = new G4Isotope("O18", 8, 18, 17.999*g/mole);

    // Natural occuring oxygen
    Oxygen = new G4Element("Oxygen", "O", 3);
    Oxygen->AddIsotope(O16, 99.757*perCent);
    Oxygen->AddIsotope(O17, 0.038*perCent);
    Oxygen->AddIsotope(O18, 0.205*perCent);

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
    Cu65 = new G4Isotope("Cu65", 29, 66, 64.9277929*g/mole);

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
    World = new StorkMaterialHT("Galactic", 1, 1, 1.e-25*g/cm3, 0*joule/g/kelvin, 0.0*joule/(s*m*kelvin), kStateGas,
                              2.73*kelvin, 3.e-18*pascal);

    // (Material #1) Beryllium Sheild with Impurities
    Reflector = new StorkMaterialHT("Reflector", ReflectorDensity, 17, 1.83*joule/g/kelvin, 218*joule/(s*m*kelvin), kStateSolid, ReflectorTemp);
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
    LW = new StorkMaterialHT("H2O", LWDensity, 2, 4.1813*joule/g/kelvin, 0.5984*joule/(s*m*kelvin), kStateLiquid, LWTemp);
    LW->AddElement(Oxygen, 1);
    LW->AddElement(H1,     2);

    // (Material #4) Air
    Air = new StorkMaterialHT("Air", AirDensity, 2, 1.0035*joule/g/kelvin, 0.024*joule/(s*m*kelvin), kStateGas, AirTemp);
    Air->AddElement(Oxygen, 0.21174);
    Air->AddElement(N,      0.78826);

    // (Material #5) Zr
    Zr = new StorkMaterialHT("Zirconium", ZrDensity, 1, 0.278*joule/g/kelvin, 8.625*joule/(s*m*kelvin), kStateSolid, ZrTemp);
    Zr->AddElement(Zirc, 1);

    // (Material #6) Aluminum with impurities
    AlAlloy1 = new StorkMaterialHT("AlAlloy1", AlAlloyDensity, 5, 0.897*joule/g/kelvin, 205*joule/(s*m*kelvin), kStateSolid, AlAlloyTemp1);
    AlAlloy1->AddElement(Al, 0.9792);
    AlAlloy1->AddElement(Si, 0.0060);
    AlAlloy1->AddElement(Cu, 0.0028);
    AlAlloy1->AddElement(Mg, 0.0100);
    AlAlloy1->AddElement(Cr, 0.0020);

    AlAlloy2 = new StorkMaterialHT("AlAlloy2", AlAlloyDensity, 5, 0.897*joule/g/kelvin, 205*joule/(s*m*kelvin), kStateSolid, AlAlloyTemp2);
    AlAlloy2->AddElement(Al, 0.9792);
    AlAlloy2->AddElement(Si, 0.0060);
    AlAlloy2->AddElement(Cu, 0.0028);
    AlAlloy2->AddElement(Mg, 0.0100);
    AlAlloy2->AddElement(Cr, 0.0020);

    AlAlloy3 = new StorkMaterialHT("AlAlloy3", AlAlloyDensity, 5, 0.897*joule/g/kelvin, 205*joule/(s*m*kelvin), kStateSolid, AlAlloyTemp3);
    AlAlloy3->AddElement(Al, 0.9792);
    AlAlloy3->AddElement(Si, 0.0060);
    AlAlloy3->AddElement(Cu, 0.0028);
    AlAlloy3->AddElement(Mg, 0.0100);
    AlAlloy3->AddElement(Cr, 0.0020);

    AlAlloy4 = new StorkMaterialHT("AlAlloy4", AlAlloyDensity, 5, 0.897*joule/g/kelvin, 205*joule/(s*m*kelvin), kStateSolid, AlAlloyTemp4);
    AlAlloy4->AddElement(Al, 0.9792);
    AlAlloy4->AddElement(Si, 0.0060);
    AlAlloy4->AddElement(Cu, 0.0028);
    AlAlloy4->AddElement(Mg, 0.0100);
    AlAlloy4->AddElement(Cr, 0.0020);

    // (Material #7) Cadmium
    Cadmium = new StorkMaterialHT("Cadmium", CadmiumDensity, 1, 0.231*joule/g/kelvin, 92*joule/(s*m*kelvin), kStateSolid, CadmiumTemp);
    Cadmium->AddElement(Cd, 1);

    // (Materail #8) Heavy Water
    HW = new StorkMaterialHT("D2O", HWDensity, 2,  4.224211211*joule/g/kelvin, 0.589*joule/(s*m*kelvin), kStateLiquid, HWTemp);
    HW->AddElement(D2,     2);
    HW->AddElement(Oxygen, 1);


    // Add materials to the map indexed by either ZA (format ZZAAA or ZZ)
    // For composite materials:  world is 0, heavy water is 1, UHW is 2
    matMap["Galactic"] = World;
    matMap["H2O"] = LW;
    matMap["D2O"] = HW;
    matMap["Zirconium"] = Zr;
    matMap["AlAlloy1"] = AlAlloy1;
    matMap["AlAlloy2"] = AlAlloy2;
    matMap["AlAlloy3"] = AlAlloy3;
    matMap["AlAlloy4"] = AlAlloy4;
    matMap["Reflector"] = Reflector;
    matMap["Cadmium"] = Cadmium;
    matMap["Air"] = Air;


    // (Material #3) Fuel Rods (19.95% Enriched Uranium in (UO2))
    std::stringstream matName;
    for(G4int i = 0; i <34; i++)
    {
        matName.str("");
        matName << "Fuel" << i;
        Fuel = new StorkMaterialHT(matName.str(), FuelDensities[i], 2, 0.2411519506*joule/g/kelvin, 21.5*joule/(s*m*kelvin), kStateSolid, FuelTemperatures[i]);
        Fuel->AddElement(Oxygen, 2);
        Fuel->AddElement(LEU,    1);

        matMap[matName.str()] = Fuel;
    }


    matChanged = false;

    return;
}
