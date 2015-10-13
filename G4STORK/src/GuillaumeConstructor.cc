#include "GuillaumeConstructor.hh"

GuillaumeConstructor::GuillaumeConstructor()
: StorkVWorldConstructor(), ZirconiumLogical(0), WaterHolesLowerLogical(0), WaterHolesUpperLogical(0),
AirGapLogical(0), FuelRodLogical(0), LowerPinLogical(0), ReflectorLogical(0), D2OContainerLogical(0),
D2OLogical(0), contRodZirLogical(0), contRodAlumLogical(0), contRodCadLogical(0), contRodCentLogical(0),
insAlumLogical(0), insBeamLogical(0), outSmallAlumLogical(0), outLargeAlumLogical(0), cadLinLogical(0),
outSmallBeamLogical(0), outLargeBeamLogical(0), alumShellLogical(0), cellLogical(0)
{
	// Set default member variables (from file or default values)
	latticePitch = 28.575*cm;
	fuelTemp = 859.99*kelvin;
	fuelDensity = 10.5541*g/cm3;
	moderatorTemp = 336.16*kelvin;
	moderatorDensity = 1.08875*g/cm3;

	// Set up variable property map
	variablePropMap[MatPropPair(fuel,temperature)] = &fuelTemp;
    variablePropMap[MatPropPair(fuel,density)] = &fuelDensity;
    variablePropMap[MatPropPair(moderator,temperature)] = &moderatorTemp;
    variablePropMap[MatPropPair(moderator,density)] = &moderatorDensity;
    variablePropMap[MatPropPair(all,dimension)] = &latticePitch;

    cellVisAtt=NULL;
	alumShellVisAtt=NULL;
	D2OContainerAtt=NULL;
	insAlumVisAtt=NULL;
	insBeamVisAtt=NULL;
	outSmallAlumVisAtt=NULL;
	outLargeAlumVisAtt=NULL;
	cadLinTubeVisAtt=NULL;
	outSmallBeamVisAtt=NULL;
	outLargeBeamVisAtt=NULL;
	ReflectorAtt=NULL;
	D2OAtt=NULL;
	LowerPinAtt=NULL;
	WaterHolesUpperAtt=NULL;
	WaterHolesLowerAtt=NULL;
	ZirconiumAtt=NULL;
	AirGapAtt=NULL;
	FuelRodAtt=NULL;
	contRodZirVisAtt=NULL;
	contRodAlumVisAtt=NULL;
	contRodCadVisAtt=NULL;
	contRodCentVisAtt=NULL;
}

GuillaumeConstructor::~GuillaumeConstructor()
{
	// Delete visualization attributes
	if(ZirconiumAtt)
        delete ZirconiumAtt;
	if(WaterHolesLowerAtt)
        delete WaterHolesLowerAtt;
	if(WaterHolesUpperAtt)
        delete WaterHolesUpperAtt;
	if(AirGapAtt)
        delete AirGapAtt;
	if(FuelRodAtt)
        delete FuelRodAtt;
	if(LowerPinAtt)
        delete LowerPinAtt;
	if(ReflectorAtt)
        delete ReflectorAtt;
	if(D2OContainerAtt)
        delete D2OContainerAtt;
	if(D2OAtt)
        delete D2OAtt;
	if(contRodZirVisAtt)
        delete contRodZirVisAtt;
    if(contRodAlumVisAtt)
        delete contRodAlumVisAtt;
    if(contRodCadVisAtt)
        delete contRodCadVisAtt;
    if(contRodCentVisAtt)
        delete contRodCentVisAtt;
    if(insAlumVisAtt)
        delete insAlumVisAtt;
    if(insBeamVisAtt)
        delete insBeamVisAtt;
    if(outSmallAlumVisAtt)
        delete outSmallAlumVisAtt;
    if(outLargeAlumVisAtt)
        delete outLargeAlumVisAtt;
    if(cadLinTubeVisAtt)
        delete cadLinTubeVisAtt;
    if(outSmallBeamVisAtt)
        delete outSmallBeamVisAtt;
    if(outLargeBeamVisAtt)
        delete outLargeBeamVisAtt;
    if(alumShellVisAtt)
        delete alumShellVisAtt;
    if(cellVisAtt)
        delete cellVisAtt;
}

// ConstructWorld()
// Construct the geometry and materials of the Guillaume Reactor.
G4VPhysicalVolume* GuillaumeConstructor::ConstructWorld()
{


	G4SolidStore* theSolids = G4SolidStore::GetInstance();

	// Set static dimensions
	//Note the format for cylinder dimensions is (inner radius, outer radius, height)

	// Reactor pool dimensions
    G4double buffer = 1.0*cm;
	reactorDim = G4ThreeVector(0., 133.0*cm, 564.0*cm);

	// World dimensions
	encWorldDim = 2.0*G4ThreeVector(reactorDim[1]+buffer,reactorDim[1]+buffer, reactorDim[2]/2+buffer);

    // Reflector dimensions
    G4double refAnnDim[3] = {11.049*cm, 21.2344*cm, 22.748*cm};
    G4double refBottomDim[3] = {0.0*cm, 16.113125*cm, 10.16*cm};
    G4double refTopDim[3] = {1.3890625*cm, 12.065*cm, 0.15875*cm};

    // D2O Colum Container and Water
    G4double D20ContainerDim[3] = {21.2344*cm, 30*cm, 22.748*cm};
    G4double D20Dim[3] = {22.2344*cm, 29*cm, 20.6975*cm};

    // Beamtube dimensions
    G4double smallBTubeDim[3] = {0.0*cm, 1.40208*cm, 515.332*cm};
    G4double smallLongBTubeDim[3] = {0.0*cm, 1.40208*cm, 522.332*cm};
    G4double largeBTubeDim[3] = {0.0*cm, 1.6*cm, 515.332*cm};

    // Aluminum Tube dimensions
    G4double smallAlumTubeDim[3] = {0.0*cm, 1.56718*cm, 515.332*cm};
    G4double smallLongAlumTubeDim[3] = {0.0*cm, 1.56718*cm, 522.332*cm};
    G4double largeAlumTubeDim[3] = {0.0*cm, 1.905*cm, 515.332*cm};
    G4double alumTubePos[3]={14.56182*cm, 0.4*CLHEP::pi, 0.};
    G4double outAlumTubePos[3]={24.0*cm, 0.4*CLHEP::pi, 0.2*CLHEP::pi};

    // Aluminium Reactor Shell
    G4double alumShellTubeDim[3] = {30.0*cm, 31.0*cm, 541.0*cm};
    G4double alumShellPlateDim[3] = {0.0*cm, 31.0*cm, 1*cm};

    // Cadmium lining
    G4double cadLinTubeDim[3] = {1.56718*cm, 1.61798*cm, 22.748*cm};

    // Control Rod
    G4double contRodCentTubeDim[3] = {0.0*cm, 0.09652*cm, 24.76*cm};
    G4double contRodCadTubeDim[3] = {0.0*cm, 0.14732*cm, 24.76*cm};
    G4double contRodAlumTubeDim[3] = {0.0*cm, 0.62357*cm, 40.64*cm};
    G4double contRodZirTubeDim[3] = {1.229*cm, 1.331*cm, 23.2335*cm};

    /* Begining Of Reactor Core Dimensions*/
	// Zirconium lower/upper plate dimensions
	G4double LowerZrDim[3] = {1.331*cm, 11.049*cm, 0.279*cm};
	G4double UpperZrDim[3] = {LowerZrDim[0], LowerZrDim[1], LowerZrDim[2]};

	// Zirconium holes lower/upper plate
	G4double WaterHolesLowerZrDim[3] = {0., 0.262*cm, LowerZrDim[2]};
    G4double WaterHolesUpperZrDim[3] = {0., 0.19*cm, UpperZrDim[2]};
	G4double PinHolesDim[3] = {0., 0.15*cm, LowerZrDim[2]};

    // Hole Position lower/upper plate
	G4double LowerZrHolePos[4] = {0.551815*cm, 0.318516*cm, 0.0, 0.637286*cm};
	G4double UpperZrHolePos[4] = {0.551815*cm, 0.318516*cm, 0.0, 0.637286*cm};

	// Zirconium Rod/Air Gap/Fuel Dimensions
	G4double ZirconiumRodDim[3] = {0., 0.262*cm, 23.3805*cm};
    G4double AirGapDim[3] = {0., 0.212*cm, 23.1105*cm};
    G4double FuelRodDim[3] = {0., 0.2064*cm, 22.6975*cm};
    /* End Of Reactor Core Dimensions*/

    // Lattice Matrix
    G4double latticeMat[25][25] ={{7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7},
                        {7, 7,  7,  7,  7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	8,	8,	8,	8,	7,	7,	7,	7,	7},
                        {7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	8,	8,	9,	8,	9,	8,	9,	8,	8,	7,	7,	7},
                        {7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	8,	9,	8,	9,	9,	8,	8,	9,	9,	8,	8,	8,	7,	7},
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

		// Create world solid
		new G4Box("worldBox", encWorldDim[0]/2, encWorldDim[1]/2, encWorldDim[2]/2);

        // Tub Made of Water
		new G4Tubs("cellTube", 0., reactorDim[1], reactorDim[2]/2, 0., 2.0*CLHEP::pi);

        // Create Aluminium Shell
		new G4Tubs("alumShellTube", alumShellTubeDim[0], alumShellTubeDim[1], alumShellTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("alumShellPlate", alumShellPlateDim[0], alumShellPlateDim[1], alumShellPlateDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4UnionSolid("alumShell", theSolids->GetSolid("alumShellPlate"), theSolids->GetSolid("alumShellTube"), 0, G4ThreeVector(0., 0., 271.*cm));


        // Create Reflector Solids
        new G4Tubs("reflectTop", refTopDim[0], refTopDim[1], refTopDim[2]/2, 0., CLHEP::pi);
		new G4Tubs("reflectAnnulus", refAnnDim[0], refAnnDim[1], refAnnDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("reflectBottom", refBottomDim[0], refBottomDim[1], refBottomDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4UnionSolid("reflector0", theSolids->GetSolid("reflectAnnulus"), theSolids->GetSolid("reflectTop"), 0, G4ThreeVector(0., 0., 13.969375*cm));
		new G4UnionSolid("reflector", theSolids->GetSolid("reflector0"), theSolids->GetSolid("reflectBottom"), 0, G4ThreeVector(0., 0., -16.962*cm));

        // D20 Container
        new G4Tubs("D2OContainer1", D20ContainerDim[0], D20ContainerDim[1], D20ContainerDim[2]/2, 1.570796327*rad, 4.712388980*rad);
        new G4Tubs("D2OContainer2", 0., D20ContainerDim[1]+1.082*cm, D20ContainerDim[2]/2, 2.751192606*rad, 0.7808000945*rad);
        new G4IntersectionSolid("D2OContainer", theSolids->GetSolid("D2OContainer1"), theSolids->GetSolid("D2OContainer2"), 0, G4ThreeVector(1.082*cm,0.,0.));
        new G4Tubs("D2O", D20Dim[0], D20Dim[1], D20Dim[2]/2, 2.751192606*rad, 0.7808000945*rad);

        // Create Aluminium Tube Solids
		new G4Tubs("smallAlumTube", smallAlumTubeDim[0], smallAlumTubeDim[1], smallAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("smallLongAlumTube", smallLongAlumTubeDim[0], smallLongAlumTubeDim[1], smallLongAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("largeAlumTube", largeAlumTubeDim[0], largeAlumTubeDim[1], largeAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);

		// Create Beam Tube Solids
		new G4Tubs("smallBeamTube", smallBTubeDim[0], smallBTubeDim[1], smallBTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("smallLongBeamTube", smallLongBTubeDim[0], smallLongBTubeDim[1], smallLongBTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("cadLinTube", cadLinTubeDim[0], cadLinTubeDim[1], cadLinTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("largeBeamTube", largeBTubeDim[0], largeBTubeDim[1], largeBTubeDim[2]/2, 0., 2.0*CLHEP::pi);

        // Create Control Rod Solids
		new G4Tubs("contRodCentTube", contRodCentTubeDim[0], contRodCentTubeDim[1], contRodCentTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("contRodCadTube", contRodCadTubeDim[0], contRodCadTubeDim[1], contRodCadTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("contRodAlumTube", contRodAlumTubeDim[0], contRodAlumTubeDim[1], contRodAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("contRodZirTube", contRodZirTubeDim[0], contRodZirTubeDim[1], contRodZirTubeDim[2]/2, 0., 2.0*CLHEP::pi);

        /* Beginning Of Reactor Core Geometries*/
        // Create zirconium lower and upper plate
        new G4Tubs("LowerZrTub", LowerZrDim[0], LowerZrDim[1], LowerZrDim[2]/2, 0, 2.0*CLHEP::pi);
        new G4Tubs("UpperZrTub", UpperZrDim[0], UpperZrDim[1], UpperZrDim[2]/2, 0, 2.0*CLHEP::pi);

        // Water Holes lower and upper Zr plate
        new G4Tubs("WaterHolesLower", WaterHolesLowerZrDim[0], WaterHolesLowerZrDim[1], WaterHolesLowerZrDim[2]/2, 0, 2.0*CLHEP::pi);
        new G4Tubs("WaterHolesUpper", WaterHolesUpperZrDim[0], WaterHolesUpperZrDim[1], WaterHolesUpperZrDim[2]/2, 0, 2.0*CLHEP::pi);

        // Pin Holes Lower Zr Plate
        new G4Tubs("PinHolesLower", PinHolesDim[0], PinHolesDim[1], PinHolesDim[2]/2, 0, 2.0*CLHEP::pi);

        // Zirconium Rods
        new G4Tubs("ZirconiumRod", ZirconiumRodDim[0], ZirconiumRodDim[1], ZirconiumRodDim[2]/2+1*mm, 0, 2.0*CLHEP::pi);
        new G4Tubs("AirGapRod", AirGapDim[0], AirGapDim[1], AirGapDim[2]/2, 0, 2.0*CLHEP::pi);
        new G4Tubs("FuelRod", FuelRodDim[0], FuelRodDim[1], FuelRodDim[2]/2, 0, 2.0*CLHEP::pi);

        // Making a union of every part of the core that is made of zirconium
        Zirconium = new G4UnionSolid("Zirconium", theSolids->GetSolid("LowerZrTub"), theSolids->GetSolid("UpperZrTub"), 0, G4ThreeVector(0,0,22.5*cm));
        for(G4int y=1; y<24; y++)
        {
            for(G4int x=1; x<24; x++)
            {
                if(latticeMat[y][x] == 8)
                {

                    G4double Center[2] = {((y-12)*0.551833696+(x-12)*1.103632018)*cm, (12-y)*0.955804*cm};
                    // The extra 1mm is there sch that the two volumes not share the same surface.
                    // This would lead to an undefined scenario.
                    Zirconium = new G4UnionSolid("ZirconiumRods", Zirconium, theSolids->GetSolid("ZirconiumRod"), 0,
                                     G4ThreeVector(Center[0],Center[1],(ZirconiumRodDim[2]+LowerZrDim[2]-1*mm)/2));
                }
            }
        }
        /* End Of Reactor Core Geometries */

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

    //create aluminium shell
    alumShellLogical = new G4LogicalVolume(theSolids->GetSolid("alumShell"),matMap["AlAlloy1"],"alumShellLogical");
    new G4PVPlacement(0, G4ThreeVector(0.,0.,22.5*cm-reactorDim[2]/2), alumShellLogical,"alumShellPhysical",cellLogical,0,0);

    // Reflector Logical Volume is being created
    ReflectorLogical = new G4LogicalVolume(theSolids->GetSolid("reflector"), matMap["Reflector"],"ReflectorLogical");
    new G4PVPlacement(0, G4ThreeVector(0, 0, 53.042*cm-reactorDim[2]/2),  ReflectorLogical, "ReflectorPhysical", cellLogical, 0, 0);

    // Outter Tubes
    outSmallAlumLogical = new G4LogicalVolume(theSolids->GetSolid("smallLongAlumTube"),matMap["AlAlloy1"],"outAlumTubeLogical");
    holePos.set(outAlumTubePos[0]*cos(outAlumTubePos[1]*4+outAlumTubePos[2]),outAlumTubePos[0]*sin(outAlumTubePos[1]*4+outAlumTubePos[2]),302.834*cm-reactorDim[2]/2);
    new G4PVPlacement(0, holePos, outSmallAlumLogical,"outSmallAlumTubePhysical1", cellLogical,0,0);
    holePos.set(outAlumTubePos[0]*cos(outAlumTubePos[1]*1+outAlumTubePos[2]),outAlumTubePos[0]*sin(outAlumTubePos[1]*1+outAlumTubePos[2]),302.834*cm-reactorDim[2]/2);
    new G4PVPlacement(0, holePos, outSmallAlumLogical,"outSmallAlumTubePhysical2", cellLogical,0,0);
    outLargeAlumLogical = new G4LogicalVolume(theSolids->GetSolid("largeAlumTube"),matMap["AlAlloy1"],"outAlumTubeLogical");
    holePos.set(outAlumTubePos[0]*cos(outAlumTubePos[2]),outAlumTubePos[0]*sin(outAlumTubePos[2]), 302.834*cm-reactorDim[2]/2);
    new G4PVPlacement(0, holePos, outLargeAlumLogical,"outLargeAlumTubePhysical", cellLogical,0,0);
    cadLinLogical = new G4LogicalVolume(theSolids->GetSolid("cadLinTube"),matMap["Cadmium"],"cadLinLogical");
    holePos.set(outAlumTubePos[0]*cos(outAlumTubePos[1]*3+outAlumTubePos[2]),outAlumTubePos[0]*sin(outAlumTubePos[1]*3+outAlumTubePos[2]),302.834*cm-reactorDim[2]/2);
    new G4PVPlacement(0, holePos, cadLinLogical,"cadLinTube", cellLogical,0,0);
    new G4PVPlacement(0, holePos, outSmallAlumLogical,"outSmallAlumTubePhysical3", cellLogical,0,0);
    outSmallBeamLogical = new G4LogicalVolume(theSolids->GetSolid("smallLongBeamTube"),matMap["Air"],"outSmallBeamLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), outSmallBeamLogical,"outSmallBeamTubePhysical",outSmallAlumLogical,0,0);
    outLargeBeamLogical = new G4LogicalVolume(theSolids->GetSolid("largeBeamTube"),matMap["Air"],"outLargeBeamLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), outLargeBeamLogical,"outLargeBeamTubePhysical",outLargeAlumLogical,0,0);

    // Inner Irradiation Tubes
    insAlumLogical = new G4LogicalVolume(theSolids->GetSolid("smallAlumTube"),matMap["AlAlloy3"],"insAlumLogical");
    G4int copy =0;
    for (G4int i=0; i<5; i++)
    {
        holePos.set((alumTubePos[0]*cos(alumTubePos[1]*i+alumTubePos[2])), (alumTubePos[0]*sin(alumTubePos[1]*i+alumTubePos[2])), 253.292*cm);
        new G4PVPlacement(0, holePos, insAlumLogical,"insAlumTubePhysical",ReflectorLogical, copy, 0);
        copy++;
    }

    insBeamLogical = new G4LogicalVolume(theSolids->GetSolid("smallBeamTube"),matMap["Air"],"insBeamLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), insBeamLogical,"insBeamTubePhysical",insAlumLogical,0,0);

    // D20 Container
    D2OContainerLogical = new G4LogicalVolume(theSolids->GetSolid("D2OContainer"), matMap["AlAlloy2"],"D2OContainerLogical");
    new G4PVPlacement(0, G4ThreeVector(0, 0, 53.042*cm-reactorDim[2]/2),  D2OContainerLogical, "D2OPhysical", cellLogical, 0, 0);
    D2OLogical = new G4LogicalVolume(theSolids->GetSolid("D2O"), matMap["D2O"],"D2OLogical");
    new G4PVPlacement(0, G4ThreeVector(0, 0, 0.25375*cm),  D2OLogical, "D2OPhysical", D2OContainerLogical, 0, 0);

    // Create the control rod
    contRodZirLogical = new G4LogicalVolume(theSolids->GetSolid("contRodZirTube"),matMap["Zirconium"], "contRodZirLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 53.02925*cm-reactorDim[2]/2), contRodZirLogical,"contRodZirPhysical",cellLogical,0,0);
    contRodAlumLogical = new G4LogicalVolume(theSolids->GetSolid("contRodAlumTube"),matMap["AlAlloy4"], "contRodAlumLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 82.14*cm-reactorDim[2]/2), contRodAlumLogical,"contRodAlumPhysical",cellLogical,0,0);
    contRodCadLogical = new G4LogicalVolume(theSolids->GetSolid("contRodCadTube"),matMap["Cadmium"], "contRodCadLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., -5.44*cm), contRodCadLogical,"contRodCadPhysical",contRodAlumLogical,0,0);
    contRodCentLogical = new G4LogicalVolume(theSolids->GetSolid("contRodCentTube"),matMap["Air"], "contRodCentLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), contRodCentLogical,"contRodCentPhysical",contRodCadLogical,0,0);

    // Creates zirconium ensemble
    ZirconiumLogical = new G4LogicalVolume(Zirconium, matMap["Zirconium"], "ZirconiumLogical");
    new G4PVPlacement(0, G4ThreeVector(0, 0, 41.5535*cm-reactorDim[2]/2),  ZirconiumLogical, "ZirconiumPhysical", cellLogical, 0, 0);

    /* Placing all the logical volumes */
    // Creating holes in lower and Upper Zirconium Plate
    WaterHolesLowerLogical = new G4LogicalVolume(theSolids->GetSolid("WaterHolesLower"), matMap["H2O"], "WaterHolesLowerLogical");
    WaterHolesUpperLogical = new G4LogicalVolume(theSolids->GetSolid("WaterHolesUpper"), matMap["H2O"], "WaterHolesUpperLogical");

    // Creating Air Gap in fuel Assemblie and fuel rod elements
    AirGapLogical = new G4LogicalVolume(theSolids->GetSolid("AirGapRod"), matMap["Air"], "AirGapLogical");
    FuelRodLogical = new G4LogicalVolume(theSolids->GetSolid("FuelRod"), matMap["Fuel"], "FuelRodLogical");

    // Creating the  logical volume for the pins
    LowerPinLogical = new G4LogicalVolume(theSolids->GetSolid("PinHolesLower"), matMap["H2O"], "LowerPinLogical");

    // Placing the Fuel Rod in the center of the Air Gap
    new G4PVPlacement(0, G4ThreeVector(0,0,-0.0875*cm),  FuelRodLogical, "FuelRodPhysical", AirGapLogical, 0, 0);

    G4bool *neighbour;
    G4int copynum = 0, AirGapNum = 0, LowerPinNum = 0;
    for(G4int y=1; y<24; y++)
    {
        for(G4int x=1; x<24; x++)
        {
            if(latticeMat[y][x] != 7)
            {
                G4double Center[2] = {((y-12)*0.551833696+(x-12)*1.103632018)*cm, (12-y)*0.955804*cm};
                if(latticeMat[y-1][x] != 7 && latticeMat[y][x-1] != 7)
                {
                    G4bool DUMMY[6] = {0,0,1,1,1,0};
                    neighbour = DUMMY;
                }
                else if(latticeMat[y-1][x] != 7)
                {
                    G4bool DUMMY[6] = {0,1,1,1,1,0};
                    neighbour = DUMMY;
                }
                else if(latticeMat[y][x-1] != 7)
                {
                    G4bool DUMMY[6] = {0,0,1,1,1,1};
                    neighbour = DUMMY;
                }
                else
                {
                    G4bool DUMMY[6] = {1,1,1,1,1,1};
                    neighbour = DUMMY;
                }


                // Removes the upper left hole
                if(neighbour[0])
                {
                    if( sqrt((Center[0]-LowerZrHolePos[0])*(Center[0]-LowerZrHolePos[0])+(Center[1]+LowerZrHolePos[1])*(Center[1]+LowerZrHolePos[1]))+WaterHolesLowerZrDim[1] < 11.049*cm
                       &&  sqrt((Center[0]-LowerZrHolePos[0])*(Center[0]-LowerZrHolePos[0])+(Center[1]+LowerZrHolePos[1])*(Center[1]+LowerZrHolePos[1]))-WaterHolesLowerZrDim[1] > 1.331*cm)
                    {
                       new G4PVPlacement(0, G4ThreeVector(Center[0]-LowerZrHolePos[0],Center[1]+LowerZrHolePos[1],0),
                                       WaterHolesLowerLogical, "LowerHolesPhysical", ZirconiumLogical, copynum,0);
                        new G4PVPlacement(0, G4ThreeVector(Center[0]-UpperZrHolePos[0],Center[1]+UpperZrHolePos[1],22.5*cm),
                                       WaterHolesUpperLogical, "UpperHolesPhysical", ZirconiumLogical, copynum,0);
                        copynum++;
                    }
                }

                // Removes the lower left hole
                if(neighbour[1])
                {
                    if( sqrt((Center[0]-LowerZrHolePos[0])*(Center[0]-LowerZrHolePos[0])+(Center[1]-LowerZrHolePos[1])*(Center[1]-LowerZrHolePos[1]))+WaterHolesLowerZrDim[1] < 11.049*cm
                       &&  sqrt((Center[0]-LowerZrHolePos[0])*(Center[0]-LowerZrHolePos[0])+(Center[1]-LowerZrHolePos[1])*(Center[1]-LowerZrHolePos[1]))-WaterHolesLowerZrDim[1] > 1.331*cm)
                    {
                        new G4PVPlacement(0, G4ThreeVector(Center[0]-LowerZrHolePos[0],Center[1]-LowerZrHolePos[1],0),
                                       WaterHolesLowerLogical, "LowerHolesPhysical", ZirconiumLogical, copynum,0);
                        new G4PVPlacement(0, G4ThreeVector(Center[0]-UpperZrHolePos[0],Center[1]-UpperZrHolePos[1],22.5*cm),
                                       WaterHolesUpperLogical, "UpperHolesPhysical", ZirconiumLogical, copynum,0);
                        copynum++;
                    }
                }

                // Removes the lower center hole
                if(neighbour[2])
                {
                    if( sqrt((Center[0]-LowerZrHolePos[2])*(Center[0]-LowerZrHolePos[2])+(Center[1]-LowerZrHolePos[3])*(Center[1]-LowerZrHolePos[3]))+WaterHolesLowerZrDim[1] < 11.049*cm
                       &&  sqrt((Center[0]-LowerZrHolePos[2])*(Center[0]-LowerZrHolePos[2])+(Center[1]-LowerZrHolePos[3])*(Center[1]-LowerZrHolePos[3]))-WaterHolesLowerZrDim[1] > 1.331*cm)
                    {
                        new G4PVPlacement(0, G4ThreeVector(Center[0]-LowerZrHolePos[2],Center[1]-LowerZrHolePos[3],0),
                                       WaterHolesLowerLogical, "LowerHolesPhysical", ZirconiumLogical, copynum,0);
                        new G4PVPlacement(0, G4ThreeVector(Center[0]-UpperZrHolePos[2],Center[1]-UpperZrHolePos[3],22.5*cm),
                                       WaterHolesUpperLogical, "UpperHolesPhysical", ZirconiumLogical, copynum,0);
                        copynum++;
                    }
                }

                // Removes the lower right hole
                if(neighbour[3])
                {
                    if( sqrt((Center[0]+LowerZrHolePos[0])*(Center[0]+LowerZrHolePos[0])+(Center[1]-LowerZrHolePos[1])*(Center[1]-LowerZrHolePos[1]))+WaterHolesLowerZrDim[1] < 11.049*cm
                       &&  sqrt((Center[0]+LowerZrHolePos[0])*(Center[0]+LowerZrHolePos[0])+(Center[1]-LowerZrHolePos[1])*(Center[1]-LowerZrHolePos[1]))-WaterHolesLowerZrDim[1] > 1.331*cm)
                    {
                        new G4PVPlacement(0, G4ThreeVector(Center[0]+LowerZrHolePos[0],Center[1]-LowerZrHolePos[1],0),
                                       WaterHolesLowerLogical, "LowerHolesPhysical", ZirconiumLogical, copynum,0);
                        new G4PVPlacement(0, G4ThreeVector(Center[0]+UpperZrHolePos[0],Center[1]-UpperZrHolePos[1],22.5*cm),
                                       WaterHolesUpperLogical, "UpperHolesPhysical", ZirconiumLogical, copynum,0);
                        copynum++;
                    }
                }

                // Removes the upper right hole
                if(neighbour[4])
                {
                    if( sqrt((Center[0]+LowerZrHolePos[0])*(Center[0]+LowerZrHolePos[0])+(Center[1]+LowerZrHolePos[1])*(Center[1]+LowerZrHolePos[1]))+WaterHolesLowerZrDim[1] < 11.049*cm
                       &&  sqrt((Center[0]+LowerZrHolePos[0])*(Center[0]+LowerZrHolePos[0])+(Center[1]+LowerZrHolePos[1])*(Center[1]+LowerZrHolePos[1]))-WaterHolesLowerZrDim[1] > 1.331*cm)
                    {
                        new G4PVPlacement(0, G4ThreeVector(Center[0]+LowerZrHolePos[0],Center[1]+LowerZrHolePos[1],0),
                                       WaterHolesLowerLogical, "LowerHolesPhysical", ZirconiumLogical, copynum,0);
                        new G4PVPlacement(0, G4ThreeVector(Center[0]+UpperZrHolePos[0],Center[1]+UpperZrHolePos[1],22.5*cm),
                                       WaterHolesUpperLogical, "UpperHolesPhysical", ZirconiumLogical, copynum,0);
                        copynum++;
                    }
                }

                // Removes the upper center hole
                if(neighbour[5])
                {
                    if( sqrt((Center[0]+LowerZrHolePos[2])*(Center[0]+LowerZrHolePos[2])+(Center[1]+LowerZrHolePos[3])*(Center[1]+LowerZrHolePos[3]))+WaterHolesLowerZrDim[1] < 11.049*cm
                       &&  sqrt((Center[0]+LowerZrHolePos[2])*(Center[0]+LowerZrHolePos[2])+(Center[1]+LowerZrHolePos[3])*(Center[1]+LowerZrHolePos[3]))-WaterHolesLowerZrDim[1] > 1.331*cm)
                    {
                        new G4PVPlacement(0, G4ThreeVector(Center[0]+LowerZrHolePos[2],Center[1]+LowerZrHolePos[3], 0),
                                       WaterHolesLowerLogical, "LowerHolesPhysical", ZirconiumLogical, copynum, 0);
                        new G4PVPlacement(0, G4ThreeVector(Center[0]+UpperZrHolePos[2],Center[1]+UpperZrHolePos[3], 22.5*cm),
                                       WaterHolesUpperLogical, "UpperHolesPhysical", ZirconiumLogical, copynum, 0);
                        copynum++;
                    }
                }

                if(latticeMat[x][y] == 8)
                {
                    new G4PVPlacement(0, G4ThreeVector(Center[0], Center[1], 11.82975*cm),  AirGapLogical, "AirGapPhysical", ZirconiumLogical, AirGapNum, 0);
                    AirGapNum++;
                }
                else
                {
                    if(Center[0]*Center[0]+ Center[1]*Center[1] > (1.331*cm)*(1.331*cm))
                    {
                        new G4PVPlacement(0, G4ThreeVector(Center[0], Center[1], 0),  LowerPinLogical, "LowerPinPhysical", ZirconiumLogical, LowerPinNum, 0);
                        LowerPinNum++;
                    }
                }
            }
        }
    }


	// Add sensitive detector to ALL logical volumes
	worldLogical->SetSensitiveDetector( sDReactor );
    ZirconiumLogical->SetSensitiveDetector( sDReactor );
    WaterHolesLowerLogical->SetSensitiveDetector( sDReactor );
    WaterHolesUpperLogical->SetSensitiveDetector( sDReactor );
    AirGapLogical->SetSensitiveDetector( sDReactor );
    FuelRodLogical->SetSensitiveDetector( sDReactor );
    LowerPinLogical->SetSensitiveDetector( sDReactor );
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

    if(worldVisAtt)
        delete worldVisAtt;
    if(ZirconiumAtt)
        delete ZirconiumAtt;
	if(WaterHolesLowerAtt)
        delete WaterHolesLowerAtt;
	if(WaterHolesUpperAtt)
        delete WaterHolesUpperAtt;
	if(AirGapAtt)
        delete AirGapAtt;
	if(FuelRodAtt)
        delete FuelRodAtt;
	if(LowerPinAtt)
        delete LowerPinAtt;
	if(ReflectorAtt)
        delete ReflectorAtt;
	if(D2OContainerAtt)
        delete D2OContainerAtt;
	if(D2OAtt)
        delete D2OAtt;
	if(contRodZirVisAtt)
        delete contRodZirVisAtt;
    if(contRodAlumVisAtt)
        delete contRodAlumVisAtt;
    if(contRodCadVisAtt)
        delete contRodCadVisAtt;
    if(contRodCentVisAtt)
        delete contRodCentVisAtt;
    if(insAlumVisAtt)
        delete insAlumVisAtt;
    if(insBeamVisAtt)
        delete insBeamVisAtt;
    if(outSmallAlumVisAtt)
        delete outSmallAlumVisAtt;
    if(outLargeAlumVisAtt)
        delete outLargeAlumVisAtt;
    if(cadLinTubeVisAtt)
        delete cadLinTubeVisAtt;
    if(outSmallBeamVisAtt)
        delete outSmallBeamVisAtt;
    if(outLargeBeamVisAtt)
        delete outLargeBeamVisAtt;
    if(alumShellVisAtt)
        delete alumShellVisAtt;
    if(cellVisAtt)
        delete cellVisAtt;

    worldVisAtt = new G4VisAttributes(G4Colour(0.,0.,210.0/255.0));
    worldVisAtt->SetVisibility(false);
    worldLogical->SetVisAttributes(worldVisAtt);

    // Water Tub Visualization
    cellVisAtt = new G4VisAttributes(G4Colour(47.0/255.0,225.0/255.0,240.0/255.0));
    cellLogical->SetVisAttributes(cellVisAtt);

    // Aluminum Reactor Shell Visualization
    alumShellVisAtt = new G4VisAttributes(G4Colour(150.0/255.0,150.0/255.0,150.0/255.0));
    alumShellLogical->SetVisAttributes(alumShellVisAtt);

    // Zirconium Visualization
    ZirconiumAtt = new G4VisAttributes(G4Colour(0.,0.,0.));
    ZirconiumLogical->SetVisAttributes(ZirconiumAtt);

    // Lowwer Hole Visualization
    WaterHolesLowerAtt = new G4VisAttributes(G4Colour(0.,0.,1.));
    WaterHolesLowerLogical->SetVisAttributes(WaterHolesLowerAtt);

    // Upper Hole Visualization
    WaterHolesUpperAtt = new G4VisAttributes(G4Colour(0.,0.,1.));
    WaterHolesUpperLogical->SetVisAttributes(WaterHolesUpperAtt);

    // Air Gap Visualization
    AirGapAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    AirGapLogical->SetVisAttributes(AirGapAtt);

    // Fuel Visualization
    FuelRodAtt = new G4VisAttributes(G4Colour(0.,1.,0.));
    FuelRodLogical->SetVisAttributes(FuelRodAtt);

    // Lower Pin Visualization
    LowerPinAtt = new G4VisAttributes(G4Colour(0.,0.,1.));
    LowerPinLogical->SetVisAttributes(LowerPinAtt);

    // Reflector Visualization
    ReflectorAtt = new G4VisAttributes(G4Colour(0.,0.5,0.5));
    ReflectorLogical->SetVisAttributes(ReflectorAtt);


    // D2O Column and Water Visualization
    D2OContainerAtt = new G4VisAttributes(G4Colour(0.5,0.5,0.));
    D2OContainerLogical->SetVisAttributes(D2OContainerAtt);
    D2OAtt = new G4VisAttributes(G4Colour(0.,0.,1.));
    D2OLogical->SetVisAttributes(D2OAtt);

    // Control Rod Visualization
    contRodZirVisAtt = new G4VisAttributes(G4Colour(0.,0.,0.));
    contRodZirLogical->SetVisAttributes(contRodZirVisAtt);

    contRodAlumVisAtt = new G4VisAttributes(G4Colour(0.5,0.5,0.5));
    contRodAlumLogical->SetVisAttributes(contRodAlumVisAtt);

    contRodCadVisAtt = new G4VisAttributes(G4Colour(195.0/255.0,174.0/255.0,238.0/255.0));
    contRodCadLogical->SetVisAttributes(contRodCadVisAtt);

    contRodCentVisAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    contRodCentLogical->SetVisAttributes(contRodCentVisAtt);

    // Irradiations Sites Visualization
    insAlumVisAtt = new G4VisAttributes(G4Colour(150.0/255.0,150.0/255.0,150.0/255.0));
    insAlumVisAtt->SetVisibility(false);
    insAlumLogical->SetVisAttributes(insAlumVisAtt);

    insBeamVisAtt = new G4VisAttributes(G4Colour(183.0/255.0,230.0/255.0,240.0/255.0));
    insBeamLogical->SetVisAttributes(insBeamVisAtt);

    outSmallAlumVisAtt = new G4VisAttributes(G4Colour(150.0/255.0,150.0/255.0,150/255.0));
    outSmallAlumLogical->SetVisAttributes(outSmallAlumVisAtt);

    outLargeAlumVisAtt = new G4VisAttributes(G4Colour(150.0/255.0,150.0/255.0,150.0/255.0));
    outLargeAlumLogical->SetVisAttributes(outLargeAlumVisAtt);

    cadLinTubeVisAtt = new G4VisAttributes(G4Colour(195.0/255.0,174.0/255.0,238.0/255.0));
    cadLinLogical->SetVisAttributes(cadLinTubeVisAtt);

    outSmallBeamVisAtt = new G4VisAttributes(G4Colour(183.0/255.0,230.0/255.0,240.0/255.0));
    outSmallBeamLogical->SetVisAttributes(outSmallBeamVisAtt);

    outLargeBeamVisAtt = new G4VisAttributes(G4Colour(183.0/255.0,230.0/255.0,240.0/255.0));
    outLargeBeamLogical->SetVisAttributes(outLargeBeamVisAtt);

    return worldPhysical;
}


// ConstructMaterials()
// Define and build the materials in the C6 lattice cell.
void GuillaumeConstructor::ConstructMaterials()
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
    G4double ReflectorTemp=(22.5+273.15)*kelvin;
    G4double LWTemp=(30.6+273.15)*kelvin;
    G4double FuelTemp=(57.32+273.15)*kelvin;
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
    Reflector->AddElement(Eu151,  2.627389e-7);

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
    AlAlloy1 = new G4Material("Aluminum Alloy", AlAlloyDensity, 5, kStateSolid, AlAlloyTemp1);
    AlAlloy1->AddElement(Al, 0.9792);
    AlAlloy1->AddElement(Si, 0.0060);
    AlAlloy1->AddElement(Cu, 0.0028);
    AlAlloy1->AddElement(Mg, 0.0100);
    AlAlloy1->AddElement(Cr, 0.0020);

    AlAlloy2 = new G4Material("Aluminum Alloy", AlAlloyDensity, 5, kStateSolid, AlAlloyTemp2);
    AlAlloy2->AddElement(Al, 0.9792);
    AlAlloy2->AddElement(Si, 0.0060);
    AlAlloy2->AddElement(Cu, 0.0028);
    AlAlloy2->AddElement(Mg, 0.0100);
    AlAlloy2->AddElement(Cr, 0.0020);

    AlAlloy3 = new G4Material("Aluminum Alloy", AlAlloyDensity, 5, kStateSolid, AlAlloyTemp3);
    AlAlloy3->AddElement(Al, 0.9792);
    AlAlloy3->AddElement(Si, 0.0060);
    AlAlloy3->AddElement(Cu, 0.0028);
    AlAlloy3->AddElement(Mg, 0.0100);
    AlAlloy3->AddElement(Cr, 0.0020);

    AlAlloy4 = new G4Material("Aluminum Alloy", AlAlloyDensity, 5, kStateSolid, AlAlloyTemp4);
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
