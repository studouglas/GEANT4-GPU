#include "DebugConstructor.hh"

DebugConstructor::DebugConstructor()
: StorkVWorldConstructor(), cellLogical(0), alumShellLogical(0), alumContLogical(0), D2OContLogical(0), reflectorLogical(0), insAlumLogical(0), insBeamLogical(0),
                            outSmallAlumLogical(0), outLargeAlumLogical(0), cadLinLogical(0), outSmallBeamLogical(0), outLargeBeamLogical(0), coreWaterLogical(0),
                            coreWaterSliceLogical(0), zircGridLogical(0), airGapsLatLogical(0), /*airGapsLatHLogical(0)*/ airGapsLatHRLogical(0), airGapsLatHR2Logical(0), fuelLatLogical(0),
                            /*fuelLatHLogical(0)*/ fuelLatHRLogical(0), fuelLatHR2Logical(0), contRodZirLogical(0), contRodAlumLogical(0), contRodCadLogical(0), contRodCentLogical(0)
{
    // Set default member variables (from file or default values)
	contRodH = -199.86*cm;

	// Set up variable property map
	variablePropMap[MatPropPair(controlrod,position)] = &contRodH;

	cellVisAtt=NULL;
	alumShellVisAtt=NULL;
	alumContVisAtt=NULL;
	D2OContVisAtt=NULL;
	reflectorVisAtt=NULL;
	insAlumVisAtt=NULL;
	insBeamVisAtt=NULL;
	outSmallAlumVisAtt=NULL;
	outLargeAlumVisAtt=NULL;
	cadLinTubeVisAtt=NULL;
	outSmallBeamVisAtt=NULL;
	outLargeBeamVisAtt=NULL;
	coreWaterVisAtt=NULL;
	coreWaterSliceVisAtt=NULL;
	airGapsLatVisAtt=NULL;
	airGapsLatHVisAtt=NULL;
	fuelLatVisAtt=NULL;
	fuelLatHVisAtt=NULL;
	contRodZirVisAtt=NULL;
	contRodAlumVisAtt=NULL;
	contRodCadVisAtt=NULL;
	contRodCentVisAtt=NULL;

}

DebugConstructor::~DebugConstructor()
{
	// Delete visualization attributes
	if(cellVisAtt)
        delete cellVisAtt;
    if(alumShellVisAtt)
        delete alumShellVisAtt;
    if(alumContVisAtt)
        delete alumContVisAtt;
    if(D2OContVisAtt)
        delete D2OContVisAtt;
    if(reflectorVisAtt)
        delete reflectorVisAtt;
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
    if(zircGridVisAtt)
        delete zircGridVisAtt;
    if(coreWaterVisAtt)
        delete coreWaterVisAtt;
    if(coreWaterSliceVisAtt)
        delete coreWaterSliceVisAtt;
    if(airGapsLatVisAtt)
        delete airGapsLatVisAtt;
    if(airGapsLatHVisAtt)
        delete airGapsLatHVisAtt;
    if(fuelLatVisAtt)
        delete fuelLatVisAtt;
    if(fuelLatHVisAtt)
        delete fuelLatHVisAtt;
    if(contRodZirVisAtt)
        delete contRodZirVisAtt;
    if(contRodAlumVisAtt)
        delete contRodAlumVisAtt;
    if(contRodCadVisAtt)
        delete contRodCadVisAtt;
    if(contRodCentVisAtt)
        delete contRodCentVisAtt;

}

// ConstructWorld()
// Construct the geometry and materials of the SLOWPOKE Reactor.
G4VPhysicalVolume* DebugConstructor::ConstructWorld()
{


	G4SolidStore* theSolids = G4SolidStore::GetInstance();

    // Aluminum tube position
    G4double alumTubePos[4]={14.56182*cm, 0.4*CLHEP::pi, 0., 253.292*cm};
    G4double outAlumTubePos[4]={24.0*cm, 0.4*CLHEP::pi, 0.2*CLHEP::pi,20.834*cm};
    // distance between mother volumes and daughters
    G4ThreeVector disCellToAlumShell = G4ThreeVector(-30.*cm,-30.*cm,11.5*cm);
    G4ThreeVector disCellToAlumCont = G4ThreeVector(-30.*cm,-30.*cm,-228.958*cm);
    G4ThreeVector disAlumContToD2OCont = G4ThreeVector(0.,0.,0.);
    G4ThreeVector disCellToCadLin = G4ThreeVector(0.,0.,-228.958*cm);
    G4ThreeVector disCadLinToOutAlumTube = G4ThreeVector(0.,0.,249.792*cm);
    G4ThreeVector disCellToReflector = G4ThreeVector(-30.*cm,-30.*cm,-228.958*cm);
    G4ThreeVector disCellToUpGrid = G4ThreeVector(-30.*cm, -30.*cm, -217.9465*cm);
    G4ThreeVector disUpGridToLowGrid = G4ThreeVector(0.,0.,-22.5*cm);
    G4ThreeVector disUpGridToSheathe = G4ThreeVector(0.,0.,-10.80975*cm);
    G4ThreeVector disSheatheToAirGaps = G4ThreeVector(0.,0.,0.1395*cm);
    G4ThreeVector disAirGapsToFuel = G4ThreeVector(0., 0., -0.0875*cm);
    G4ThreeVector disCellToContRodZ = G4ThreeVector(-30.*cm, -30.*cm, -228.97075*cm);
    G4ThreeVector disCellToContRodA = G4ThreeVector(-30.*cm, -30.*cm, contRodH);
    G4ThreeVector disContRodAToContRodCad = G4ThreeVector(0., 0., -5.44*cm);
    G4ThreeVector disContRodCadToContRodCent = G4ThreeVector(0., 0., 0.);

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

        // Aluminium Reactor Shell
        G4double alumShellTubeDim[3] = {30.0*cm, 31.0*cm, 541.0*cm};
        G4double alumShellPlateDim[3] = {0.0*cm, 31.0*cm, 1*cm};

        // D2O Column
        G4double alumContPart1TubeDim[3] = {21.2344*cm+0.01, 30.0*cm-0.01, 22.748*cm};
        G4double alumContPart2TubeDim[3] = {0.*cm, 35.0*cm, 30*cm};
        G4ThreeVector alumContTrans = G4ThreeVector(1.530179074*cm, 0., 0.);
        G4double D2OTubeDim[3] = {22.2344*cm, 29.0*cm, 20.6975*cm};

        // Reflector dimensions
        // made top shim a half circle which is is slightly off from the actual shape
        G4double refAnnDim[3] = {11.049*cm+0.1, 21.2344*cm, 22.748*cm};
        G4double refBottomDim[3] = {0.0*cm, 16.113125*cm, 10.16*cm};
        G4double refTopDim[3] = {1.3890625*cm, 12.065*cm, 0.15875*cm};

        // Beamtube dimensions
        G4double smallBTubeDim[3] = {0.0*cm, 1.40208*cm, 515.332*cm};
        G4double smallLongBTubeDim[3] = {0.0*cm, 1.40208*cm, 522.332*cm};
        G4double largeBTubeDim[3] = {0.0*cm, 1.6*cm, 522.332*cm};
        // Aluminium Tube dimensions
        G4double smallAlumTubeDim[3] = {0.0*cm, 1.56718*cm, 515.332*cm};
        G4double smallLongAlumTubeDim[3] = {0.0*cm, 1.56718*cm, 522.332*cm};
        G4double largeAlumTubeDim[3] = {0.0*cm, 1.905*cm, 522.332*cm};
        // Cadmium lining
        G4double cadLinTubeDim[3] = {0.0*cm, 1.61798*cm, 22.748*cm};
        // Control Rod
        G4double contRodCentTubeDim[3] = {0.0*cm, 0.09652*cm, 24.76*cm};
        G4double contRodCadTubeDim[3] = {0.0*cm, 0.14732*cm, 24.76*cm};
        G4double contRodAlumTubeDim[3] = {0.0*cm, 0.62357*cm, 40.64*cm};
        G4double contRodZirTubeDim[3] = {1.229*cm, 1.331*cm-.01, 23.2335*cm};

        // Sheathe dimensions, height extends from bottom of lower grid plate to to the top of the top endcap
        G4double sheatheDim[3] = {0., 0.262*cm , 23.6595*cm};
         // Sheathe dimensions, height extends from bottom of lower grid plate to to the top of the top endcap
        G4double airGapDim[3] = {0., 0.212*cm , 23.1105*cm};
        // Fuel pin dimensions
        G4double fuelPinDim[3] = {0., 0.2064*cm , 22.6975*cm};

        //Lattice Matrix
        G4int latticeMat[13][25] ={{7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7},
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
                            {7,	7,	7,	8,	8,	9,	9,	8,	9,	8,	8,	9,	7,	9,	8,	8,	9,	8,	9,	9,	8,	8,	7,	7,	7}};

        // Lattice cell dimensions
        // hole pattern format radius, angle of increment and the offset angle, check radius
        G4double latticeCellDim[3] = {1.103667393*cm, 0.955804*cm, 0.279*cm};
        G4double gridPlateDim[3] = {1.331*cm, 11.049*cm, 0.279*cm};
        G4double lowGridHolesDim[3] = {0., 0.262*cm, 0.3*cm};
        G4double lowGridCentreHoleDim[3] = {0., 0.15*cm, 0.3*cm};
        G4double upperGridHolesDim[3]= {0., 0.19*cm, 0.3*cm};
        G4double holePat[3]= {0.637286*cm, CLHEP::pi/3, CLHEP::pi/6};
        G4double unitRegDim[6]={0, (gridPlateDim[1]-gridPlateDim[0])/10, 0., CLHEP::pi/(3*10), 0., 0.3*cm};
        G4double regDim[6]={gridPlateDim[0], gridPlateDim[1], CLHEP::pi/3, 2*CLHEP::pi/3, -gridPlateDim[2]/2, gridPlateDim[2]/2};


		//Geometry positioning data
		std::stringstream latticeType1;
		G4ThreeVector latCellPos;
		G4ThreeVector holePos;
		G4ThreeVector holeRPos;

		// Create world solid
		new G4Box("worldBox", encWorldDim[0]/2, encWorldDim[1]/2, encWorldDim[2]/2);

		// Create cell solid
		new G4Tubs("cellTube", 0., cellDim[1], cellDim[2]/2, 0., 2.0*CLHEP::pi);

				// Create Aluminium Shell
		new G4Tubs("alumShellTube", alumShellTubeDim[0], alumShellTubeDim[1], alumShellTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("alumShellPlate", alumShellPlateDim[0], alumShellPlateDim[1], alumShellPlateDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4UnionSolid("alumShell", theSolids->GetSolid("alumShellTube"), theSolids->GetSolid("alumShellPlate"), 0, G4ThreeVector(0., 0., -271.*cm));

		// Create D2O Column
		new G4Tubs("alumContPart1Tube", alumContPart1TubeDim[0], alumContPart1TubeDim[1], alumContPart1TubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("alumContPart2Tube", alumContPart2TubeDim[0], alumContPart2TubeDim[1], alumContPart2TubeDim[2]/2, 3.5319927*rad, 5.502385213*rad);
		new G4SubtractionSolid("alumContTube", theSolids->GetSolid("alumContPart1Tube"), theSolids->GetSolid("alumContPart2Tube"), 0, alumContTrans);
		new G4Tubs("D2OTube", D2OTubeDim[0], D2OTubeDim[1], D2OTubeDim[2]/2, 2.751192606*rad, 0.780800094*rad);

		// Create Reflector Solids
		new G4Tubs("reflectTop", refTopDim[0], refTopDim[1], refTopDim[2]/2, 0., CLHEP::pi);
		new G4Tubs("reflectAnnulus", refAnnDim[0], refAnnDim[1], refAnnDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("reflectBottom", refBottomDim[0], refBottomDim[1], refBottomDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4UnionSolid("reflectorP1", theSolids->GetSolid("reflectAnnulus"), theSolids->GetSolid("reflectTop"), 0, G4ThreeVector(0., 0., 13.969375*cm));
		new G4UnionSolid("reflector", theSolids->GetSolid("reflectorP1"), theSolids->GetSolid("reflectBottom"), 0, G4ThreeVector(0., 0., -16.962*cm));

        // Create Aluminium Tube Solids
		new G4Tubs("smallAlumTube", smallAlumTubeDim[0], smallAlumTubeDim[1], smallAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("smallLongAlumTube", smallLongAlumTubeDim[0], smallLongAlumTubeDim[1], smallLongAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("largeAlumTube", largeAlumTubeDim[0], largeAlumTubeDim[1], largeAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);

		// Create Beam Tube Solids
		new G4Tubs("smallBeamTube", smallBTubeDim[0], smallBTubeDim[1], smallBTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("smallLongBeamTube", smallLongBTubeDim[0], smallLongBTubeDim[1], smallLongBTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("cadLinTube", cadLinTubeDim[0], cadLinTubeDim[1], cadLinTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("largeBeamTube", largeBTubeDim[0], largeBTubeDim[1], largeBTubeDim[2]/2, 0., 2.0*CLHEP::pi);

        // Create Fuel Bundle

//        //create the mothervolume for a 1/3 slice of the fuel bundle (contains a reflected pair of 1/6 slices of the fuel bundle)
        new G4Box("testBox",  0.5*cm, 0.5*cm, 0.5*cm);
//
        //create the mothervolume containing the full fuel bundle (contains the replication of core water slice)
        new G4Tubs("coreWater",  contRodZirTubeDim[1], refAnnDim[0], sheatheDim[2]/2, 0., 2*CLHEP::pi);
        //create the mothervolume for a 1/3 slice of the fuel bundle (contains a reflected pair of 1/6 slices of the fuel bundle)
        new G4Tubs("coreWaterSlice",  contRodZirTubeDim[1], refAnnDim[0], sheatheDim[2]/2, CLHEP::pi/3, 2*CLHEP::pi/3);

        //creates the base zirconium grid slice that the holes will be subtracted from to form the upper and lower grid plate
        new G4Tubs("gridPlate",  gridPlateDim[0], gridPlateDim[1], gridPlateDim[2]/2, CLHEP::pi/3, CLHEP::pi/3);

		// creates the three types of holes to be used to create the two hole patterns, the upGridHolesLat and lowGridHolesLat
		new G4Tubs("upGridHole", upperGridHolesDim[0], upperGridHolesDim[1], upperGridHolesDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("lowGridHole", lowGridHolesDim[0], lowGridHolesDim[1], lowGridHolesDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("lowGridCentreHole", lowGridCentreHoleDim[0], lowGridCentreHoleDim[1], lowGridCentreHoleDim[2]/2, 0., 2.0*CLHEP::pi);
        // Create the sheathe for fuel pins
		new G4Tubs("sheatheTube", sheatheDim[0], sheatheDim[1], sheatheDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("sheatheTubeHR", sheatheDim[0], sheatheDim[1], sheatheDim[2]/2, CLHEP::pi/3, CLHEP::pi);
		new G4Tubs("sheatheTubeHR2", sheatheDim[0], sheatheDim[1], sheatheDim[2]/2, 5*CLHEP::pi/3, CLHEP::pi);

		// Create the Air Gaps between the fuel and the sheathe
		new G4Tubs("airGaps", airGapDim[0], airGapDim[1], airGapDim[2]/2, 0., 2.0*CLHEP::pi);
//		new G4Tubs("airGapsH", airGapDim[0], airGapDim[1], airGapDim[2]/2, 0., CLHEP::pi);
		new G4Tubs("airGapsHR", airGapDim[0], airGapDim[1], airGapDim[2]/2, 2*CLHEP::pi/3, CLHEP::pi);
		new G4Tubs("airGapsHR2", airGapDim[0], airGapDim[1], airGapDim[2]/2, CLHEP::pi/3, CLHEP::pi);

		// Create a fuel pins
		new G4Tubs("fuelPin", fuelPinDim[0], fuelPinDim[1], fuelPinDim[2]/2, 0., 2.0*CLHEP::pi);
//		new G4Tubs("fuelPinH", fuelPinDim[0], fuelPinDim[1], fuelPinDim[2]/2, 0., CLHEP::pi);
		new G4Tubs("fuelPinHR", fuelPinDim[0], fuelPinDim[1], fuelPinDim[2]/2, 2*CLHEP::pi/3, CLHEP::pi);
		new G4Tubs("fuelPinHR2", fuelPinDim[0], fuelPinDim[1], fuelPinDim[2]/2, CLHEP::pi/3, CLHEP::pi);

        solidList upperGridHoles;
        solidList lowerGridHoles;
        solidList sheatheTubes;
        G4int i=3;
        G4int j=12;
        G4double rho;
        G4double phi;
		while (i<12)
		{
            while (j<25)
            {
                if(latticeMat[i][j]==8)
                {
                    latCellPos.set(((12-i)*(-latticeCellDim[0]*0.5)+(j-12)*latticeCellDim[0]), ((12-i)*(latticeCellDim[1])), 0.);
                    for(G4int k=0; k<6; k++)
                    {
                        holeRPos.set(holePat[0]*cos(holePat[1]*(k)+holePat[2]), holePat[0]*sin(holePat[1]*(k)+holePat[2]),0.);
                        holePos=latCellPos+holeRPos;
                        phi = holePos.phi();
                        rho = holePos.rho();

                        if((phi>=CLHEP::pi/3)&&(phi<=2*CLHEP::pi/3)&&(rho>gridPlateDim[0])&&(rho<gridPlateDim[1]))
                        {
                            upperGridHoles.push_back(std::make_pair(theSolids->GetSolid("upGridHole"), holePos));
                            lowerGridHoles.push_back(std::make_pair(theSolids->GetSolid("lowGridHole"), holePos));
                        }
                    }
                    latticeType1.str("");
                    if(i==j)
                        latticeType1 << "HR";
                    else if(j==12)
                        latticeType1 << "HR2";

                    sheatheTubes.push_back(std::make_pair(theSolids->GetSolid("sheatheTube"+latticeType1.str()), latCellPos));
                }
                else if(latticeMat[i][j]==9)
                {
                    latCellPos.set(((12-i)*(-latticeCellDim[0]*0.5)+(j-12)*latticeCellDim[0]), ((12-i)*(latticeCellDim[1])), 0.);
                    for(G4int k=0; k<6; k++)
                    {
                        holeRPos.set(holePat[0]*cos(holePat[1]*(k)+holePat[2]), holePat[0]*sin(holePat[1]*(k)+holePat[2]),0.);
                        holePos=latCellPos+holeRPos;
                        phi = holePos.phi();
                        rho = holePos.rho();

                        if((phi>=CLHEP::pi/3)&&(phi<=2*CLHEP::pi/3)&&(rho>gridPlateDim[0])&&(rho<gridPlateDim[1]))
                        {
                            upperGridHoles.push_back(std::make_pair(theSolids->GetSolid("upGridHole"), holePos));
                            lowerGridHoles.push_back(std::make_pair(theSolids->GetSolid("lowGridHole"), holePos));
                        }
                    }

                    lowerGridHoles.push_back(std::make_pair(theSolids->GetSolid("lowGridCentreHole"), latCellPos));
                }
                j++;
            }
            j=12;
            i++;
		}

		UnionBinaryTree* upGridHolesLat = new UnionBinaryTree(&upperGridHoles);
		//UnionBinaryTree* lowGridHolesLat = new UnionBinaryTree(&lowerGridHoles);
		//UnionBinaryTree* sheatheTubeLat = new UnionBinaryTree(&sheatheTubes);

		solidPos upGridHolesLatPair = upGridHolesLat->GetUnionSolid("upGridHolesLat", 0, cylUnit, unitRegDim, regDim, 0.0, radCyl, 1.0, NULL, true);
//		solidPos lowGridHolesLatPair = lowGridHolesLat->GetUnionSolid("lowGridHolesLat", 0, cylUnit, unitRegDim, regDim, 0.0, radCyl, 1.0, NULL, true);
//		solidPos sheatheTubeLatPair = sheatheTubeLat->GetUnionSolid("sheatheTubeLat", 0, cylUnit, unitRegDim, regDim, 0.0, radCyl, 1.0, NULL, true);

		// creates the upGridPlate and the lowGridPlate from the unions of the upGridHolesLat and lowGridHolesLat with the base gridPlate
		new G4SubtractionSolid("upGridPlate", theSolids->GetSolid("gridPlate"), upGridHolesLatPair.first, 0, upGridHolesLatPair.second);
//		new G4SubtractionSolid("lowGridPlate", theSolids->GetSolid("gridPlate"), lowGridHolesLatPair.first, 0, lowGridHolesLatPair.second);
//
//        // creates the zirconium grid slice (zircGridPlate) from the union of the upGridPlate and the lowGridPlate
//		new G4UnionSolid("zircGridSliceP1", theSolids->GetSolid("upGridPlate"), theSolids->GetSolid("lowGridPlate"), 0, disUpGridToLowGrid);
//		new G4UnionSolid("zircGridSlice", theSolids->GetSolid("zircGridSliceP1"), sheatheTubeLatPair.first, 0, sheatheTubeLatPair.second);



		//creates a reflection of the zirconium grid slice
//		new G4ReflectedSolid("zircGridSliceRefl", theSolids->GetSolid("zircGridSlice") , G4ReflectY3D());
//
//        zRot->rotateZ(-CLHEP::pi/3);
//
//        //creates the full ziconium grid by adding up the slices
//		new G4UnionSolid("zircGridPlate1/3", theSolids->GetSolid("zircGridSlice"), theSolids->GetSolid("zircGridSliceRefl"), 0, G4ThreeVector(0.,0.,0.));
//
//		new G4UnionSolid("zircGridPlate2/3", theSolids->GetSolid("zircGridPlate1/3"), theSolids->GetSolid("zircGridPlate1/3"), G4Transform3D(*zRot, originHoleSec[0]));
//
//        zRot->rotateZ(-2*CLHEP::pi/3);
//
//		new G4UnionSolid("zircGridPlate", theSolids->GetSolid("zircGridPlate2/3"), theSolids->GetSolid("zircGridPlate1/3"), G4Transform3D(*zRot, originHoleSec[0]));


        // Created the Control Rod Solids
		new G4Tubs("contRodCentTube", contRodCentTubeDim[0], contRodCentTubeDim[1], contRodCentTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("contRodCadTube", contRodCadTubeDim[0], contRodCadTubeDim[1], contRodCadTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("contRodAlumTube", contRodAlumTubeDim[0], contRodAlumTubeDim[1], contRodAlumTubeDim[2]/2, 0., 2.0*CLHEP::pi);
		new G4Tubs("contRodZirTube", contRodZirTubeDim[0], contRodZirTubeDim[1], contRodZirTubeDim[2]/2, 0., 2.0*CLHEP::pi);

		geomChanged = false;
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

    // all objects inside the cell are offset by (-30.0cm,-30cm,0)
    //create aluminium shell
    alumShellLogical = new G4LogicalVolume(theSolids->GetSolid("alumShell"),
									  matMap["AlAlloy1"],"alumShellLogical");
    new G4PVPlacement(0, disCellToAlumShell, alumShellLogical,"alumShellPhysical",
                      cellLogical,0,0);

    alumContLogical = new G4LogicalVolume(theSolids->GetSolid("alumContTube"),
									  matMap["AlAlloy2"],"alumContLogical");
    new G4PVPlacement(0, disCellToAlumCont, alumContLogical,"alumContPhysical", cellLogical,0,0);

    //create heavy water
    D2OContLogical = new G4LogicalVolume(theSolids->GetSolid("D2OTube"),
									  matMap["D2O"],"D2OContLogical");
    new G4PVPlacement(0, disAlumContToD2OCont, D2OContLogical,"D2OContPhysical",
                      alumContLogical,0,0);

     //create aluminium container
    reflectorLogical = new G4LogicalVolume(theSolids->GetSolid("reflector"),
									  matMap["Reflector"],"reflectorLogical");
    new G4PVPlacement(0, disCellToReflector, reflectorLogical,"reflectorPhysical",
                      cellLogical,0,0);

    insAlumLogical = new G4LogicalVolume(theSolids->GetSolid("smallAlumTube"),
									  matMap["AlAlloy3"],"insAlumLogical");

    for (G4int i=0; i<5; i++)
		{
            volName.str("");
            volName << i;
            volPos.set((alumTubePos[0]*cos(alumTubePos[1]*i+alumTubePos[2])), (alumTubePos[0]*sin(alumTubePos[1]*i+alumTubePos[2])), alumTubePos[3]);
            new G4PVPlacement(0, volPos, insAlumLogical,"insAlumTubePhysical"+volName.str(), reflectorLogical,0,0);
		}

    insBeamLogical = new G4LogicalVolume(theSolids->GetSolid("smallBeamTube"),
									  matMap["Air"],"insBeamLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), insBeamLogical,"insBeamTubePhysical",
                      insAlumLogical,0,0);


    outSmallAlumLogical = new G4LogicalVolume(theSolids->GetSolid("smallLongAlumTube"),
									  matMap["AlAlloy1"],"outAlumTubeLogical");
    outLargeAlumLogical = new G4LogicalVolume(theSolids->GetSolid("largeAlumTube"),
									  matMap["AlAlloy1"],"outAlumTubeLogical");
    cadLinLogical = new G4LogicalVolume(theSolids->GetSolid("cadLinTube"),
									  matMap["Cadmium"],"cadLinLogical");

    volPos.set(outAlumTubePos[0]*cos(outAlumTubePos[2])-30.0*cm,
            outAlumTubePos[0]*sin(outAlumTubePos[2])-30.0*cm, outAlumTubePos[3]);

    new G4PVPlacement(0, volPos, outLargeAlumLogical,"outLargeAlumTubePhysical", cellLogical,0,0);

    volPos.set(outAlumTubePos[0]*cos(outAlumTubePos[1]*3+outAlumTubePos[2])-30.0*cm,
            outAlumTubePos[0]*sin(outAlumTubePos[1]*3+outAlumTubePos[2])-30.0*cm,0.);

    new G4PVPlacement(0, volPos+disCellToCadLin, cadLinLogical,"cadLinTubePhysical", cellLogical,0,0);
    new G4PVPlacement(0, disCadLinToOutAlumTube, outSmallAlumLogical,"outSmallAlumTubePhysical2", cadLinLogical,0,0);

    volPos.set(outAlumTubePos[0]*cos(outAlumTubePos[1]*1+outAlumTubePos[2])-30.0*cm,
            outAlumTubePos[0]*sin(outAlumTubePos[1]*1+outAlumTubePos[2])-30.0*cm, outAlumTubePos[3]);

    new G4PVPlacement(0, volPos, outSmallAlumLogical,"outSmallAlumTubePhysical1", cellLogical,0,0);

    volPos.set(outAlumTubePos[0]*cos(outAlumTubePos[1]*4+outAlumTubePos[2])-30.0*cm,
            outAlumTubePos[0]*sin(outAlumTubePos[1]*4+outAlumTubePos[2])-30.0*cm, outAlumTubePos[3]);

    new G4PVPlacement(0, volPos, outSmallAlumLogical,"outSmallAlumTubePhysical3", cellLogical,0,0);



    outSmallBeamLogical = new G4LogicalVolume(theSolids->GetSolid("smallLongBeamTube"),
									  matMap["Air"],"outSmallBeamLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), outSmallBeamLogical,"outSmallBeamTubePhysical",
                      outSmallAlumLogical,0,0);
    outLargeBeamLogical = new G4LogicalVolume(theSolids->GetSolid("largeBeamTube"),
									  matMap["Air"],"outLargeBeamLogical");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), outLargeBeamLogical,"outLargeBeamTubePhysical",
                      outLargeAlumLogical,0,0);

    //Create fuel bundle
    //mother volume for replicas
//    coreWaterLogical = new G4LogicalVolume(theSolids->GetSolid("coreWater"),
//										 matMap["H2O"], "coreWaterLogical");
//    new G4PVPlacement(0, disCellToUpGrid+disUpGridToSheathe, coreWaterLogical,"coreWaterPhysical",
//                      cellLogical,0,0);
//
//    //mother volume for the reflected pair of the fuel bundle slice
//    coreWaterSliceLogical = new G4LogicalVolume(theSolids->GetSolid("coreWaterSlice"),
//										 matMap["H2O"], "coreWaterSliceLogical");
//
//    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), coreWaterSliceLogical,"coreWaterSlicePhysical",
//                      coreWaterLogical,0,0);

//    //create Zirconium grid plates
    zircGridLogical = new G4LogicalVolume(theSolids->GetSolid("upGridPlate"), matMap["Zirconium"], "zircGridLogical");

    new G4PVPlacement(0, disCellToUpGrid, zircGridLogical,"zircGridSlicePhysical",
                      cellLogical,0,0,0);

//create Zirconium grid plates
//    zircGridLogical = new G4LogicalVolume(theSolids->GetSolid("testBox"), matMap["Zirconium"], "zircGridLogical");
//
//    new G4PVPlacement(0, G4ThreeVector(-5.0*cm,0.,0.), zircGridLogical,"zircGridSlicePhysical",
//                      coreWaterSliceLogical,0,0,0);

//// Create the air gaps
//    airGapsLatLogical = new G4LogicalVolume(theSolids->GetSolid("airGaps"),
//										 matMap["Zirconium"], "airGapsLatLogical");
//
////    airGapsLatHLogical = new G4LogicalVolume(theSolids->GetSolid("airGapsH"),
////										 matMap["Zirconium"], "airGapsLatLogical");
//
//    airGapsLatHRLogical = new G4LogicalVolume(theSolids->GetSolid("airGapsHR"),
//										 matMap["Zirconium"], "airGapsLatLogical");
//
//    airGapsLatHR2Logical = new G4LogicalVolume(theSolids->GetSolid("airGapsHR2"),
//										 matMap["Zirconium"], "airGapsLat2Logical");
//
//    fuelLatLogical = new G4LogicalVolume(theSolids->GetSolid("fuelPin"), matMap["Fuel"], "fuelLatLogical");
//    new G4PVPlacement(0, disAirGapsToFuel, fuelLatLogical,"fuelLatPhysical", airGapsLatLogical,0,0,0);
//
////    fuelLatHLogical = new G4LogicalVolume(theSolids->GetSolid("fuelPinH"), matMap["Fuel"], "fuelLatHLogical");
////    new G4PVPlacement(0, disAirGapsToFuel, fuelLatHLogical,"fuelLatHPhysical", airGapsLatHLogical,0,0,0);
//
//    fuelLatHRLogical = new G4LogicalVolume(theSolids->GetSolid("fuelPinHR"), matMap["Fuel"], "fuelLatHRLogical");
//    new G4PVPlacement(0, disAirGapsToFuel, fuelLatHRLogical,"fuelLatHRPhysical", airGapsLatHRLogical,0,0,0);
//
//    fuelLatHR2Logical = new G4LogicalVolume(theSolids->GetSolid("fuelPinHR2"), matMap["Fuel"], "fuelLatHR2Logical");
//    new G4PVPlacement(0, disAirGapsToFuel, fuelLatHR2Logical,"fuelLatHR2Physical", airGapsLatHR2Logical,0,0,0);
//
//    G4int k=0;
//    G4int i=3;
//    G4int j=1;
//    G4int g=13;
//    G4bool check=false;
//
//    while (i<22)
//    {
//        if(i>12)
//        {
//            g=g-1;
//            check=true;
//        }
//
//        while (j<g)
//        {
//            if(latticeMat[i][j]==8)
//            {
//                volName.str("");
//                volName << k;
//
//                volPos.set(((12-i)*(-latticeCellDim[0]*0.5)+(j-12)*latticeCellDim[0]), ((12-i)*(latticeCellDim[1])), disUpGridToSheathe[2]+disSheatheToAirGaps[2]);
//
//                if (!((j==12)||(j==g&&check)))
//                    new G4PVPlacement(0, volPos, airGapsLatLogical,"airGapsLatPhysical"+volName.str(),
//                                                                       zircGridLogical,0,0,0);
//                else if (j==12)
//                    new G4PVPlacement(0, volPos, airGapsLatHRLogical,"airGapsLatPhysical"+volName.str(),
//                                                                       zircGridLogical,0,0,0);
//                else
//                    new G4PVPlacement(0, volPos, airGapsLatHR2Logical,"airGapsLatPhysical"+volName.str(),
//                                                                       zircGridLogical,0,0,0);
//                k++;
//            }
//            j++;
//        }
//    j=1;
//     i++;
//    }


    // Place the reflected part using G4ReflectionFactory
//    G4ReflectionFactory::Instance()->Place(G4ReflectY3D(), "reflZircGridSlice",
//                                         zircGridLogical, coreWaterSliceLogical, 0, 0);

    //Replicate the mother volume of the reflected slices
//    new G4PVReplica("zircGridPhysical", coreWaterSliceLogical, coreWaterLogical, kPhi, 3, 2*CLHEP::pi/3);

//    G4RotationMatrix* zRot = new G4RotationMatrix();
//    zRot->rotateZ(2*CLHEP::pi/3);
//
//    new G4PVPlacement(G4Transform3D(*zRot, G4ThreeVector(0., 0., 0.)), coreWaterSliceLogical,"coreWaterSlicePhysical1",
//                      coreWaterLogical,0,0);
//
//    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), coreWaterSliceLogical,"coreWaterSlicePhysical2",
//                      coreWaterLogical,0,0);
//
//    zRot->rotateZ(2*CLHEP::pi/3);
//    new G4PVPlacement(G4Transform3D(*zRot, G4ThreeVector(0., 0., 0.)), coreWaterSliceLogical,"coreWaterSlicePhysical3",
//                      coreWaterLogical,0,0);
//
//    delete zRot;

    //Create the control rod
    contRodZirLogical = new G4LogicalVolume(theSolids->GetSolid("contRodZirTube"),
									  matMap["Zirconium"], "contRodZirLogical");
    new G4PVPlacement(0, disCellToContRodZ, contRodZirLogical,"contRodZirPhysical",
                      cellLogical,0,0);

    contRodAlumLogical = new G4LogicalVolume(theSolids->GetSolid("contRodAlumTube"),
									  matMap["AlAlloy4"], "contRodAlumLogical");
    new G4PVPlacement(0, disCellToContRodA, contRodAlumLogical,"contRodAlumPhysical",
                      cellLogical,0,0);

    contRodCadLogical = new G4LogicalVolume(theSolids->GetSolid("contRodCadTube"),
									  matMap["Cadmium"], "contRodCadLogical");
    new G4PVPlacement(0, disContRodAToContRodCad, contRodCadLogical,"contRodCadPhysical",
                      contRodAlumLogical,0,0);

    contRodCentLogical = new G4LogicalVolume(theSolids->GetSolid("contRodCentTube"),
									  matMap["Air"], "contRodCentLogical");
    new G4PVPlacement(0, disContRodCadToContRodCent, contRodCentLogical,"contRodCentPhysical",
                      contRodCadLogical,0,0);

	// Add sensitive detector to ALL logical volumes
	worldLogical->SetSensitiveDetector( sDReactor );
	cellLogical->SetSensitiveDetector( sDReactor );
	alumShellLogical->SetSensitiveDetector( sDReactor );
	alumContLogical->SetSensitiveDetector( sDReactor );
	D2OContLogical->SetSensitiveDetector( sDReactor );
	reflectorLogical->SetSensitiveDetector( sDReactor );
	insAlumLogical->SetSensitiveDetector( sDReactor );
	insBeamLogical->SetSensitiveDetector( sDReactor );
	outSmallAlumLogical->SetSensitiveDetector( sDReactor );
	outLargeAlumLogical->SetSensitiveDetector( sDReactor );
	cadLinLogical->SetSensitiveDetector( sDReactor );
	outSmallBeamLogical->SetSensitiveDetector( sDReactor );
	outLargeBeamLogical->SetSensitiveDetector( sDReactor );
//	coreWaterLogical->SetSensitiveDetector( sDReactor );
//	coreWaterSliceLogical->SetSensitiveDetector( sDReactor );
	zircGridLogical->SetSensitiveDetector( sDReactor );
//	airGapsLatLogical->SetSensitiveDetector( sDReactor );
////	airGapsLatHLogical->SetSensitiveDetector( sDReactor );
//	airGapsLatHRLogical->SetSensitiveDetector( sDReactor );
//	airGapsLatHR2Logical->SetSensitiveDetector( sDReactor );
//	fuelLatLogical->SetSensitiveDetector( sDReactor );
////	fuelLatHLogical->SetSensitiveDetector( sDReactor );
//	fuelLatHRLogical->SetSensitiveDetector( sDReactor );
//	fuelLatHR2Logical->SetSensitiveDetector( sDReactor );
	contRodZirLogical->SetSensitiveDetector( sDReactor );
	contRodAlumLogical->SetSensitiveDetector( sDReactor );
	contRodCadLogical->SetSensitiveDetector( sDReactor );
	contRodCentLogical->SetSensitiveDetector( sDReactor );


    // Set visualization attributes

    if(worldVisAtt)
        delete worldVisAtt;
    if(cellVisAtt)
        delete cellVisAtt;
    if(alumShellVisAtt)
        delete alumShellVisAtt;
    if(alumContVisAtt)
        delete alumContVisAtt;
    if(D2OContVisAtt)
        delete D2OContVisAtt;
    if(reflectorVisAtt)
        delete reflectorVisAtt;
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
    if(zircGridVisAtt)
        delete zircGridVisAtt;
    if(coreWaterVisAtt)
        delete coreWaterVisAtt;
    if(coreWaterSliceVisAtt)
        delete coreWaterSliceVisAtt;
    if(airGapsLatVisAtt)
        delete airGapsLatVisAtt;
    if(airGapsLatHVisAtt)
        delete airGapsLatHVisAtt;
    if(fuelLatVisAtt)
        delete fuelLatVisAtt;
    if(fuelLatHVisAtt)
        delete fuelLatHVisAtt;
    if(contRodZirVisAtt)
        delete contRodZirVisAtt;
    if(contRodAlumVisAtt)
        delete contRodAlumVisAtt;
    if(contRodCadVisAtt)
        delete contRodCadVisAtt;
    if(contRodCentVisAtt)
        delete contRodCentVisAtt;

    worldVisAtt = new G4VisAttributes(G4Colour(1.,1.,1.));
    worldVisAtt->SetVisibility(false);
    worldLogical->SetVisAttributes(worldVisAtt);

// light blue
    cellVisAtt = new G4VisAttributes(G4Colour(47.0/255.0,225.0/255.0,240.0/255.0));
    cellVisAtt->SetVisibility(false);
    cellLogical->SetVisAttributes(cellVisAtt);

//dark yellow
    alumShellVisAtt = new G4VisAttributes(G4Colour(210.0/255.0,172.0/255.0,0.0/255.0));
    alumShellVisAtt->SetVisibility(false);
    alumShellLogical->SetVisAttributes(alumShellVisAtt);

//light orange
    alumContVisAtt = new G4VisAttributes(G4Colour(255.0/255.0,173.0/255.0,0.0/255.0));
    alumContVisAtt->SetVisibility(false);
    alumContLogical->SetVisAttributes(alumContVisAtt);

//dark blue
    D2OContVisAtt = new G4VisAttributes(G4Colour(0.,0.,210.0/255.0));
    D2OContVisAtt->SetVisibility(false);
    D2OContLogical->SetVisAttributes(D2OContVisAtt);

//dark gray
    reflectorVisAtt = new G4VisAttributes(G4Colour(45.0/255.0,45.0/255.0,45.0/255.0));
    reflectorVisAtt->SetVisibility(false);
    reflectorLogical->SetVisAttributes(reflectorVisAtt);

//light purple
    insAlumVisAtt = new G4VisAttributes(G4Colour(192.0/255.0,0.0/255.0,255.0/255.0));
    insAlumVisAtt->SetVisibility(false);
    insAlumLogical->SetVisAttributes(insAlumVisAtt);

    insBeamVisAtt = new G4VisAttributes(G4Colour(183.0/255.0,230.0/255.0,240.0/255.0));
    insBeamVisAtt->SetVisibility(false);
    insBeamLogical->SetVisAttributes(insBeamVisAtt);

//dark purple
    outSmallAlumVisAtt = new G4VisAttributes(G4Colour(70.0/255.0,0.0/255.0,74.0/255.0));
    outSmallAlumVisAtt->SetVisibility(false);
    outSmallAlumLogical->SetVisAttributes(outSmallAlumVisAtt);

//dark green
    outLargeAlumVisAtt = new G4VisAttributes(G4Colour(63.0/255.0,119.0/255.0,0.0/255.0));
    outLargeAlumVisAtt->SetVisibility(false);
    outLargeAlumLogical->SetVisAttributes(outLargeAlumVisAtt);

//dark orange
    cadLinTubeVisAtt = new G4VisAttributes(G4Colour(164.0/255.0,57.0/255.0,0.0/255.0));
    cadLinTubeVisAtt->SetVisibility(false);
    cadLinLogical->SetVisAttributes(cadLinTubeVisAtt);

    outSmallBeamVisAtt = new G4VisAttributes(G4Colour(183.0/255.0,230.0/255.0,240.0/255.0));
    outSmallBeamVisAtt->SetVisibility(false);
    outSmallBeamLogical->SetVisAttributes(outSmallBeamVisAtt);

    outLargeBeamVisAtt = new G4VisAttributes(G4Colour(183.0/255.0,230.0/255.0,240.0/255.0));
    outLargeBeamVisAtt->SetVisibility(false);
    outLargeBeamLogical->SetVisAttributes(outLargeBeamVisAtt);

//    coreWaterVisAtt = new G4VisAttributes(G4Colour(47.0/255.0,225.0/255.0,240.0/255.0));
//    coreWaterVisAtt->SetVisibility(false);
//    coreWaterLogical->SetVisAttributes(coreWaterVisAtt);
//
//    coreWaterSliceVisAtt = new G4VisAttributes(G4Colour(47.0/255.0,225.0/255.0,240.0/255.0));
//    coreWaterSliceVisAtt->SetVisibility(true);
//    coreWaterSliceLogical->SetVisAttributes(coreWaterSliceVisAtt);

    zircGridVisAtt = new G4VisAttributes(G4Colour(0.,0.,1.));
    zircGridVisAtt->SetVisibility(true);
    zircGridLogical->SetVisAttributes(zircGridVisAtt);

//    airGapsLatVisAtt = new G4VisAttributes(G4Colour(0.0/255.0,255.0/255.0,0.0/255.0));
//    airGapsLatVisAtt->SetVisibility(true);
//    airGapsLatLogical->SetVisAttributes(airGapsLatVisAtt);
////    airGapsLatHLogical->SetVisAttributes(airGapsLatVisAtt);
//    airGapsLatHRLogical->SetVisAttributes(airGapsLatVisAtt);
//    airGapsLatHR2Logical->SetVisAttributes(airGapsLatVisAtt);
//
//    fuelLatVisAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
//    fuelLatVisAtt->SetVisibility(true);
//    fuelLatLogical->SetVisAttributes(fuelLatVisAtt);
////    fuelLatHLogical->SetVisAttributes(fuelLatVisAtt);
//    fuelLatHRLogical->SetVisAttributes(fuelLatVisAtt);
//    fuelLatHR2Logical->SetVisAttributes(fuelLatVisAtt);

//bright red
    contRodZirVisAtt = new G4VisAttributes(G4Colour(1.,0.,0.));
    contRodZirVisAtt->SetVisibility(false);
    contRodZirLogical->SetVisAttributes(contRodZirVisAtt);

//bright yellow
    contRodAlumVisAtt = new G4VisAttributes(G4Colour(255.0/255.0,255.0/255.0,0.0/255.0));
    contRodAlumVisAtt->SetVisibility(false);
    contRodAlumLogical->SetVisAttributes(contRodAlumVisAtt);

//bright green
    contRodCadVisAtt = new G4VisAttributes(G4Colour(0.0/255.0,255.0/255.0,0.0/255.0));
    contRodCadVisAtt->SetVisibility(false);
    contRodCadLogical->SetVisAttributes(contRodCadVisAtt);

//dark red
    contRodCentVisAtt = new G4VisAttributes(G4Colour(119.0/255.0,0.0/255.0,0.0/255.0));
    contRodCentVisAtt->SetVisibility(false);
    contRodCentLogical->SetVisAttributes(contRodCentVisAtt);

    return worldPhysical;
}


// ConstructMaterials()
// Define and build the materials in the C6 lattice cell.
void DebugConstructor::ConstructMaterials()
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
