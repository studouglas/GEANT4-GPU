/*
StorkWorld.cc

Created by:		Liam Russell
Date:			17-02-2011
Modified:		11-03-2013

Source code for the StorkWorld class.

*/


// Include header file

#include "StorkWorld.hh"

// Include headers here to avoid circular reference with StorkVWorldConstructor
#include "BareSphereConstructor.hh"
#include "C6LatticeConstructor.hh"
#include "InfiniteUniformLatticeConstructor.hh"
#include "ZED2Constructor.hh"
#include "SLOWPOKEConstructor.hh"
#include "SCWRConstructor.hh"
#include "SCWRDopplerConstructor.hh"
#include "SCWRJasonConstructor.hh"
#include "Q_ZED2Constructor.hh"
//#include "DebugConstructor.hh"
#include "TestConstructor.hh"


// Overloaded Constructor
StorkWorld::StorkWorld(const StorkParseInput* infile)
: worldPhysical(0), theWorld(0)
{
    // Add the three basic worlds to the world map
    AddWorld("C6Lattice", new C6LatticeConstructor());
    AddWorld("Sphere", new BareSphereConstructor());
    AddWorld("Cube", new InfiniteUniformLatticeConstructor());
    AddWorld("ZED2", new ZED2Constructor());
    AddWorld("SLOWPOKE", new SLOWPOKEConstructor());
    AddWorld("SCWR", new SCWRConstructor());
    AddWorld("SCWRJason", new SCWRJasonConstructor());
    AddWorld("SCWRDoppler", new SCWRDopplerConstructor());
    AddWorld("Q_ZED2", new Q_ZED2Constructor());
//    AddWorld("Debug", new DebugConstructor());
    AddWorld("Test", new TestConstructor());

    // Copy user inputs
    inFile = infile;
    worldName = infile->GetWorld();
}

StorkWorld::StorkWorld()
: worldPhysical(0), theWorld(0)
{
    // Add the three basic worlds to the world map
    AddWorld("C6Lattice", new C6LatticeConstructor());
    AddWorld("Sphere", new BareSphereConstructor());
    AddWorld("ZED2", new ZED2Constructor());
    AddWorld("Cube", new InfiniteUniformLatticeConstructor());
    AddWorld("SLOWPOKE", new SLOWPOKEConstructor());
    AddWorld("SCWR", new SCWRConstructor());
    AddWorld("SCWRJason", new SCWRJasonConstructor());
    AddWorld("SCWRDoppler", new SCWRDopplerConstructor());
    AddWorld("Q_ZED2", new Q_ZED2Constructor());
//    AddWorld("Debug", new DebugConstructor());
    AddWorld("Test", new TestConstructor());
}

// Destructor
StorkWorld::~StorkWorld()
{
//    // Delete all materials, elements and isotopes
//    delete theWorld;

    // Delete the worlds
    StorkWorldMap::iterator itr = availableWorlds.begin();

    while(itr != availableWorlds.end())
    {
        delete itr->second;
        itr++;
    }

    availableWorlds.clear();
}


void StorkWorld::InitializeWorldData(const StorkParseInput* infile)
{
    // Copy user inputs
    inFile = infile;
    worldName = infile->GetWorld();
}

void StorkWorld::InitializeWorldData(G4String worlnam)
{
    // Copy user inputs

    worldName = worlnam;
}

// Construct()
// Use one of the constructors to build the world.
G4VPhysicalVolume* StorkWorld::Construct()
{
    // Set the world constructor
    theWorld = availableWorlds[worldName];

    if(!theWorld)
    {
        G4cerr << "***ERROR: " << worldName
               << " is not one of the available worlds." << G4endl;

        return NULL;
    }

    // Build the world
    worldPhysical = theWorld->ConstructNewWorld(inFile);

    return worldPhysical;
}


// AddWorld()
// Add world to the available worlds map
void StorkWorld::AddWorld(G4String name, StorkVWorldConstructor *aNewWorld)
{
    availableWorlds[name] = aNewWorld;
}


// UpdateWorld()
// Updates the simulation world using a vector of proposed changes
G4VPhysicalVolume* StorkWorld::UpdateWorld(StorkMatPropChangeVector theChanges)
{
	return theWorld->UpdateWorld(theChanges);
}

// HasMatChanged()
// Returns the value of matChanged in the world class
G4bool StorkWorld::HasMatChanged()
{
    return theWorld->HasMatChanged();
}

// HasPhysChanged()
// Returns the calue of physChanged in the world calss
G4bool StorkWorld::HasPhysChanged()
{
    return theWorld->HasPhysChanged();
}

// SetPhysChanged()
// Sets the value of physChanged in the world class
void StorkWorld::SetPhysChanged(G4bool value)
{
    theWorld->SetPhysChanged(value);
}


// EvaluateHeatCapacities()
// Output material temperature header to specified file in Input file
void StorkWorld::SaveMaterialTemperatureHeader(G4String fname)
{
    theWorld->SaveMaterialTemperatureHeader(fname);
}

// EvaluateHeatCapacities()
// Output materials temperature to specified file in Input file
void StorkWorld::SaveMaterialTemperatures(G4String fname, G4int runNumber)
{
    theWorld->SaveMaterialTemperatures(fname, runNumber);
}

// GetWorldBoxDimensions()
// Get dimensions of smallest box enclosing the simulation world
G4ThreeVector StorkWorld::GetWorldBoxDimensions()
{
	return theWorld->GetEncWorldDim();
}


// GetWorldDimensions()
// Get the dimensions of the world
G4ThreeVector StorkWorld::GetWorldDimensions()
{
	return theWorld->GetReactorDim();
}


// GetWorldProperty()
// Get the current value of a material-property in the simulation world
G4double StorkWorld::GetWorldProperty(MatPropPair matProp)
{
	return theWorld->GetWorldProperty(matProp);
}


// GetLogicalVolume()
// Get the world logical volume
G4LogicalVolume* StorkWorld::GetWorldLogicalVolume()
{
	return theWorld->GetWorldLogical();
}

// GetMaterialMap()
// Get the world material map
StorkMaterialMap* StorkWorld::GetMaterialMap(void)
{
    return theWorld->GetMaterialMap();
}

// DumpGeometricalTree()
// Public - prints entire geometry
void StorkWorld::DumpGeometricalTree()
{
	DumpGeometricalTree(worldPhysical);
}


// DumpGeometricalTree()
// Private - Print the geometrical tree of the world
void StorkWorld::DumpGeometricalTree(G4VPhysicalVolume *vol,
                                                  G4int depth)
{
    for(G4int i=0; i<depth; i++)
    {
        G4cout << "  ";
    }

    G4cout << vol->GetName() << "[" << vol->GetCopyNo() << "] "
           << vol->GetLogicalVolume()->GetName() << " "
           << vol->GetLogicalVolume()->GetNoDaughters() << " "
           << vol->GetLogicalVolume()->GetMaterial()->GetName();

    if(vol->GetLogicalVolume()->GetSensitiveDetector())
    {
        G4cout << " " << vol->GetLogicalVolume()->GetSensitiveDetector()
                            ->GetFullPathName();
    }

    G4cout << G4endl;
    for(int i=0;i<vol->GetLogicalVolume()->GetNoDaughters();i++)
    {
        DumpGeometricalTree(vol->GetLogicalVolume()->GetDaughter(i),depth+1);
    }

    G4cout << G4endl;
}

void StorkWorld::SetMatChanged(G4bool value)
{
    theWorld->SetMatChanged(value);
}

G4ThreeVector StorkWorld::GetFuelDimensions()
{
    return theWorld->GetFuelDimensions();
}

G4double* StorkWorld::GetFuelTemperatures()
{
    return theWorld->GetFuelTemperatures();
}

G4double* StorkWorld::GetFuelDensities()
{
    return theWorld->GetFuelDensities();
}

G4double* StorkWorld::GetFuelRadii()
{
    return theWorld->GetFuelRadii();
}
