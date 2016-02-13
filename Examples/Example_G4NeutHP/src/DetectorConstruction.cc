#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"
#include "G4RunManager.hh"

#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

DetectorConstruction::DetectorConstruction()
	:G4VUserDetectorConstruction(), fPBox(0), fLBox(0), fMaterial(0),
	 fDetectorMessenger(0){ //initialization list for DetectorConstruction constructor
	fBoxSide = 1*m;
	DefineMaterials();
	SetMaterial("Water_ts");
	fDetectorMessenger = new DetectorMessenger(this);
}

DetectorConstruction::~DetectorConstruction(){
	delete fDetectorMessenger;
}

void DetectorConstruction::DefineMaterials(){

}

G4Material* DetectorConstruction::MaterialWithSingleIsotope(G4String name, G4String symbol, G4double density, G4int Z, G4int A){ //define a material from an isoptope

}

G4VPhysicalVolume* DetectorConstruction::ConstructVolumes(){

}

void DetectorConstruction::SetMaterial(G4String materialChoice){

}

void DetectorConstruction::SetSize(G4double value){

}



