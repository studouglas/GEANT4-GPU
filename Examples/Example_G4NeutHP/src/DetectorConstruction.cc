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
	//SetMaterial("Uranium");
	fDetectorMessenger = new DetectorMessenger(this);
}

DetectorConstruction::~DetectorConstruction(){
	delete fDetectorMessenger;
}

void DetectorConstruction::DefineMaterials(){
	//define uranium element
	//create uranium material using said element (URN->AddElement(uranium)
	//
}

G4Material* DetectorConstruction::MaterialWithSingleIsotope(G4String name, G4String symbol, G4double density, G4int Z, G4int A){ //define a material from an isoptope

}

G4VPhysicalVolume* DetectorConstruction::ConstructVolumes(){
	//cleanup old geometry
	G4GeometryManager::GetInstance()->OpenGeometry();
	G4PhysicalVolumeStore::GetInstance()->Clean();
	G4LogicalVolumeStore::GetInstance()->Clean();
	G4SolidStore::GetInstance()->Clean();
}

void DetectorConstruction::SetMaterial(G4String materialChoice){
	// search the material by its name
  G4Material* pttoMaterial =
     G4NistManager::Instance()->FindOrBuildMaterial(materialChoice);   
  
  if (pttoMaterial) { 
    if(fMaterial != pttoMaterial) {
      fMaterial = pttoMaterial;
      if(fLBox) { fLBox->SetMaterial(pttoMaterial); }
      G4RunManager::GetRunManager()->PhysicsHasBeenModified();
    }
  } else {
    G4cout << "\n--> warning from DetectorConstruction::SetMaterial : "
           << materialChoice << " not found" << G4endl;
  } 
}

void DetectorConstruction::SetSize(G4double value){

}



