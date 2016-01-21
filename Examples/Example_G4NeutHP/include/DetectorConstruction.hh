#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4LogicalVolume;
class G4Material;
class DetectorMessenger;

class DetectorConstruction : public G4VUserDetectorConstruction
{
	public:
  
		DetectorConstruction(); //constructor
		~DetectorConstruction(); //destructor
  
		virtual G4VPhysicalVolume* Construct();

		G4Material* 
		MaterialWithSingleIsotope(G4String, G4String, G4double, G4int, G4int);
         
		void SetSize     (G4double);              
		void SetMaterial (G4String);            

  	const G4VPhysicalVolume* GetWorld(){return fPBox;};           
                    
		G4double           GetSize()       {return fBoxSize;};      
		G4Material*        GetMaterial()   {return fMaterial;};
     
		void PrintParameters();
                       
	private:
  
		G4VPhysicalVolume* fPBox;
		G4LogicalVolume*   fLBox;
     
		G4double           fBoxSize;
		G4Material*        fMaterial;     
     
		DetectorMessenger* fDetectorMessenger;
    
		void               DefineMaterials();
		G4VPhysicalVolume* ConstructVolumes();     
};

#endif
