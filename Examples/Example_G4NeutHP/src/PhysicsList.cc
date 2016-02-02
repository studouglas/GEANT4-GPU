#include "PhysicsList.hh"

#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "NeutronHPphysics.hh"

#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BosonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

PhysicsList::PhysicsList():G4VModularPhysicsList() {
	SetVerboseLevel(1);
  
	//add new units
	new G4UnitDefinition( "millielectronVolt", "meV", "Energy", 1.e-3*eV);   
	new G4UnitDefinition( "mm2/g",  "mm2/g", "Surface/Mass", mm2/g);
	new G4UnitDefinition( "um2/mg", "um2/mg","Surface/Mass", um*um/mg);  
    
	//Neutron Physics
	RegisterPhysics(new NeutronHPphysics("neutronHP"));  
}

PhysicsList::~PhysicsList() { } //class destructor

void PhysicsList::ConstructParticle() {
	G4BosonConstructor  pBosonConstructor;
  pBosonConstructor.ConstructParticle();

  G4LeptonConstructor pLeptonConstructor;
  pLeptonConstructor.ConstructParticle();

  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4IonConstructor pIonConstructor;
  pIonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();
}

void PhysicsList::SetCuts() {
	SetCutValue(0*mm, "proton");
}
