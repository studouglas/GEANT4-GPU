#include "Run.hh"
#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "HistoManager.hh"

#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

Run::Run(DetectorConstruction* detector)
		:	G4Run(), fDetector(detector), fParticle(0), fEkin(0.),
			fNbStep1(0), fNbStep2(0), fTrackLen1(0.), fTrackLen2(0.),
			fTime1(0.), fTime2(0.){ //initialization list

}

Run::~Run() {} //class destructor

void Run::SetPrimary(G4ParticleDefinition* particle, G4double energy) {
	fParticle = particle;
	fEkin = energy;
}

void Run::CountProcesses(const G4VProcess* process) {

}

void Run::ParticleCount(G4String name, G4double Ekin) {

}

void Run::SumTrackLength(G4int nstep1, G4int nstep2,
												 G4double trackl1, G4double trackl2,
												 G4double time1, G4double time2) {
	
}

void Run::Merge(const G4Run* run) {

}

void Run::EndOfRun)( {

}
