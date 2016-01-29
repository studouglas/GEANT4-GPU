#include "RunAction.hh"
#include "Run.hh"
#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "HistoManager.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

#include "Randomize.hh"
#include <iomanip>

RunAction::RunAction(DetectorConstruction* detector, PrimaryGeneratorAction* primaryGen):	G4UserRunAction(),
																			fDetector(detector),
																			fPrimary(primaryGen),
																			fRun(0), fHistoManager(0) {

}

RunAction::~RunAction() {
	delete fHistoManager;
}

G4Run* RunAction::GenerateRun() {

}

void RunAction::BeginOfRunAction(const G4Run*) {

}

void RunAction::EndOfRunACtion(const G4Run*) {

}
