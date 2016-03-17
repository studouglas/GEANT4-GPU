//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file hadronic/Hadr03/Hadr03.cc
/// \brief Main program of the hadronic/Hadr03 example
//
//
// $Id: TestEm1.cc,v 1.16 2010-04-06 11:11:24 maire Exp $
// 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifdef G4MULTITHREADED
#include "G4MTRunManager.hh"
#else
#include "G4RunManager.hh"
#endif

#include "G4UImanager.hh"
#include "Randomize.hh"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"
#include "SteppingVerbose.hh"

#include "G4ParticleHPVector.hh"

#ifdef G4VIS_USE
#include "G4VisExecutive.hh"
#endif

#ifdef G4UI_USE
#include "G4UIExecutive.hh"
#endif

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void printTheData(G4ParticleHPVector vect){
	for (int i = 0; i < 20; i++){
		printf("Data at %d: X:%f Y:%f |", i, vect.GetPoint(i).GetX(), vect.GetPoint(i).GetY());
	}
	printf("\n");
}
void printTheIntegral(G4double* vect){
	for (int i = 0; i < 20; i++){
		printf("Data at %d: Value:%f |", i, vect[i]);
	}
	printf("\n");
}
int main(int argc, char** argv) {
	G4int I = 10;
	G4double dub = 10.2, sum;
	G4ParticleHPVector vect = G4ParticleHPVector();
	G4ParticleHPVector active = G4ParticleHPVector();
	G4ParticleHPVector passive = G4ParticleHPVector();
	G4ParticleHPDataPoint dat = G4ParticleHPDataPoint();
	G4InterpolationManager man = G4InterpolationManager();
	G4InterpolationScheme sch = G4InterpolationScheme();
	sch = LINLOG;+
	printf("------TESTING ALL METHODS IN G4ParticleHPVector-------\n");
	printf("Populating data...");
	G4double k = 1.02;
	for (int i = 0; i < 20; i++){
		vect.SetData(i, (double)30 + i, (double)30 + i);
		active.SetData(i, 30 + (i * 5), 30 + (i * 5));
		passive.SetData(i, 30 + (i * 5), 30 + (i * 5));
	}
	printf("Done\n");
	printf("Testing function G4double GetX(G4int)\n");
	printf("Input: %d\nOutput: %f\n\n", I, vect.GetX(I));

	printf("Testing function G4double GetY(G4int)\n");
	printf("Input: %d\nOutput: %f\n", I, vect.GetY(I));
	printf("Testing function G4double GetY(G4double)\n");
	printf("Input: %f\nOutput: %f\n\n", dub, vect.GetX(dub));

	printf("Testing function G4double GetXsec(G4double e, G4int min)\n");
	printf("Input: %f,%d\nOutput: %f\n\n", dub, I, vect.GetXsec(dub, I));

	printf("Testing function G4double GetMeanX()\n");
	printf("Input: N/A\nOutput: %f\n\n", vect.GetMeanX());

	printf("Testing function G4double SetLabel()\n");
	vect.SetLabel(dub);
	printf("Input: N/A\nOutput: N/A\n");
	printf("Value(s) modified after function call: label:%f\n\n", vect.GetLabel());

	printf("Testing function G4double GetLabel()\n");
	printf("Input: N/A\nOutput: %f\n\n", vect.GetLabel());

	printf("Testing function void SetData(G4int i, G4double x, G4double y)\n");
	printf("Input: %d,%f,%f\nOutput: N/A\n", 0, dub, dub + 1.0);
	vect.SetData(0, dub, dub + 1);
	printf("Value(s) modified after function call:%f\n\n", vect.GetPoint(0).GetX());

	printf("Testing function G4double  SampleLin()\n");
	printf("Input: N/A\nOutput: %f\n\n", vect.SampleLin());

	printf("Testing function G4double Debug()\n");
	printf("Input: N/A\nOutput:\n");
	printTheIntegral(vect.Debug());
	printf("\n");

	printf("Testing function G4int GetVectorLength()\n");
	printf("Input: N/A\nOutput:%d\n\n",vect.GetVectorLength());
	//printf("Testing function void SetScheme(G4int aPoint, const G4InterpolationScheme & aScheme)\n");
	//vect.SetScheme(1, sch);
	//printf("Input: %d\nOutput: %s\n", I, vect.GetScheme(1));

	//	printf("Testing function G4InterpolationScheme GetScheme(G4int anIndex)\n");
	//	printf("Input: %d\nOutput: %s\n", I, vect.GetScheme(I));

	/*printf("Testing function void Integrate()\n");
	printf("Input: N/A\nOutput: N/A\n\n");
	printf("Value(s) modified before function call: totalIntegral:%f", vect.getTotalIntegral());
	printf("Value(s) modified after function call: totalIntegral:%f", vect.getTotalIntegral());*/
	
	printf("Testing function G4double GetIntegral()\n");
	printf("Input: N/A\nOutput: N/A\n");
	printf("Value(s) modified after function call: totalIntegral:%f\n\n", vect.GetIntegral());

	printf("Testing function void Times(G4double factor)\n");
	printf("Input: %f\nOutput: N/A\n", 2.0);
	printf("Value(s) modified before function call: theData:\n");
	printTheData(vect);
	printf("Value(s) modified before function call: theIntegral:\n");
	printTheIntegral(vect.Debug());
	vect.Times(2.0);
	printf("Value(s) modified after function call: theData:\n");
	//printTheData(vect);
	printf("Value(s) modified after function call: theIntegral:\n");
	//printTheIntegral(vect.Debug());

	printf("Testing function void IntegrateAndNormalise()\n");
	printf("Input: N/A\nOutput: N/A\n");
	printf("Value(s) modified before function call: totalIntegral:\n");
	printf("Not yet implemented!\n");
	vect.IntegrateAndNormalise();
	printf("Value(s) modified after function call: totalIntegral:\n");
	printf("Not yet implemented!\n");



	printf("Testing function void Merge(G4ParticleHPVector * active, G4ParticleHPVector * passive)");
	printf("Input: active:\n");
	printTheData(active);
	printf("Passive:\n");
	printTheData(passive); 
	printf("Value(s) modified before function call: theData:\n");
	//printTheData(vect);
	//vect.Merge(active,passive);
	printf("Value(s) modified after function call: theData:\n");
	//printTheData(vect);
	// Construct the default run manager
	G4Random::setTheEngine(new CLHEP::RanecuEngine);
#ifdef G4MULTITHREADED
	G4MTRunManager* runManager = new G4MTRunManager;
	runManager->SetNumberOfThreads(G4Threading::G4GetNumberOfCores());
#else
	//my Verbose output class
	G4VSteppingVerbose::SetInstance(new SteppingVerbose);
	G4RunManager* runManager = new G4RunManager;
#endif

	// set mandatory initialization classes
	DetectorConstruction* det = new DetectorConstruction;
	runManager->SetUserInitialization(det);

	PhysicsList* phys = new PhysicsList;
	runManager->SetUserInitialization(phys);

	runManager->SetUserInitialization(new ActionInitialization(det));

	// get the pointer to the User Interface manager 
	G4UImanager* UI = G4UImanager::GetUIpointer();

	if (argc != 1)   // batch mode  
	{
		G4String command = "/control/execute ";
		G4String fileName = argv[1];
		UI->ApplyCommand(command + fileName);
	}

	else           //define visualization and UI terminal for interactive mode
	{
#ifdef G4VIS_USE
		G4VisManager* visManager = new G4VisExecutive;
		visManager->Initialize();
#endif    

#ifdef G4UI_USE
		G4UIExecutive * ui = new G4UIExecutive(argc, argv);
		ui->SessionStart();
		delete ui;
#endif

#ifdef G4VIS_USE
		delete visManager;
#endif     
	}

	// job termination 
	//
	delete runManager;

	return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......