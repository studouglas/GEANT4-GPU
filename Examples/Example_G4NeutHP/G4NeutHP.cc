#ifdef G4MULTITHREADED
	#include "G4MTRunManager.hh"
#else
	#include "G4RunManager.hh"
#endif
#include "G4UImanager.hh"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"
#include "SteppingVerbose.hh"

int main() {
	#ifdef G4MULTITHREADED
		G4MTRunManager* runManager = new G4MTRunManager;
		runMan->SetNumberOfThreads(G4Threading::G4GetNumberOfCores());
	#else
		//G4VSteppingVerbose::SetInstance(new SteppingVerbose);
		G4RunManager* runManager = new G4RunManager;
	#endif
	
	//initialize and set geometry and detector
	DetectorConstruction* detector = new DetectorConstruction;
	runManager->SetUserInitialization(detector);

	//initialize and set physics lists
	PhysicsList* physics = new PhysicsList;
	runManager->SetUserInitialization(physics);

	//set action initial action/primary generator action
	runManager->SetUserInitialization(new ActionInitialization(detector));
	
	runManager->Initialize(); //this line initializes the G4 kernel (and calculates cross sections)
	
	G4UImanager* UI = G4UImanager::GetUIpointer();
	UI->ApplyCommand("/run/verbose 1");
	UI->ApplyCommand("event/verbose 1");
	UI->ApplyCommand("/tracking/verbose 1");

	int events = 3; //number of events
	runManager->BeamOn(events); //invokes a run with n number of events

	
	//terminate the run
	delete runManager;
	return 0;
}
