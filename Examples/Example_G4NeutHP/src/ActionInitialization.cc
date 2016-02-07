#include "ActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "TrackingAction.hh"
#include "SteppingAction.hh"
#include "SteppingVerbose.hh"
#include "StackingAction.hh"

ActionInitialization::ActionInitialization(DetectoRConstruction* detector):	G4VUserActionInitialization(),
						fDetector(detector) { //initialization list

}

ActionInitialization::~ActionInitialization() {

}

void ActionInitialization::BuildForMaster() const {
	RunAction* runAction = new RunAction(fDetector, 0);
	SetUserAction(runAction);
}

void ActionInitialization::Build() const {
	PrimaryGeneratorAction* primary = new PrimaryGeneratorAction();
	SetUserAction(primary);
    
	//RunAction* runAction = new RunAction(fDetector, primary );
	//SetUserAction(runAction);
  
	//TrackingAction* trackingAction = new TrackingAction();
	//SetUserAction(trackingAction);
  
	//SteppingAction* steppingAction = new SteppingAction(trackingAction);
	//SetUserAction(steppingAction);
  
	//StackingAction* stackingAction = new StackingAction();
	//SetUserAction(stackingAction); 
}

G4VSteppingVerbose* ActionInitialization::InitializeSteppingVerbose() const {
	return new SteppingVerbose();
}


