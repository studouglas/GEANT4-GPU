#ifndef ActionInitialization_h
#define ActionInitialization_h 1

#include "G4VUserActionInitialization.hh"

class DetectorConstruction;
class G4VSteppingVerbose;

class ActionInitialization : public G4VUserActionInitialization{
	public:

    ActionInitialization(DetectorConstruction* detector);
		virtual ~ActionInitialization();

		virtual void BuildForMaster() const;
		virtual void Build() const;
    
		virtual G4VSteppingVerbose* InitializeSteppingVerbose() const;
   
	private:

		DetectorConstruction* fDetector;
};

#endif

