/*
StorkProcessManager.hh

Created by:		Liam Russell
Date:			24-05-2012
Modified:		11-03-2013

Used to manage hadronic processes and get the number of interaction lengths
left from each process.  This class is made as a singleton so that it can
be called from anywhere in the code using a static member function.

*/

#ifndef STORKPROCESSMANAGER_H
#define STORKPROCESSMANAGER_H

// Include header files
#include "G4ProcessManager.hh"
#include "StorkHadronFissionProcess.hh"
#include "StorkHadronCaptureProcess.hh"
#include "StorkNeutronInelasticProcess.hh"
#include "StorkHadronElasticProcess.hh"
#include "globals.hh"

class StorkHadronElasticProcess;


class StorkProcessManager
{
	public:
		// Static member functions

		static StorkProcessManager* GetStorkProcessManagerPtr();
		static StorkProcessManager* GetStorkProcessManagerPtrIfExists();


	public:
		// Public member functions

		G4double* GetNumberOfInteractionLengthsLeft(const G4double prevStepSize) const;
		G4String* GetProcessOrder() const { return procOrder; }
		G4int GetProcessIndex(const G4String procName) const;
		G4VProcess* GetProcess(const G4String procName) const;

		~StorkProcessManager();


	protected:
		// Protected (hidden) constructor

		StorkProcessManager();


	private:
		// Private member variables

		// Hadronic process pointers
        StorkHadronElasticProcess *theElasticProcess;
        StorkNeutronInelasticProcess *theInelasticProcess;
        StorkHadronFissionProcess *theFissionProcess;
        StorkHadronCaptureProcess *theCaptureProcess;

        // Number of interaction lengths left array
        G4double *n_lambda;
        G4String *procOrder;
        G4int numProcesses;


	private:
		// Static member variables

		static StorkProcessManager *theManager;
};

#endif // STORKPROCESSMANAGER_H
