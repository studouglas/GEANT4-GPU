//
//  StorkSteppingAction.hh
//  G4-STORK_AT
//
//  Created by Andrew Tan on 2014-07-12.
//  Copyright (c) 2014 andrewtan. All rights reserved.
//
#ifndef _STORKSTEPPINGACTION_H_
#define _STORKSTEPPINGACTION_H_

#include "G4UserSteppingAction.hh"
#include "G4SteppingManager.hh"
#include "G4UserEventAction.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ProcessType.hh"
#include "G4HadronicProcessType.hh"
#include "G4TransportationManager.hh"
#include "StorkEventAction.hh"
#include "StorkHadProjectile.hh"
#include "StorkHadronFissionProcess.hh"
#include "StorkProcessManager.hh"
#include "G4FissionFragmentGenerator.hh"

class StorkSteppingAction: public G4UserSteppingAction
{
    public:
    StorkSteppingAction(StorkEventAction* eventAction);
    virtual ~StorkSteppingAction();
    
    //virtual void UserSteppingAction(const G4Step* theStep);
    
    protected:
    
    G4SteppingManager *SteppingMan;
    StorkEventAction *myEventAction;
    
    

    

};

#endif
