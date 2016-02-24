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
/// \file parallel/ThreadsafeScorers/include/TSActionInitialization.hh
/// \brief Definition of the TSActionInitialization class
//
//
// $Id: TSActionInitialization.hh 93110 2015-11-05 08:37:42Z jmadsen $
//
//
/// Standard ActionInitialization class creating a RunAction instance for the
///     master thread and RunAction and PrimaryGeneratorAction instances for
///     the worker threads
//
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......




#ifndef tsactioninitialization_hh_
#define tsactioninitialization_hh_

#include "globals.hh"
#include "G4VUserActionInitialization.hh"

#include "G4AutoLock.hh"
#include "G4Threading.hh"

class TSActionInitialization : public G4VUserActionInitialization
{
public:
    // Constructor and Destructors
    TSActionInitialization();
    virtual ~TSActionInitialization();

    static TSActionInitialization* Instance();

public:
    virtual void BuildForMaster() const;
    virtual void Build() const;

private:
    // Private functions
    static TSActionInitialization* fgInstance;

};

#endif
