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
//
//
 // Hadronic Process: Very Low Energy Neutron X-Sections
 // original by H.P. Wellisch, TRIUMF, 14-Feb-97
 // Has the Cross-section data for all materials.

#ifndef StorkNeutronHPDataConstructor_h
#define StorkNeutronHPDataConstructor_h 1
#include "globals.hh"
#include "G4Element.hh"
#include "G4NeutronHPElasticData.hh"
#include "G4NeutronHPInelasticData.hh"
#include "G4NeutronHPFissionData.hh"
#include "G4NeutronHPCaptureData.hh"
#include "StorkNeutronHPElementData.hh"

#include <vector>

class StorkNeutronHPDataConstructor
{
public:

  StorkNeutronHPDataConstructor(G4String dirName);

  ~StorkNeutronHPDataConstructor();

  inline G4PhysicsVector * MakePhysicsVector(G4Element * thE, G4NeutronHPFissionData * theP, G4String dirName)
  {
     if ( numEle <= (G4int)thE->GetIndex() ) addPhysicsVector(dirName);
     return DoPhysicsVector((*theData[thE->GetIndex()]).GetData(theP));
  }
  inline G4PhysicsVector * MakePhysicsVector(G4Element * thE, G4NeutronHPCaptureData * theP, G4String dirName)
  {
     if ( numEle <= (G4int)thE->GetIndex() ) addPhysicsVector(dirName);
     return DoPhysicsVector((*theData[thE->GetIndex()]).GetData(theP));
  }
  inline G4PhysicsVector * MakePhysicsVector(G4Element * thE, G4NeutronHPElasticData * theP, G4String dirName)
  {
     if ( numEle <= (G4int)thE->GetIndex() ) addPhysicsVector(dirName);
     return DoPhysicsVector((*theData[thE->GetIndex()]).GetData(theP));
  }
  inline G4PhysicsVector * MakePhysicsVector(G4Element * thE, G4NeutronHPInelasticData * theP, G4String dirName)
  {
//     G4cout << "entered StorkNeutronHPDataConstructor::MakePhysicsVector!!!"<<G4endl;
//     G4cout << "thE->GetIndex()="<<thE->GetIndex()<<G4endl;
     if ( numEle <= (G4int)thE->GetIndex() ) addPhysicsVector(dirName);
     return DoPhysicsVector((*theData[thE->GetIndex()]).GetData(theP));
  }

  G4PhysicsVector * DoPhysicsVector(G4NeutronHPVector * theVector);

  static StorkNeutronHPDataConstructor * Instance(G4String dirName);

private:

  std::vector< StorkNeutronHPElementData* > theData;
  G4int numEle;
  void addPhysicsVector(G4String dirName);
  G4String CSDirName;

};

#endif
