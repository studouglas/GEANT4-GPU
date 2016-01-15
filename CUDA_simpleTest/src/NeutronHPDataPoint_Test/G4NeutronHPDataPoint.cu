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
#ifndef G4NeutronHPDataPoint_h
#define G4NeutronHPDataPoint_h 1

#include "globals.hh"

class G4NeutronHPDataPoint
{
  public:
  
  G4NeutronHPDataPoint(){energy = 0; xSec = 0;}
  G4NeutronHPDataPoint(G4double e, G4double x){ energy = e; xSec = x;}
  
  __host__ __device__ void operator= (const G4NeutronHPDataPoint & aSet)
  {
    if(&aSet!=this)
    {
      energy = aSet.GetEnergy();
      xSec   = aSet.GetXsection();
    }
  }

  __host__ __device__ ~G4NeutronHPDataPoint(){}
  
  __host__ __device__ G4double GetEnergy() const   {return energy;}
  __host__ __device__ G4double GetXsection() const {return xSec;}
  
  __host__ __device__ void SetEnergy(G4double e)  {energy = e;}
  __host__ __device__ void SetXsection(G4double x){xSec = x;}
  
  __host__ __device__ G4double GetX() const {return energy;}
  __host__ __device__ G4double GetY() const {return xSec;}
  
  __host__ __device__ void SetX(G4double e)  {energy = e;}
  __host__ __device__ void SetY(G4double x)  {xSec = x;}
  
  __host__ __device__ void SetData(G4double e, G4double x) {energy = e; xSec = x;}
  
  private:
  
  G4double energy;
  G4double xSec;
};

#endif
