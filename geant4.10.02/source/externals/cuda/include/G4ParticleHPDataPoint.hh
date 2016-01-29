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
// P. Arce, June-2014 Conversion neutron_hp to particle_hp
//
#ifndef G4ParticleHPDataPoint_h
#define G4ParticleHPDataPoint_h 1

//#include "globals.hh"

class G4ParticleHPDataPoint
{
  public:
  
  G4ParticleHPDataPoint() {energy = 0; xSec = 0;}
  G4ParticleHPDataPoint(double e, double x){ energy = e; xSec = x;}
  
  void operator= (const G4ParticleHPDataPoint & aSet)
  {
    if(&aSet!=this)
    {
      energy = aSet.GetEnergy();
      xSec   = aSet.GetXsection();
    }
  }

//  ~G4ParticleHPDataPoint(){}
  
  inline double GetEnergy() const   {return energy;}
  inline double GetXsection() const {return xSec;}
  
  inline void SetEnergy(double e)  {energy = e;}
  inline void SetXsection(double x){xSec = x;}
  
  inline double GetX() const {return energy;}
  inline double GetY() const {return xSec;}
  
  inline void SetX(double e)  {energy = e;}
  inline void SetY(double x)  {xSec = x;}
  
  inline void SetData(double e, double x) {energy = e; xSec = x;}
  
  public:
  
  double energy;
  double xSec;
};

#endif
