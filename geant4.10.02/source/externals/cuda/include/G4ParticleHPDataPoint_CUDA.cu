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
#ifndef G4ParticleHPDataPoint_CUDA_h
#define G4ParticleHPDataPoint_CUDA_h 1

#include <cuda.h>
#include <cuda_runtime.h>

class G4ParticleHPDataPoint_CUDA
{
  public:  
  G4ParticleHPDataPoint_CUDA() {
    energy = 0; 
    xSec = 0;
  }
  G4ParticleHPDataPoint_CUDA(double e, double x){ 
    energy = e; 
    xSec = x;
  }
  
  __host__ __device__ void operator= (const G4ParticleHPDataPoint_CUDA & aSet) {
    if(&aSet!=this) {
      energy = aSet.GetEnergy();
      xSec   = aSet.GetXsection();
    }
  }

  __host__ __device__ ~G4ParticleHPDataPoint_CUDA() { }
  
  __host__ __device__ double GetEnergy() const {
    return energy;
  }
  __host__ __device__ double GetXsection() const { 
    return xSec;
  }
  
  __host__ __device__ void SetEnergy(double e) { 
    energy = e;
  }
  __host__ __device__ void SetXsection(double x) {
    xSec = x;
  }
  
  __host__ __device__ double GetX() const { 
    return energy;
  }
  __host__ __device__ double GetY() const {
    return xSec;
  }
  
  __host__ __device__ void SetX(double e) {
    energy = e;
  }
  __host__ __device__ void SetY(double x) {
    xSec = x;
  }
  
  __host__ __device__ void SetData(double e, double x) {
    energy = e; xSec = x;
  }
  
  private:
  double energy;
  double xSec;
};

#endif
