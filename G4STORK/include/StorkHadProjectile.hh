//
//  StorkHadProjectile.h
//  G4-STORK_AT
//
//  Created by Andrew Tan on 2014-07-30.
//  Copyright (c) 2014 andrewtan. All rights reserved.
//

#ifndef __G4_STORK_AT__StorkHadProjectile__
#define __G4_STORK_AT__StorkHadProjectile__

#include <iostream>
#include "globals.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4LorentzVector.hh"
#include "G4LorentzVector.hh"
#include "G4LorentzRotation.hh"
#include "G4HadProjectile.hh"
#include "G4Track.hh"
#include "G4DynamicParticle.hh"

class G4Track;
class G4DynamicParticle;

class StorkHadProjectile
{
public:
    StorkHadProjectile();
    StorkHadProjectile(const G4Track &aT);
    StorkHadProjectile(const G4DynamicParticle &aT, const G4Material *theMat);
    ~StorkHadProjectile();

    void Initialise(const G4Track &aT);

    inline const G4Material * GetMaterial() const;
    inline const G4ParticleDefinition * GetDefinition() const;
    inline const G4LorentzVector & Get4Momentum() const;
    inline G4LorentzRotation & GetTrafoToLab();
    inline G4double GetKineticEnergy() const;
    inline G4double GetTotalEnergy() const;
    inline G4double GetTotalMomentum() const;
    inline G4double GetGlobalTime() const;
    inline G4double GetBoundEnergy() const;
    inline void SetGlobalTime(G4double t);
    inline void SetBoundEnergy(G4double e);

private:

    // hide assignment operator as private
    StorkHadProjectile& operator=(const StorkHadProjectile &right);
    StorkHadProjectile(const StorkHadProjectile& );

    const G4Material * theMat;
    G4LorentzVector theOrgMom;
    G4LorentzVector theMom;
    const G4ParticleDefinition * theDef;
    G4LorentzRotation toLabFrame;
    G4double theTime;
    G4double theBoundEnergy;
};

const G4Material * StorkHadProjectile::GetMaterial() const
{
    return theMat;
}

const G4ParticleDefinition * StorkHadProjectile::GetDefinition() const
{
    return theDef;
}

inline const G4LorentzVector& StorkHadProjectile::Get4Momentum() const
{
    return theMom;
}

inline G4LorentzRotation& StorkHadProjectile::GetTrafoToLab()
{
    return toLabFrame;
}

G4double StorkHadProjectile::GetTotalEnergy() const
{
    return Get4Momentum().e();
}

G4double StorkHadProjectile::GetTotalMomentum() const
{
    return Get4Momentum().vect().mag();
}

G4double StorkHadProjectile::GetKineticEnergy() const
{
    G4double ekin = GetTotalEnergy() - GetDefinition()->GetPDGMass();
    if(ekin < 0.0) { ekin = 0.0; }
    return ekin;
}

inline G4double StorkHadProjectile::GetGlobalTime() const
{
    return theTime;
}

inline G4double StorkHadProjectile::GetBoundEnergy() const
{
    return theBoundEnergy;
}

inline void StorkHadProjectile::SetGlobalTime(G4double t)
{
    theTime = t;
}

inline void StorkHadProjectile::SetBoundEnergy(G4double e)
{
    theBoundEnergy = e;
}

#endif
