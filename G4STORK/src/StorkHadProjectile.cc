//
//  StorkHadProjectile.cc
//  G4-STORK_AT
//
//  Created by Andrew Tan on 2014-07-30.
//  Copyright (c) 2014 andrewtan. All rights reserved.
//

#include "StorkHadProjectile.hh"


StorkHadProjectile::StorkHadProjectile()
{
    theMat = 0;
    theDef = 0;
    theTime = 0.0;
    theBoundEnergy = 0.0;
}

StorkHadProjectile::StorkHadProjectile(const G4Track &aT)
{
    Initialise(aT);
}

StorkHadProjectile::StorkHadProjectile(const G4DynamicParticle &aT, const G4Material *Mat = NULL)
:
theOrgMom(aT.Get4Momentum()),
theDef(aT.GetDefinition())
{
    G4LorentzRotation toZ;
    toZ.rotateZ(-theOrgMom.phi());
    toZ.rotateY(-theOrgMom.theta());
    theMom = toZ*theOrgMom;
    toLabFrame = toZ.inverse();
    theTime = 0.0;
    theBoundEnergy = 0.0;
    theMat = Mat;
}

StorkHadProjectile::~StorkHadProjectile()
{}

void StorkHadProjectile::Initialise(const G4Track &aT)
{
    theMat = aT.GetMaterial();
    theOrgMom = aT.GetDynamicParticle()->Get4Momentum();
    theDef = aT.GetDefinition();

    G4LorentzRotation toZ;
    toZ.rotateZ(-theOrgMom.phi());
    toZ.rotateY(-theOrgMom.theta());
    theMom = toZ*theOrgMom;
    toLabFrame = toZ.inverse();

    //VI time of interaction starts from zero
    //   not global time of a track
    theTime = 0.0;
    theBoundEnergy = 0.0;
}

