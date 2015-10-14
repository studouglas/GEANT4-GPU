//
//  StorkHadronicInteraction.h
//  G4-STORK_AT
//
//  Created by Andrew Tan on 2014-07-30.
//  Copyright (c) 2014 andrewtan. All rights reserved.
//

#ifndef __G4_STORK_AT__StorkHadronicInteraction__
#define __G4_STORK_AT__StorkHadronicInteraction__

#include <iostream>
#include "G4HadronicInteraction.hh"
#include "G4HadFinalState.hh"
#include "G4Material.hh"
#include "G4Nucleus.hh"
#include "G4Track.hh"
#include "G4HadProjectile.hh"
#include "G4ReactionDynamics.hh"

class StorkHadronicInteraction: public G4HadronicInteraction
{
public: // With description
    
    StorkHadronicInteraction(const G4String& modelName = "HadronicModel");
    
    virtual ~StorkHadronicInteraction();
    
    virtual G4HadFinalState *ApplyYourself(const G4HadProjectile &aTrack,
                                           G4Nucleus & targetNucleus ) = 0;
    // The interface to implement for final state production code.
    
    virtual G4double SampleInvariantT(const G4ParticleDefinition* p,
                                      G4double plab,
                                      G4int Z, G4int A);
    // The interface to implement sampling of scattering or change exchange
    
    virtual G4bool IsApplicable(const G4HadProjectile &/*aTrack*/,
                                G4Nucleus & /*targetNucleus*/)
    { return true;}
    
    inline G4double GetMinEnergy() const
    { return theMinEnergy; }
    
    G4double GetMinEnergy( const G4Material *aMaterial,
                          const G4Element *anElement ) const;
    
    inline void SetMinEnergy( G4double anEnergy )
    { theMinEnergy = anEnergy; }
    
    void SetMinEnergy( G4double anEnergy, const G4Element *anElement );
    
    void SetMinEnergy( G4double anEnergy, const G4Material *aMaterial );
    
    inline G4double GetMaxEnergy() const
    { return theMaxEnergy; }
    
    G4double GetMaxEnergy( const G4Material *aMaterial,
                          const G4Element *anElement ) const;
    
    inline void SetMaxEnergy( const G4double anEnergy )
    { theMaxEnergy = anEnergy; }
    
    void SetMaxEnergy( G4double anEnergy, const G4Element *anElement );
    
    void SetMaxEnergy( G4double anEnergy, const G4Material *aMaterial );
    
    inline const StorkHadronicInteraction* GetMyPointer() const
    { return this; }
    
    virtual G4int GetVerboseLevel() const
    { return verboseLevel; }
    
    virtual void SetVerboseLevel( G4int value )
    { verboseLevel = value; }
    
    inline const G4String& GetModelName() const
    { return theModelName; }
    
    void DeActivateFor(const G4Material* aMaterial);
    
    inline void ActivateFor( const G4Material *aMaterial )
    {
        Block();
        SetMaxEnergy(GetMaxEnergy(), aMaterial);
        SetMinEnergy(GetMinEnergy(), aMaterial);
    }
    
    void DeActivateFor( const G4Element *anElement );
    inline void ActivateFor( const G4Element *anElement )
    {
        Block();
        SetMaxEnergy(GetMaxEnergy(), anElement);
        SetMinEnergy(GetMinEnergy(), anElement);
    }
    
    G4bool IsBlocked( const G4Material *aMaterial ) const;
    G4bool IsBlocked( const G4Element *anElement) const;
    
    inline void SetRecoilEnergyThreshold(G4double val)
    { recoilEnergyThreshold = val; }
    
    G4double GetRecoilEnergyThreshold() const
    { return recoilEnergyThreshold;}
    
    inline G4bool operator==(const StorkHadronicInteraction &right ) const
    { return ( this == (G4HadronicInteraction *) &right ); }
    
    inline G4bool operator!=(const StorkHadronicInteraction &right ) const
    { return ( this != (G4HadronicInteraction *) &right ); }
    
    virtual const std::pair<G4double, G4double> GetFatalEnergyCheckLevels() const;
    
    virtual std::pair<G4double, G4double> GetEnergyMomentumCheckLevels() const;
    
    inline void SetEnergyMomentumCheckLevels(G4double relativeLevel, G4double absoluteLevel)
    { epCheckLevels.first = relativeLevel;
        epCheckLevels.second = absoluteLevel; }
    
    virtual void ModelDescription(std::ostream& outFile) const ; //=0;
    
private:
    
    StorkHadronicInteraction(const G4HadronicInteraction &right );
    const G4HadronicInteraction& operator=(const G4HadronicInteraction &right);
    
protected:
    
    inline void SetModelName(const G4String& nam)
    { theModelName = nam; }
    
    inline G4bool IsBlocked() const { return isBlocked;}
    inline void Block() { isBlocked = true; }
    
    G4HadFinalState theParticleChange;
    // the G4HadFinalState object which is modified and returned
    // by address by the ApplyYourself method,
    // (instead of aParticleChange as found in G4VProcess)
    
    G4int verboseLevel;
    // control flag for output messages
    // 0: silent
    // 1: warning messages
    // 2: more
    // (instead of verboseLevel as found in G4VProcess)
    
    // these two have global validity energy range
    G4double theMinEnergy;
    G4double theMaxEnergy;
    
    G4bool isBlocked;
    
private:
    
    G4double recoilEnergyThreshold;
    
    G4String theModelName;
    
    std::pair<G4double, G4double> epCheckLevels;
    
    std::vector<std::pair<G4double, const G4Material *> > theMinEnergyList;
    std::vector<std::pair<G4double, const G4Material *> > theMaxEnergyList;
    std::vector<std::pair<G4double, const G4Element *> > theMinEnergyListElements;
    std::vector<std::pair<G4double, const G4Element *> > theMaxEnergyListElements;
    std::vector<const G4Material *> theBlockedList;
    std::vector<const G4Element *> theBlockedListElements;
};


#endif /* defined(__G4_STORK_AT__StorkHadronicInteraction__) */
