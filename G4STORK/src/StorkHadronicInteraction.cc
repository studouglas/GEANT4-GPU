//
//  StorkHadronicInteraction.cpp
//  G4-STORK_AT
//
//  Created by Andrew Tan on 2014-07-30.
//  Copyright (c) 2014 andrewtan. All rights reserved.
//

#include "StorkHadronicInteraction.hh"
#include "G4HadronicInteraction.hh"
#include "G4SystemOfUnits.hh"
#include "G4HadronicInteractionRegistry.hh"
#include "G4HadronicException.hh"

StorkHadronicInteraction::StorkHadronicInteraction(const G4String& modelName) :
verboseLevel(0), theMinEnergy(0.0), theMaxEnergy(25.0*GeV),
isBlocked(false), recoilEnergyThreshold(0.0), theModelName(modelName),
epCheckLevels(DBL_MAX, DBL_MAX)
{
    G4HadronicInteractionRegistry::Instance()->RegisterMe(this);
}


StorkHadronicInteraction::~StorkHadronicInteraction()
{
    G4HadronicInteractionRegistry::Instance()->RemoveMe(this);
}


G4double
StorkHadronicInteraction::SampleInvariantT(const G4ParticleDefinition*,
                                        G4double, G4int, G4int)
{
    return 0.0;
}

G4double StorkHadronicInteraction::GetMinEnergy(
                                             const G4Material *aMaterial, const G4Element *anElement ) const
{
    if( IsBlocked(aMaterial) ) { return 0.0; }
    if( IsBlocked(anElement) ) { return 0.0; }
    size_t length = theMinEnergyListElements.size();
    if(0 < length) {
        for(size_t i=0; i<length; ++i ) {
            if( anElement == theMinEnergyListElements[i].second )
            { return theMinEnergyListElements[i].first; }
        }
    }
    length = theMinEnergyList.size();
    if(0 < length) {
        for(size_t i=0; i<length; ++i ) {
            if( aMaterial == theMinEnergyList[i].second )
            { return theMinEnergyList[i].first; }
        }
    }
    if(IsBlocked()) { return 0.0; }
    if( verboseLevel > 1 ) {
        G4cout << "*** Warning from HadronicInteraction::GetMinEnergy" << G4endl
        << "    material " << aMaterial->GetName()
        << " not found in min energy List" << G4endl;
    }
    return theMinEnergy;
}

void StorkHadronicInteraction::SetMinEnergy(G4double anEnergy,
                                         const G4Element *anElement )
{
    if( IsBlocked(anElement) ) {
        G4cout << "*** Warning from HadronicInteraction::SetMinEnergy" << G4endl
        << "    The model is not active for the Element  "
        << anElement->GetName() << "." << G4endl;
    }
    size_t length = theMinEnergyListElements.size();
    if(0 < length) {
        for(size_t i=0; i<length; ++i ) {
            if( anElement == theMinEnergyListElements[i].second )
            {
                theMinEnergyListElements[i].first = anEnergy;
                return;
            }
        }
    }
    theMinEnergyListElements.push_back(std::pair<G4double, const G4Element *>(anEnergy, anElement));
}

void StorkHadronicInteraction::SetMinEnergy(G4double anEnergy,
                                         const G4Material *aMaterial )
{
    if( IsBlocked(aMaterial) ) {
        G4cout << "*** Warning from HadronicInteraction::SetMinEnergy" << G4endl
        << "    The model is not active for the Material "
        << aMaterial->GetName() << "." << G4endl;
    }
    size_t length = theMinEnergyList.size();
    if(0 < length) {
        for(size_t i=0; i<length; ++i ) {
            if( aMaterial == theMinEnergyList[i].second )
            {
                theMinEnergyList[i].first = anEnergy;
                return;
            }
        }
    }
    theMinEnergyList.push_back(std::pair<G4double, const G4Material *>(anEnergy, aMaterial));
}

G4double StorkHadronicInteraction::GetMaxEnergy(const G4Material *aMaterial,
                                             const G4Element *anElement ) const
{
    if( IsBlocked(aMaterial) ) { return 0.0; }
    if( IsBlocked(anElement) ) { return 0.0; }
    size_t length = theMaxEnergyListElements.size();
    if(0 < length) {
        for(size_t i=0; i<length; ++i ) {
            if( anElement == theMaxEnergyListElements[i].second )
            { return theMaxEnergyListElements[i].first; }
        }
    }
    length = theMaxEnergyList.size();
    if(0 < length) {
        for(size_t i=0; i<length; ++i ) {
            if( aMaterial == theMaxEnergyList[i].second )
            { return theMaxEnergyList[i].first; }
        }
    }
    if(IsBlocked()) { return 0.0; }
    if( verboseLevel > 1 ) {
        G4cout << "*** Warning from HadronicInteraction::GetMaxEnergy" << G4endl
        << "    material " << aMaterial->GetName()
        << " not found in min energy List" << G4endl;
    }
    return theMaxEnergy;
}

void StorkHadronicInteraction::SetMaxEnergy(G4double anEnergy,
                                         const G4Element *anElement )
{
    if( IsBlocked(anElement) ) {
        G4cout << "*** Warning from HadronicInteraction::SetMaxEnergy" << G4endl
        << "Warning: The model is not active for the Element  "
        << anElement->GetName() << "." << G4endl;
    }
    size_t length = theMaxEnergyListElements.size();
    if(0 < length) {
        for(size_t i=0; i<length; ++i ) {
            if( anElement == theMaxEnergyListElements[i].second )
            {
                theMaxEnergyListElements[i].first = anEnergy;
                return;
            }
        }
    }
    theMaxEnergyListElements.push_back(std::pair<G4double, const G4Element *>(anEnergy, anElement));
}

void StorkHadronicInteraction::SetMaxEnergy(G4double anEnergy,
                                         const G4Material *aMaterial )
{
    if( IsBlocked(aMaterial) ) {
        G4cout << "*** Warning from HadronicInteraction::SetMaxEnergy" << G4endl
        << "Warning: The model is not active for the Material "
        << aMaterial->GetName() << "." << G4endl;
    }
    size_t length = theMaxEnergyList.size();
    if(0 < length) {
        for(size_t i=0; i<length; ++i ) {
            if( aMaterial == theMaxEnergyList[i].second )
            {
                theMaxEnergyList[i].first = anEnergy;
                return;
            }
        }
    }
    theMaxEnergyList.push_back(std::pair<G4double, const G4Material *>(anEnergy, aMaterial));
}

void StorkHadronicInteraction::DeActivateFor( const G4Material *aMaterial )
{
    theBlockedList.push_back(aMaterial);
}

void G4HadronicInteraction::DeActivateFor( const G4Element *anElement )
{
    theBlockedListElements.push_back(anElement);
}


G4bool StorkHadronicInteraction::IsBlocked(const G4Material* aMaterial) const
{
    for (size_t i=0; i<theBlockedList.size(); ++i) {
        if (aMaterial == theBlockedList[i]) return true;
    }
    return false;
}


G4bool StorkHadronicInteraction::IsBlocked(const G4Element* anElement) const
{
    for (size_t i=0; i<theBlockedListElements.size(); ++i) {
        if (anElement == theBlockedListElements[i]) return true;
    }
    return false;
}

const std::pair<G4double, G4double> StorkHadronicInteraction::GetFatalEnergyCheckLevels() const
{
	// default level of Check
	return std::pair<G4double, G4double>(2.*perCent, 1. * GeV);
}

std::pair<G4double, G4double>
StorkHadronicInteraction::GetEnergyMomentumCheckLevels() const
{
    return epCheckLevels;
}


void StorkHadronicInteraction::ModelDescription(std::ostream& outFile) const
{
    outFile << "The description for this model has not been written yet.\n";
}

