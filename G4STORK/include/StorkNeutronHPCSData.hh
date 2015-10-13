/*
StorkNeutronHPCSData.hh

Created by:		Liam Russell
Date:			22-06-2011
Modified:		17-02-2012

Header file for StorkNeutronHPCSData class.

Modified version of G4NeutronHP****Data classes (elastic, inelastic, fission,
or capture). This class sets up the appropriate cross section data for one of
the four reactions and calculates Doppler broadened cross sections.

The orginal version was modified to cover all four interaction types and it
allows for different temperatures in the cross sections.  This can negate the
need for Doppler broadening which speeds up the calculation.

*/

#ifndef NEUTRONHPCSDATA_H
#define NEUTRONHPCSDATA_H

#include "G4NeutronHPCaptureData.hh"
#include "G4NeutronHPElasticData.hh"
#include "G4NeutronHPInelasticData.hh"
#include "G4NeutronHPFissionData.hh"

#include "G4VCrossSectionDataSet.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4ParticleDefinition.hh"
#include "G4PhysicsTable.hh"
#include "G4SystemOfUnits.hh"

#include "G4Neutron.hh"
#include "G4ElementTable.hh"
#include "StorkNeutronHPDataConstructor.hh"

#include "G4Nucleus.hh"
#include "G4NucleiProperties.hh"
#include "G4Electron.hh"
#include "ElementNames.hh"


class StorkNeutronHPCSData : public G4VCrossSectionDataSet
{
    public:

        StorkNeutronHPCSData(char aType, G4String dirName, G4double FSTemp);
        ~StorkNeutronHPCSData();

        G4bool IsApplicable(const G4DynamicParticle*, const G4Element*);

    public:

        G4bool IsZAApplicable(const G4DynamicParticle* , G4double /*ZZ*/,
							  G4double /*AA*/)
            { return false;}

        //G4bool IsElementApplicable(const G4DynamicParticle*, G4int, const G4Material*);
        //G4double GetElementCrossSection(const G4DynamicParticle* dynPart, G4int Z, const G4Material* mat);
        G4bool IsIsoApplicable( const G4DynamicParticle* dp ,
                                                G4int /*Z*/ , G4int /*A*/ ,
                                                const G4Element* /*elm*/ ,
                                                const G4Material* /*mat*/ );

        G4double GetIsoCrossSection( const G4DynamicParticle* dp ,
                                   G4int /*Z*/ , G4int /*A*/ ,
                                   const G4Isotope* /*iso*/  ,
                                   const G4Element* element ,
                                   const G4Material* material );

        G4double GetCrossSection(const G4DynamicParticle*, const G4Element*,
								 G4double aT);
        void BuildPhysicsTable(const G4ParticleDefinition&)
        {};
        void BuildPhysicsTable(const G4ParticleDefinition&, G4String dirName);
        void DumpPhysicsTable(const G4ParticleDefinition&);


    private:

        G4PhysicsTable * theCrossSections;

        G4int numMaxOL;
        G4int numIL;
        char reactionType;

        G4double ke_cache;
        G4double xs_cache;
        G4double fsTemp;

        const G4Element* element_cache;
        const G4Material* material_cache;
};

#endif // NEUTRONHPCSDATA_H
