/*
StorkHPNeutronBuilder.hh

Created by:		Liam Russell
Date:			23-02-2012
Modified:       11-03-2013

Header file for StorkHPNeutronBuilderclass.

This class creates all of the physics processes, models, particles and data.
It takes a temperature and a bool flag. The temperature denotes the temperature
the cross sections were evaluated at, which determines whether doppler
broadening is used.

*/

#ifndef STORKHPNEUTRONBUILDER_H
#define STORKHPNEUTRONBUILDER_H

// Include header files
#include "G4VNeutronBuilder.hh"
#include "globals.hh"
#include "G4ProcessManager.hh"
#include "G4ParticleTypes.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleWithCuts.hh"

// Processes
#include "G4HadronElasticProcess.hh"
#include "G4NeutronInelasticProcess.hh"
#include "G4HadronCaptureProcess.hh"
#include "G4HadronFissionProcess.hh"

// Models
#include "StorkNeutronHPElastic.hh"
#include "StorkNeutronHPInelastic.hh"
#include "StorkNeutronHPCapture.hh"
#include "StorkNeutronHPFission.hh"

// Data Sets
#include "StorkNeutronHPCSData.hh"

//
#include "G4NeutronHPCaptureData.hh"
#include "G4NeutronHPFissionData.hh"
#include "G4NeutronHPElasticData.hh"
#include "G4NeutronHPInelasticData.hh"
//

#ifdef G4USE_TOPC
#include "topc.h"
#include "G4HadronicProcessStore.hh"
#endif


class StorkHPNeutronBuilder : public G4VNeutronBuilder
{
    public:
        StorkHPNeutronBuilder(G4String dir);
        ~StorkHPNeutronBuilder();

        void Build(G4HadronElasticProcess *aP);
        void Build(G4HadronFissionProcess *aP);
        void Build(G4HadronCaptureProcess *aP);
        void Build(G4NeutronInelasticProcess *aP);

        void SetMinEnergy(G4double aM) { theIMin = theMin = aM; }
        void SetMaxEnergy(G4double aM) { theIMax = theMax = aM; }
        void SetMinInelasticEnergy(G4double aM) { theIMin = aM; }
        void SetMaxInelasticEnergy(G4double aM) { theIMax = aM; }

        void SetFSTemperature(G4double FSTemp) { fsTemp = FSTemp; }

    private:

        // Applicability limits
        G4double theMin;
        G4double theMax;
        G4double theIMin;
        G4double theIMax;

        // User input and limits
        G4String dirName;
        G4double fsTemp;

        // Models
        StorkNeutronHPElastic *nElasticModel;
        StorkNeutronHPInelastic *nInelasticModel;
        StorkNeutronHPFission *nFissionModel;
        StorkNeutronHPCapture *nCaptureModel;

        // Data
        StorkNeutronHPCSData *HPElasticData;
        G4NeutronHPElasticData *theHPElasticData;
        StorkNeutronHPCSData *HPInelasticData;
        G4NeutronHPInelasticData *theHPInelasticData;
        StorkNeutronHPCSData *HPFissionData;
        G4NeutronHPFissionData *theHPFissionData;
        StorkNeutronHPCSData *HPCaptureData;
        G4NeutronHPCaptureData *theHPCaptureData;
};

#endif // STORKHPNEUTRONBUILDER_H
