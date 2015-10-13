/*
StorkNeutronProcessBuilder.hh

Created by:		Liam Russell
Date:			23-02-2012
Modified:

Header file for StorkNeutronProcessBuilder class.

Based on G4NeutronBuilder, this builder class sets up all neutron physics
processes and models (fission, capture, elastic, inelastic, and step limiters).

*/

#ifndef STORKNEUTRONPROCESSBUILDER_H
#define STORKNEUTRONPROCESSBUILDER_H

// Include G4-STORK headers
#include "StorkHadronFissionProcess.hh"
#include "StorkHadronCaptureProcess.hh"
#include "StorkNeutronInelasticProcess.hh"
#include "StorkHadronElasticProcess.hh"
#include "StorkHPNeutronBuilder.hh"
#include "StorkTimeStepLimiter.hh"
#include "StorkUserBCStepLimiter.hh"
#include "StorkZeroBCStepLimiter.hh"

// Include Geant4 headers
#include "G4VNeutronBuilder.hh"
//#include "G4HadronElastic.hh"
#include "G4LFission.hh"
#include "G4NeutronRadCapture.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "globals.hh"
#include "G4DiffuseElastic.hh"

#include "StorkMaterial.hh"
#include "G4LogicalVolumeStore.hh"
// Include other headers
#include <vector>
#include <typeinfo>


class StorkNeutronProcessBuilder
{
    public:
        // Public member functions

        StorkNeutronProcessBuilder(std::vector<G4int>* pBCVec, std::vector<G4int>* rBCVec, G4String FSDirName, G4int KCalcType);
        ~StorkNeutronProcessBuilder();

        // Build and register the models
        void Build();
        void RegisterMe(G4VNeutronBuilder * aB)
        {
            theModelCollections.push_back(aB);
        }

        bool ExtractTemp(G4String name, G4double &temp);


    private:
        // Private member variables

        StorkNeutronInelasticProcess * theNeutronInelastic;
        StorkHadronFissionProcess * theNeutronFission;
        StorkHadronCaptureProcess  * theNeutronCapture;
        StorkHadronElasticProcess * theNeutronElastic;
        StorkTimeStepLimiter * theStepLimiter;
        StorkUserBCStepLimiter * TheUserBoundaryCond;
        StorkZeroBCStepLimiter * TheZeroBoundaryCond;

        G4DiffuseElastic *theHighElasticModel;
        //G4ChipsElasticModel *theHighElasticModel;
        //G4HadronElastic *theHighElasticModel;
        G4LFission *theHighFissionModel;
        G4NeutronRadCapture *theHighCaptureModel;

        std::vector<G4VNeutronBuilder *> theModelCollections;

        G4int kCalcType;
        G4String fsDirName;
        G4bool wasActivated;
};

#endif // STORKNEUTRONPROCESSBUILDER_H
