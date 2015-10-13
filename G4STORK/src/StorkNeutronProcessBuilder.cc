/*
StorkNeutronProcessBuilder.cc

Created by:		Liam Russell
Date:			23-02-2012
Modified:       11-03-2013

Source code for StorkNeutronProcessBuilder class.

*/


// Include header file
#include "StorkNeutronProcessBuilder.hh"


// Constructor
StorkNeutronProcessBuilder::StorkNeutronProcessBuilder(std::vector<G4int>* pBCVec, std::vector<G4int>* rBCVec, G4String FSDirName, G4int KCalcType)
:wasActivated(false)
{
    // Create the physics processes
    theNeutronElastic = new StorkHadronElasticProcess;
    theNeutronInelastic = new StorkNeutronInelasticProcess;
    theNeutronCapture = new StorkHadronCaptureProcess;
    theNeutronFission = new StorkHadronFissionProcess;
    theStepLimiter = new StorkTimeStepLimiter;
    TheUserBoundaryCond = new StorkUserBCStepLimiter(pBCVec,rBCVec);
    TheZeroBoundaryCond = new StorkZeroBCStepLimiter(pBCVec,rBCVec);

    kCalcType = KCalcType;
    fsDirName=FSDirName;
}


// Destructor
StorkNeutronProcessBuilder::~StorkNeutronProcessBuilder()
{
    // Delete the processes
    if(theNeutronElastic)
        delete theNeutronElastic;
    if(theNeutronInelastic)
        delete theNeutronInelastic;
    if(theNeutronCapture)
        delete theNeutronCapture;
    if(theNeutronFission)
        delete theNeutronFission;
    if(theStepLimiter)
        delete theStepLimiter;
    if(TheUserBoundaryCond)
        delete TheUserBoundaryCond;
    if(TheZeroBoundaryCond)
        delete TheZeroBoundaryCond;
}


// Build()
// Build the data and models for the neutron processes
void StorkNeutronProcessBuilder::Build()
{
    G4double fsTemp=0.;
    wasActivated = true;
    StorkHPNeutronBuilder *aHPBuilder;

    if(fsDirName!="DEFAULT")
    {
        if(fsDirName[fsDirName.size()-1]=='/')
            fsDirName.erase(fsDirName.size()-1);
        ExtractTemp(fsDirName, fsTemp);
        setenv("G4NEUTRONHPDATA",fsDirName,1);
        std::vector<G4VNeutronBuilder *>::iterator i;
        for(i=theModelCollections.begin(); i!=theModelCollections.end(); i++)
        {
            aHPBuilder = dynamic_cast<StorkHPNeutronBuilder*>(*i);
            if(aHPBuilder)
                aHPBuilder->SetFSTemperature(fsTemp);
        }
    }

    // Build the models and data for the neutron processes (all energies)
    std::vector<G4VNeutronBuilder *>::iterator i;
    for(i=theModelCollections.begin(); i!=theModelCollections.end(); i++)
    {
        aHPBuilder = dynamic_cast<StorkHPNeutronBuilder*>(*i);

        // Only build low energy models for the elastic processes
        if(aHPBuilder)
            (*i)->Build(theNeutronElastic);

        (*i)->Build(theNeutronInelastic);
        (*i)->Build(theNeutronCapture);
        (*i)->Build(theNeutronFission);
    }

    // Create the high energy elastic model
    theHighElasticModel = new G4DiffuseElastic();
    theHighElasticModel->SetMinEnergy(20.0*MeV);
    theHighElasticModel->SetMaxEnergy(20000.*GeV);

    // Register model with elastic process
    theNeutronElastic->RegisterMe(theHighElasticModel);

    // Create the high energy fission model
    theHighFissionModel = new G4LFission();

    // Set the minimum energy limit
    theHighFissionModel->SetMinEnergy(20.0*MeV);
    theHighFissionModel->SetMaxEnergy(20000.*GeV);

    // Register model with fission process
    theNeutronFission->RegisterMe(theHighFissionModel);

    theHighCaptureModel = new G4NeutronRadCapture();

    // Set the minimum energy limit
    theHighCaptureModel->SetMinEnergy(20.0*MeV);
    theHighCaptureModel->SetMaxEnergy(20000.*GeV);


    // Register model with fission process
    theNeutronCapture->RegisterMe(theHighCaptureModel);

    // Add processes to the process manager
    G4ProcessManager * theProcMan = G4Neutron::Neutron()->GetProcessManager();

    theProcMan->AddDiscreteProcess(theNeutronElastic);
    theProcMan->AddDiscreteProcess(theNeutronInelastic);
    theProcMan->AddDiscreteProcess(theNeutronCapture);
    theProcMan->AddDiscreteProcess(theNeutronFission);
    if(kCalcType!=2)
        theProcMan->AddDiscreteProcess(theStepLimiter);
    theProcMan->AddDiscreteProcess(TheUserBoundaryCond);
    theProcMan->AddDiscreteProcess(TheZeroBoundaryCond);
}

bool StorkNeutronProcessBuilder::ExtractTemp(G4String name, G4double &temp)
{
    bool check = false;
    std::stringstream ss;
    G4int index = name.size()-1, startPos=1, endPos=0;
    while(index>=0)
    {
        if(check)
        {
            if(((name[index]>='0')&&(name[index]<='9'))||(name[index]=='.'))
            {
                startPos--;
            }
            else
            {
                break;
            }
        }
        else
        {
            if((name[index]>='0')&&(name[index]<='9'))
            {
                if((index+1==int(name.size()))||!((name[index+1]=='k')||(name[index+1]=='K')))
                {
                    return false;
                }
                check = true;
                startPos=endPos=index;
            }
        }
        index--;
    }

    if(endPos>=startPos)
    {
        G4String temperature = name.substr(startPos, endPos-startPos+1);
        ss.str(temperature);
        ss >> temp;
    }

    return check;
}

