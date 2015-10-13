/*
StorkHPNeutronBuilder.cc

Created by:		Liam Russell
Date:			23-02-2012
Modified:       11-03-2013

Source code for StorkHPNeutronBuilder class.

*/


// Include header file
#include "StorkHPNeutronBuilder.hh"

// Constructor
StorkHPNeutronBuilder::StorkHPNeutronBuilder(G4String dir)
{
    // Set temperature for cross sections
	dirName = dir;

	// Set default FS temperature
	fsTemp=0.;

	// Initialize the applicability limits
	theMin = theIMin = 0.*MeV;
	theMax = theIMax = 20.*MeV;

    // Initialize data pointers
	theHPCaptureData = 0;
	theHPInelasticData = 0;
	theHPElasticData = 0;
	theHPFissionData = 0;
	HPCaptureData = 0;
	HPInelasticData = 0;
	HPElasticData = 0;
	HPFissionData = 0;
	nElasticModel = 0;
	nInelasticModel = 0;
	nFissionModel = 0;
	nCaptureModel = 0;

	// Insure that the slave processes do not output the physics process table
#ifdef G4USE_TOPC
	if(!TOPC_is_master())
	{
		G4HadronicProcessStore *theStore = G4HadronicProcessStore::Instance();
		theStore->SetVerbose(0);
	}
#endif
}

// Destructor
StorkHPNeutronBuilder::~StorkHPNeutronBuilder()
{
    // Delete the cross section data
    if(theHPCaptureData!=NULL)
        delete theHPCaptureData;
    if(theHPInelasticData!=NULL)
        delete theHPInelasticData;
	if(theHPFissionData!=NULL)
        delete theHPFissionData;
	if(theHPElasticData!=NULL)
        delete theHPElasticData;
	if(HPCaptureData!=NULL)
        delete HPCaptureData;
	if(HPInelasticData!=NULL)
        delete HPInelasticData;
	if(HPFissionData!=NULL)
        delete HPFissionData;
	if(HPElasticData!=NULL)
        delete HPElasticData;
}


// Build( Elastic Process )
// Build the elastic model and data.
// Set the limits of applicability for the model
void StorkHPNeutronBuilder::Build(G4HadronElasticProcess *aP)
{
    // Create the model and data
    if(nElasticModel==0) nElasticModel = new StorkNeutronHPElastic(fsTemp);
    if(theHPElasticData==0&&HPElasticData==0)
    {
        if(dirName == "DEFAULT")
            theHPElasticData = new G4NeutronHPElasticData();
        else
            HPElasticData = new StorkNeutronHPCSData('E', dirName, fsTemp);
    }


    // Set the limits of the model
    nElasticModel->SetMinEnergy(theMin);
    nElasticModel->SetMaxEnergy(theMax);

    // Register both
    if(dirName == "DEFAULT")
        aP->AddDataSet(theHPElasticData);
    else
        aP->AddDataSet(HPElasticData);
    aP->RegisterMe(nElasticModel);
}


// Build( Inelastic Process )
// Build the elastic model and data.
// Set the limits of applicability for the model
void StorkHPNeutronBuilder::Build(G4NeutronInelasticProcess *aP)
{
    // Create the model and data
    if(nInelasticModel==0) nInelasticModel = new StorkNeutronHPInelastic(fsTemp);
    if(theHPInelasticData==0&&HPInelasticData==0)
    {
        if(dirName == "DEFAULT")
            theHPInelasticData = new G4NeutronHPInelasticData();
        else
            HPInelasticData = new StorkNeutronHPCSData('I', dirName, fsTemp);
    }
    // Set the limits of the model
    nInelasticModel->SetMinEnergy(theIMin);
    nInelasticModel->SetMaxEnergy(theIMax);

    // Register both
    if(dirName == "DEFAULT")
        aP->AddDataSet(theHPInelasticData);
    else
        aP->AddDataSet(HPInelasticData);
    aP->RegisterMe(nInelasticModel);
}


// Build( Fission Process )
// Build the elastic model and data.
// Set the limits of applicability for the model
void StorkHPNeutronBuilder::Build(G4HadronFissionProcess *aP)
{
    // Create the model and data
    if(nFissionModel==0) nFissionModel = new StorkNeutronHPFission(fsTemp);
    if(theHPFissionData==0&&HPFissionData==0)
    {
        if(dirName == "DEFAULT")
            theHPFissionData = new G4NeutronHPFissionData();
        else
            HPFissionData = new StorkNeutronHPCSData('F', dirName, fsTemp);
    }
    // Set the limits of the model
    nFissionModel->SetMinEnergy(theMin);
    nFissionModel->SetMaxEnergy(theMax);
    // Register both
    if(dirName == "DEFAULT")
        aP->AddDataSet(theHPFissionData);
    else
        aP->AddDataSet(HPFissionData);
    aP->RegisterMe(nFissionModel);
}


// Build( Capture Process )
// Build the elastic model and data.
// Set the limits of applicability for the model
void StorkHPNeutronBuilder::Build(G4HadronCaptureProcess *aP)
{
    // Create the model and data
    if(nCaptureModel==0) nCaptureModel = new StorkNeutronHPCapture(fsTemp);
    if(theHPCaptureData==0&&HPCaptureData==0)
    {
        if(dirName == "DEFAULT")
            theHPCaptureData = new G4NeutronHPCaptureData();
        else
            HPCaptureData = new StorkNeutronHPCSData('C', dirName, fsTemp);
    }
    // Set the limits of the model
    nCaptureModel->SetMinEnergy(theMin);
    nCaptureModel->SetMaxEnergy(theMax);

    // Register both
    if(dirName == "DEFAULT")
        aP->AddDataSet(theHPCaptureData);
    else
        aP->AddDataSet(HPCaptureData);
    aP->RegisterMe(nCaptureModel);
}


