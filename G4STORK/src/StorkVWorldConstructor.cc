/*
StorkVWorldConstructor.cc

Created by:		Liam Russell
Date:			23-05-2012
Modified:       11-03-2013

Source file for StorkVWorldConstructor class.

*/

// Include header file
#include "StorkVWorldConstructor.hh"


// Constructor
StorkVWorldConstructor::StorkVWorldConstructor()
: worldLogical(0), worldPhysical(0), worldVisAtt(0)
{
	// Initialize member variables
	geomChanged = true;
	matChanged = true;
	Initiation = true;
	physChanged = false;
	encWorldDim = G4ThreeVector(0.,0.,0.);
	reactorDim = encWorldDim;
	charPosition = new std::vector<G4int>;
}


// Destructor
StorkVWorldConstructor::~StorkVWorldConstructor()
{
	// Destroy all materials, elements and isotopes
    DestroyMaterials();

    if(worldVisAtt)
        delete worldVisAtt;
}


// DestroyMaterials()
// Delete all materials, elements, and isotopes
void StorkVWorldConstructor::DestroyMaterials()
{
    // Destroy all allocated materials, elements and isotopes
    size_t i;

    G4MaterialTable *matTable = (G4MaterialTable*)G4Material::GetMaterialTable();
    for(i=0; i<matTable->size(); i++)
    {
        if((*(matTable))[i])
            delete (*(matTable))[i];
    }
    matTable->clear();

    G4ElementTable *elemTable = (G4ElementTable*)G4Element::GetElementTable();
    for(i=0; i<elemTable->size(); i++)
    {
        if((*(elemTable))[i])
            delete (*(elemTable))[i];
    }
    elemTable->clear();

    G4IsotopeTable *isoTable = (G4IsotopeTable*)G4Isotope::GetIsotopeTable();
    for(i=0; i<isoTable->size(); i++)
    {
        if((*(isoTable))[i])
            delete (*(isoTable))[i];
    }
    isoTable->clear();

    return;
}


// GetWorldProperty()
// Returns the current world property associated with the given MatPropPair
G4double StorkVWorldConstructor::GetWorldProperty(MatPropPair matProp)
{
    if (IsApplicable(matProp))
    {
		return *(variablePropMap[matProp]) / theMPMan->GetUnits(matProp.second);
    }
    else
    {
        return -1.0;
    }
}


// IsApplicable()
// Determines whether the given material and property pair is valid in the
// current world
G4bool StorkVWorldConstructor::IsApplicable(MatPropPair matProp)
{
    if(variablePropMap.find(matProp)==variablePropMap.end())
        return false;
	else
	    return true;
}


// ConstructNewWorld()
// Set the initial properties (material-properties), sensitive detector,
// and neutron filter for the sensitive detector.  Then construct the world
// using a derived class implementation of ConstructWorld().
G4VPhysicalVolume*
StorkVWorldConstructor::ConstructNewWorld(const StorkParseInput* infile)
{
    // Get any initial changes (vector may be empty)
	initialChanges = infile->GetIntialWorldProperties();

	// Build sensitive detector
    G4SDManager *sDMan = G4SDManager::GetSDMpointer();
    sDReactor = new StorkNeutronSD("Reactor", infile->GetKCalcType(), infile->GetInstantDelayed(), infile->GetPrecursorDelayed());
    sDMan->AddNewDetector(sDReactor);

    // Add filters to the sensitive detectors so that they only track neutrons
    G4SDParticleFilter *nFilter = new G4SDParticleFilter("neutronFilter",
														 "neutron");
    sDReactor->SetFilter(nFilter);


    // Add any initial changes to the world properties
	if(G4int(initialChanges.size()) > 0)
		UpdateWorldProperties(initialChanges);

	return ConstructWorld();
}


// UpdateWorld()
// Update the variable material-properties.
G4VPhysicalVolume*
StorkVWorldConstructor::UpdateWorld(StorkMatPropChangeVector changes)
{
	if(UpdateWorldProperties(changes))
	{
		return ConstructWorld();
	}
	else
	{
		return worldPhysical;
	}
}


// UpdateWorldProperties()
// Check whether ALL of the proposed changes are valid.  If so, apply the
// changes.
G4bool
StorkVWorldConstructor::UpdateWorldProperties(StorkMatPropChangeVector changes)
{
		// Check whether proposed changes are applicable to the world
	for(G4int i=0; i<G4int(changes.size()); i++)
	{
		if(!IsApplicable(changes[i].GetMatPropPair()))
		{
			G4cerr << "*** ERROR: Inapplicable world changes."
				   << " Returning original world." << G4endl;

			return false;
		}
	}

	// Apply changes
	G4bool changed = false;
	for(G4int i=0; i<G4int(changes.size()); i++)
	{
        G4double previous = *(variablePropMap[changes[i].GetMatPropPair()]);

	    // Checks to make sure there was a change in the properties (to avoid unecessary optimizations)
	    if(previous != changes[i].change)
	    {
	        // Checks to see if we are changing a material property, or not
            if(theMPMan->GetPropType(changes[i].GetMatPropPair().second) == "material")
            {
                physChanged = true;
                matChanged = true;
            }
	        // Change the variable properties
            *(variablePropMap[changes[i].GetMatPropPair()]) = changes[i].change;
            changed = true;
	    }
	}

	return changed;
}



// SaveMaterialTemperatureHeader()
// Outputs the header of the temperature data file
void
StorkVWorldConstructor::SaveMaterialTemperatureHeader(G4String fname)
{
    // Declare and open file stream
	std::ofstream outFile(fname.c_str(),std::ofstream::app);

	// Check that stream is ready for use
	if(!outFile.good())
	{
		G4cerr << G4endl << "ERROR:  Could not write material temperatures to file. " << G4endl
			   << "Improper file name: " << fname << G4endl
			   << "Continuing program without material temperature data output" << G4endl;

        return;
	}

    outFile.fill(' ');
    outFile << "All temperatures are given in Kelvin." << G4endl;
    outFile << "#########################################################" << G4endl;
    outFile << "Run # ";

    // Cycle through all the elements of the map and output the name of each material
	for(std::map<G4String,G4Material*>::iterator it = matMap.begin(); it != matMap.end(); it++)
	{
	    // Need at least 6 space to output the number 5 significant digits and comma
	    if((*it).first.size() < 6)
	    {
	        outFile << std::setw(6) << (*it).first << " ";
	        charPosition->push_back(6);
	    }
	    else
	    {
	        outFile << (*it).first << " ";
	        charPosition->push_back((*it).first.size());
	    }
	}
    outFile << G4endl;
    outFile.close();
}


// SaveMaterialTemperatures()
// Outputs temperatures to file specified in StorkParseInput
void
StorkVWorldConstructor::SaveMaterialTemperatures(G4String fname, G4int runNumber)
{
    // Declare and open file stream
	std::ofstream outFile(fname.c_str(),std::ofstream::app);

	// Check that stream is ready for use
	if(!outFile.good())
	{
		G4cerr << G4endl << "ERROR:  Could not write material temperatures to file. " << G4endl
			   << "Improper file name: " << fname << G4endl
			   << "Continuing program without material temperature data output" << G4endl;

		return;
	}

    outFile.fill(' ');
    G4int matNum = 0;

    // Print the run number
    outFile << std::setw(5) << runNumber << " ";

    // Print the temperature of all the material one after the other
    for(std::map<G4String,G4Material*>::iterator it = matMap.begin(); it != matMap.end(); it++)
	{
	    if(matNum < G4int(charPosition->size()))
	    {
	        outFile << std::resetiosflags(std::ios_base::floatfield) << std::right
                << std::setprecision(5)
                << std::setw((*charPosition)[matNum]) << (*it).second->GetTemperature() << " ";
            matNum++;
	    }

	    // This is in case the length of the matMap changed from when the header was created
	    else
	    {
	       outFile << std::resetiosflags(std::ios_base::floatfield) << std::right
                << std::setprecision(5)
                << std::setw(12) << (*it).second->GetTemperature() << " ";
	    }

	}
	outFile << G4endl;
    outFile.close();
}
