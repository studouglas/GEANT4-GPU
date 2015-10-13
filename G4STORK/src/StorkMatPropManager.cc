/*
StorkMatPropManager.hh

Created by:		Liam Russell
Date:			17-08-2012
Modified:       11-03-2013

This class manages the material/property enumerators.  It contains the units,
unit names, and enumerator names of the material-property enumerators.

*/


#include "StorkMatPropManager.hh"


// Initialize static members
StorkMatPropManager* StorkMatPropManager::theMPManager = NULL;

// Static material/property values
const G4String
StorkMatPropManager::MAT_NAMES[MAX_MATS] = {"all","fuel", "coolant",
												"moderator", "poison", "controlrod"};
const G4String
StorkMatPropManager::PROP_NAMES[MAX_PROPS] = {"temperature","density",
												  "dimension","concentration", "position", "rotation"};
G4String
StorkMatPropManager::thePropType[MAX_PROPS] = {"material", "material",
                                                    "geometry", "material", "geometry", "geometry"};
G4double
StorkMatPropManager::thePropUnits[MAX_PROPS] = {kelvin, g/cm3, cm, perCent, cm, rad};

G4String
StorkMatPropManager::theUnitNames[MAX_PROPS] = {"K","g/cm3","cm","%","cm", "rad"};


// GetStorkMatPropManager()
// Static function used to get/create the material property manager.
StorkMatPropManager* StorkMatPropManager::GetStorkMatPropManager()
{
	if(!theMPManager)
		theMPManager = new StorkMatPropManager();

	return theMPManager;
}


// GetEnumKeyword()
// Get the description/name of the enumerator value (overloaded).
G4String StorkMatPropManager::GetEnumKeyword(MatEnum aMat) const
{
	return Capitalize(MAT_NAMES[aMat]);
}

G4String StorkMatPropManager::GetEnumKeyword(PropEnum aProp) const
{
	return Capitalize(PROP_NAMES[aProp]);
}


// ParseEnum()
// Find the enumerator value given a keyword which is at least 3 characters
// long (overloaded).
MatEnum StorkMatPropManager::ParseMatEnum(G4String key) const
{
	// Ensure key is at least 3 characters long
	G4int kLen = key.length();

	if(kLen > 2)
	{
		key.toLower();

		for(G4int i=0; i<MAX_MATS; i++)
		{
			if(key == G4String(MAT_NAMES[i].substr(0,kLen)))
				return (MatEnum)i;
		}
	}

	G4cerr << "*** ERROR:  Invalid material key " << key << G4endl;

	return MAX_MATS;
}

PropEnum StorkMatPropManager::ParsePropEnum(G4String key) const
{
	// Ensure key is at least 3 characters long
	G4int kLen = key.length();

	if(kLen > 2)
	{
		key.toLower();

		for(G4int i=0; i<MAX_PROPS; i++)
		{
			if(key == G4String(PROP_NAMES[i].substr(0,kLen)))
				return (PropEnum)i;
		}
	}

	G4cerr << "*** ERROR:  Invalid property key " << key << G4endl;

	return MAX_PROPS;
}

// GetPropType()
// Tells wether the change been made is a change to the material or to the geometry
G4String StorkMatPropManager::GetPropType(PropEnum aProp) const
{
    return thePropType[aProp];
}

// GetUnits()
// Get the units of a given property.
G4double StorkMatPropManager::GetUnits(PropEnum aProp) const
{
	return thePropUnits[aProp];
}


// GetUnitNames()
// Get the units of the property as a string.
G4String StorkMatPropManager::GetUnitsName(PropEnum aProp) const
{
	return theUnitNames[aProp];
}


// SetUnits()
// Set the units for a given property.s
void StorkMatPropManager::SetUnits(PropEnum aProp, G4double value,
									   G4String name)
{
	thePropUnits[aProp] = value;
	theUnitNames[aProp] = name;
}


// Capitalize()
// Capitalize the first letter of the string.
G4String StorkMatPropManager::Capitalize(G4String str) const
{
	str[0] = toupper(str[0]);

	return str;
}


