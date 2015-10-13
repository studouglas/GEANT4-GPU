/*
StorkMatPropManager.hh

Created by:		Liam Russell
Date:			17-08-2012
Modified:       11-03-2013

This class manages the material/property enumerators which are defined below.
It assigns units to the properties and contains functions that return enumerator
keywords, enumerators, and units.

This is a static class and may only be instantiated once.


NOTE: if a new material or property is added in the enumerators, the
corresponding names and units should be added to MAT_NAMES or PROP_NAMES,
PROP_UNITS and PROP_UNIT_NAMES.

*/


#ifndef STORKMATERIALPROPERTYMANAGER_H
#define STORKMATERIALPROPERTYMANAGER_H


// MatPropEnumerators

// Material and property enumerators used to identify/change the properties
// of various materials
enum MatEnum {all=0, fuel, coolant, moderator, poison, controlrod, MAX_MATS};
enum PropEnum {temperature=0, density, dimension, concentration, position, rotation, MAX_PROPS};

#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"


class StorkMatPropManager
{
	public:
        // Public member functions

        // Get the current material-property manager
		static StorkMatPropManager* GetStorkMatPropManager();

        // Destructor
		virtual ~StorkMatPropManager() {;}

		// Overloaded function that return enum keywords
		G4String GetEnumKeyword(MatEnum aMat) const;
		G4String GetEnumKeyword(PropEnum aProp) const;

		// Overloaded function that returns enum given a keyword
		MatEnum ParseMatEnum(G4String key) const;
		PropEnum ParsePropEnum(G4String key) const;

		// Get property units
		G4double GetUnits(PropEnum aProp) const;
		G4String GetUnitsName(PropEnum aProp) const;
		G4String GetPropType(PropEnum aProp) const;

		// Set the units of a property
		void SetUnits(PropEnum aProp, G4double value, G4String name);


	private:
        // Private member functions

        // Private default constructor
		StorkMatPropManager() {;}

        // Capitalize a string
		G4String Capitalize(G4String str) const;

	private:
        // Private member data

		static StorkMatPropManager *theMPManager;

		static const G4String MAT_NAMES[MAX_MATS];
		static const G4String PROP_NAMES[MAX_PROPS];

        static G4String thePropType[MAX_PROPS];
		static G4double thePropUnits[MAX_PROPS];
		static G4String theUnitNames[MAX_PROPS];
};

#endif // STORKMATERIALPROPERTYMANAGER_H
