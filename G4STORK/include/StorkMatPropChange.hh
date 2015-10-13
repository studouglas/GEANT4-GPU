/*
StorkMatPropChange.hh

Created by:		Liam Russell
Date:			09-08-2012
Modified:       11-03-2013

Definition of StorkMatPropChange class.

The StorkMatPropChange is a marshallable container class that holds a new
(double) value for a given material-property pair.

*/

#ifndef PROPERTYCHANGE_HH_INCLUDED
#define PROPERTYCHANGE_HH_INCLUDED

#include "StorkContainers.hh"

//MSH_BEGIN
class StorkMatPropChange
{
    public:
        // Public member functions

		// Default constructor
        StorkMatPropChange()
        :matN(0), propN(0), change(0.0)
        {}

        // Constructors
        StorkMatPropChange(G4int mat, G4int p, G4double c)
        :matN(mat), propN(p), change(c)
        {}

        StorkMatPropChange(MatPropPair aPair, G4double c)
        :matN(aPair.first), propN(aPair.second), change(c)
        {}

        // Destructor
		virtual ~StorkMatPropChange() {}

		// Overloaded set functions
        void Set(G4int mat, G4int p, G4double c)
        {
        	matN = mat;
        	propN = p;
        	change = c;
		}

		void Set(MatPropPair aPair, G4double c)
		{
			matN = G4int(aPair.first);
			propN = G4int(aPair.second);
			change = c;
		}

		// Get the material-property designation of the property change
		MatPropPair GetMatPropPair() const
		{
			return MatPropPair(MatEnum(matN),PropEnum(propN));
		}


    public:
		// Public member variables

        G4int matN; //MSH: primitive
        G4int propN; //MSH: primitive
        G4double change; //MSH: primitive
};
//MSH_END

#endif // PROPERTYCHANGE_HH_INCLUDED
