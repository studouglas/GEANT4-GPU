/*
StorkTripleFloat.hh

Created by:		Liam Russell
Date:			07-08-2011
Modified:		17-02-2011

Class used to replace G4ThreeVector for simple data storage and marshalling.

*/

#ifndef TRIPLEFLOAT_H
#define TRIPLEFLOAT_H

#include "G4ThreeVector.hh"

//MSH_BEGIN
class StorkTripleFloat
{
    public:
        // Public member functions

		StorkTripleFloat() {}
		StorkTripleFloat(G4ThreeVector &v3) { data = v3; };
		StorkTripleFloat(const G4ThreeVector &v3) { data = v3; }
		~StorkTripleFloat() {}

        G4double& operator[] (G4int i) { return data[i]; };
        G4ThreeVector GetData() {return data;}

	public:
        // Public member variables

		G4ThreeVector data;	//MSH: primitive
    
        void set(const G4ThreeVector coords) { data = coords;}
    
    
};
//MSH_END

#endif // TRIPLEFLOAT_H
