/*
StorkInterpManager.hh

Created by:		Liam Russell
Date:			10-08-2012
Modified:       11-03-2012

Header for StorkInterpManager class.  Used to store and access interpolation
vectors based on a material and property being changed with respect to time.

*/


#ifndef NSINTERPMANAGER_H
#define NSINTERPMANAGER_H

// Include header files
#include "StorkInterpVector.hh"
#include "StorkContainers.hh"


class StorkInterpManager
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkInterpManager() {}
        ~StorkInterpManager() {}

        // Create a vector of data points that interpolation can act upon
        G4bool CreateInterpVector(G4String name, G4String dataFile,
								  MatPropPair mpPair, G4double uy = 1.,
								  G4double ux = 1.)
        {
            MatPropInterpolator anInterpolator;

            // Set name in name vector
            anInterpolator.first = name;

            // Set matProp pair
            anInterpolator.second = mpPair;

            // Load interpolation vector from file
            anInterpolator.third.InitInterpData(dataFile,uy,ux);

            // Set the units of the Y data
            anInterpolator.fourth = uy;

            // adds the StorkTriple anInterpolator to the theInterps vector
            theInterps.push_back(anInterpolator);

            return true;
        }

        // Returns an element of theInterps the vector
        const MatPropInterpolator* GetMatPropInterpolator(G4int index) const
        {
            return &(theInterps[index]);
        }

        // Sets the [] class operator to return a reference to an element of the
        // theInterps vector
        const MatPropInterpolator* operator[] (G4int index) const
        {
            return &(theInterps[index]);
        }

		// Get the number of interpolation vectors
        G4int GetNumberOfInterpVectors() const
        {
            return G4int(theInterps.size());
        }
        G4bool IsMatModify() const
        {
            G4bool matModify=false;
            G4String propName;
            for(G4int i=0; i<G4int(theInterps.size()); i++)
        	{
				propName = matPropEnumList->GetEnumKeyword((theInterps[i].second).second);

				if((propName!="dimension")||(propName!="position")||(propName!="rotation"))
                    matModify=true;
        	}
        	return matModify;
        }
        // Get a property change vector of all changes to be made
        StorkMatPropChangeVector GetStorkMatPropChanges(G4double time) const
        {
        	StorkMatPropChangeVector theChanges;
        	StorkMatPropChange aChange;

        	for(G4int i=0; i<G4int(theInterps.size()); i++)
        	{
				aChange.Set(theInterps[i].second,theInterps[i].third.GetY(time));
				theChanges.push_back(aChange);
        	}

        	return theChanges;
        }


    private:
        // Private member variables

        // Vector of interpolating data (name, material-property, data, units)
        std::vector<MatPropInterpolator> theInterps;
        StorkMatPropManager* matPropEnumList;
};

#endif // NSINTERPMANAGER_H
