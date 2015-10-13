/*
StorkInterpVector.hh

Created by:		Wesley Ford
Date:			23-05-2012
Modified:       10-08-2012

Header for StorkInterpVector class, based off of the G4NeutronHPVector class.

This class sets up an interpolation scheme from a given data file or data
vector and can interpolates data given to it using the GetY member function.

*/

#ifndef NSINTERPVECTOR_H
#define NSINTERPVECTOR_H
#include "G4InterpolationManager.hh"
#include "G4NeutronHPInterpolator.hh"
#include <vector>
#include <iostream>
#include "globals.hh"
#include <utility>
#include <fstream>
#include "StorkMatPropManager.hh"

typedef std::pair<G4double,G4double> DataPoint;
typedef std::vector<DataPoint> DataPointVec;


class StorkInterpVector
{
    public:
        // Public member functions

        // Constructors (default, copy, initializing, and assignment operator)
        StorkInterpVector();
        StorkInterpVector(const StorkInterpVector& right);
        StorkInterpVector(G4String dataFile);
        StorkInterpVector& operator=(const StorkInterpVector &right);

        // Destructor
        virtual ~StorkInterpVector() {;}

        // Initialize interpolation data from file/stream
        void InitInterpData(G4String dataFile = "TempData.txt",
							G4double uy = 1., G4double ux = 1.);
        void InitInterpData(std::ifstream&, G4double uy = 1., G4double ux = 1.);

        // Initialize interpolation data from vector of data
        void InitInterpData(DataPointVec, G4InterpolationScheme);

        // Initialize interpolation manager
        void InitInterpManager(G4InterpolationScheme, G4int);

        // Add data point to vector
        void AddDataPoint(DataPoint);

        // Check if vector is not empty
        G4bool HasData() const { return hasData; };

        // Get interpolated value
        G4double GetY(G4double xIn) const;

        // Integrate over entire vector
        G4double Integrate();


    private:
        // Private member functions

		void CopyVector(const StorkInterpVector &right);
        void InitDataVector(std::ifstream&, G4double uy = 1., G4double ux = 1.);

    private:
        // Private member variables

        G4InterpolationManager theInterpMan;    // Interpolation manager
        G4NeutronHPInterpolator theInt;         // Interpolator
        DataPointVec data;                      // Vector of data
        G4int numPoints;                        // Number of data points
        G4bool hasData;
};

#endif // NSINTERPVECTOR_H
