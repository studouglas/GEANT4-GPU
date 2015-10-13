/*
StorkInterpVector.cc

Created by:		Wesley Ford
Date:			24-06-2012
Modified:       09-07-2012

Source code for StorkInterpVector.cc.

*/

#include "StorkInterpVector.hh"


// Default constructor
StorkInterpVector::StorkInterpVector()
{
    // Flags that no data has been given to interpolate from
    hasData = false;
    numPoints = 0;
}


// Constructs data vector from input data file
StorkInterpVector::StorkInterpVector(G4String dataFile)
{
    InitInterpData(dataFile);
}


// Copy constructor
StorkInterpVector::StorkInterpVector(const StorkInterpVector& right)
{
	CopyVector(right);
}


// Operator =
// Overloaded equality operator
StorkInterpVector& StorkInterpVector::operator=(const StorkInterpVector& right)
{
	// Make sure "this" is NOT "right"
	if(&right != this)
	{
		CopyVector(right);
	}

	return *this;
}


// CopyVector()
// Helper function for copy constructor and equality operator
void StorkInterpVector::CopyVector(const StorkInterpVector& right)
{
	theInterpMan = right.theInterpMan;
	theInt = right.theInt;
	data = right.data;
	numPoints = right.numPoints;
	hasData = right.hasData;
}


//InitInterpData( data file )
//sets the data from file and the scheme to be used
void StorkInterpVector::InitInterpData(G4String dataFile, G4double uy,
                                       G4double ux)
{
    // opens the data file and determines if it contains data
    std::ifstream inFile(dataFile);

    if(!inFile.good())
    {
        G4cerr << "*** Error: Unable to open interpolation data file "
               << dataFile << G4endl;

        return;
    }

    InitInterpData(inFile,uy,ux);
}


//InitInterpData( data stream )
//sets the data from stream and the scheme to be used
void StorkInterpVector::InitInterpData(std::ifstream &file, G4double uy,
                                       G4double ux)
{
    //initiates interp manager based of the given data file
    theInterpMan.Init(file);
    InitDataVector(file,uy,ux);

    if(!data.empty())
        hasData = true;
}


//InitInterpData( data vector )
//sets the data  from vector and the scheme to be used
void StorkInterpVector::InitInterpData(DataPointVec data2,
                                       G4InterpolationScheme scheme)
{
    //initiates interp manager based of the given data vector
    numPoints = G4int(data2.size());

    theInterpMan.Init(scheme, 1);
    data=data2;

    if(!data.empty())
        hasData = true;
}


//InitInterpManager()
//sets the range and the scheme to be used by the interpmanager
void StorkInterpVector::InitInterpManager(G4InterpolationScheme aScheme,
                                          G4int aRange)
{
    theInterpMan.Init(aScheme, aRange);
}


// InitDataVector()
// Private helper function to set data in DataPointVec from a file stream
void StorkInterpVector::InitDataVector(std::ifstream &file, G4double uy,
                                       G4double ux)
{
    // Local variables
    G4double x, y;
    DataPoint point;

    // Get total number of points in the file
    file >> numPoints;

    // Read in the points and send the pair to the data vector
    for(G4int i=0; i<numPoints; i++)
    {
        file >> x >> y;

        x *= ux;
        y *= uy;

        point = std::make_pair(x,y);
        data.push_back(point);
    }
}


// AddDataPoint()
// Adds a data point on to the data vector
void StorkInterpVector::AddDataPoint(DataPoint point)
{
    DataPointVec::iterator itr = data.begin();

    // Find the proper place to insert data point
    for(; itr != data.end(); itr++)
    {
        if(point.first > itr->first)
            break;
    }

    // Add data point and increment the number of points
    data.insert(itr,point);
    numPoints++;
}


// GetY()
// Find the interpolated y-value relative to a given x-value
G4double StorkInterpVector::GetY(G4double xIn) const
{
    // Check if the StorkInterpVector has data
    if(!hasData) return DBL_MIN;

    // Local variables
    G4int i=1;
    G4InterpolationScheme aScheme = LINLIN;

    // Check if the given x-value is outside the data range
    // If so return the first/last y-value respectively
    if(xIn < data[0].first)
    {
        return data[0].second;
    }
    else if(xIn >= data[numPoints-1].first)
    {
        return data[numPoints-1].second;
    }

    // Find the upper bound of the given x-value
    for(; i<numPoints; i++)
    {
        if(xIn<(data[i]).first)
        {
            break;
        }
    }

    // Get the scheme for the upper point from the interpolation manager
    aScheme = theInterpMan.GetScheme(i);

    // Return the interpolated value based on the given scheme
    return theInt.Interpolate(aScheme,xIn,data[i-1].first,data[i].first,
                              data[i-1].second, data[i].second);
}

//Integrate()
//integrates the given data set
G4double StorkInterpVector::Integrate()
{
    //Determines the type of integration to be done
    G4InterpolationScheme aScheme = theInterpMan.GetScheme(0);
    G4double sum=0;

    //integrates over all data points using the set scheme
    if(aScheme==LINLIN)
    {
        G4int n= data.size();
        for(G4int i=0;i<n-1;i++)
        {
            sum += (((data[i+1]).second-(data[i]).second)*0.5 +(data[i]).second)
                        * ((data[i+1]).first-(data[i]).first);
        }
    }

    return sum;
}

