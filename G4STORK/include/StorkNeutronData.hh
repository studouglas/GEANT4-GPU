/*
StorkNeutronData.hh

Created by:		Liam Russell
Date:			07-08-2011
Modified:		11-03-2013

Definition for StorkNeutronData class.

This is the basic container class used to store the survivor and delayed neutron
data. The class holds four pieces of data publicly:
    1. Global Time = the current simulation time the other data was recorded at
    2. Lifetime = the total simulation time since the hit that created the
                    neutron. For delayed neutrons, this is the initial fission
                    that set off the decay chain.
    3. Position = the position of the neutron at the global time.
    4. Momentum = the momentum of the neutron at the global time.
    5-8. Eta's = theNumberOfInteractionLengthLeft for the hadronic processes
    9. Discretionary (energy/weight/etc)
This class's main purpose is to be marshallable, and thereby, allow the
NeutronSources and NeutronList classes to be marshallable. Also, the member
variables are public to allow easy access.

*/

#ifndef NEUTRONDATA_H
#define NEUTRONDATA_H

#include "G4ThreeVector.hh"


//MSH_BEGIN
class StorkNeutronData
{
    public:
        // Public member functions

        // Default constructor
        StorkNeutronData()
        { }

        // Constructor with first 4 values
        StorkNeutronData(G4double in1, G4double in2, G4ThreeVector in3,
                    G4ThreeVector in4)
        :first(in1),second(in2),third(in3),fourth(in4),fifth(-1.0),sixth(-1.0),
            seventh(-1.0),eigth(-1.0),ninth(1.0)
        { }

        // Constructor with all nine values
        StorkNeutronData(G4double in1, G4double in2, G4ThreeVector in3,
                    G4ThreeVector in4, G4double in5, G4double in6,
                    G4double in7, G4double in8, G4double in9)
        :first(in1),second(in2),third(in3),fourth(in4),fifth(in5),sixth(in6),
            seventh(in7),eigth(in8),ninth(in9)
        { }

        // Assignment operator
        StorkNeutronData& operator=(const StorkNeutronData& other)
        {
            copy(other);
            return *this;
        }

        // Destructor
        virtual ~StorkNeutronData() { }

    public:
        // Public member data

        G4double first;  //MSH: primitive
        G4double second;  //MSH: primitive
        G4ThreeVector third;  //MSH: primitive
        G4ThreeVector fourth;  //MSH: primitive
        G4double fifth; //MSH: primitive
        G4double sixth; //MSH: primitive
        G4double seventh; //MSH: primitive
        G4double eigth; //MSH: primitive
        G4double ninth; //MSH: primitive


    private:
        // Private member functions

        // Copy helper function
        void copy(const StorkNeutronData & other)
        {
            first = other.first;
            second = other.second;
            third = other.third;
            fourth = other.fourth;
            fifth = other.fifth;
            sixth = other.sixth;
            seventh = other.seventh;
            eigth = other.eigth;
            ninth = other.ninth;
        }
};
//MSH_END

#endif // NEUTRONDATA_H
