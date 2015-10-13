/*
StorkEventData.hh

Created by:		Liam Russell
Date:			07-08-2011
Modified:		17-02-2012

Definition of StorkEventData class.

This is a container class designed to be passed from slaves to the master
when they have finished simulating an event.  All the data is public and may
be accessed directly.  It also includes comments so that Marshalgen (automated
marshalling software) may create a marshaled class definition.

*/


#ifndef STORKEVENTDATA_HH_INCLUDED
#define STORKEVENTDATA_HH_INCLUDED

// Include header files and libraries
#include "StorkContainers.hh"


//MSH_include_begin
#include "MarshaledStorkNeutronData.h"
#include "MarshaledStorkTripleFloat.h"
//MSH_include_end
//MSH_BEGIN
class StorkEventData
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkEventData()
        {
            survivors = new NeutronSources();
            delayed = new NeutronSources();
            fSites = new MSHSiteVector();
            fnEnergy = new DblVector();
            totalLifetime = 0.0;
            numNLost = numNProd = numDProd = 0;
            eventNum = -1;
        }
        virtual ~StorkEventData()
        {
            if(survivors) delete survivors;
            if(delayed) delete delayed;
            if(fSites) delete fSites;
            if(fnEnergy) delete fnEnergy;

            survivors = NULL;
            delayed = NULL;
            fSites = NULL;
            fnEnergy = NULL;
        }

        // Combine two StorkEventData objects from the same event
        void AddEventData(const StorkEventData &right)
        {
        	// Check if this data is not from the same event
        	if(eventNum != right.eventNum)
        	{
        		G4cerr << "*** WARNING: Tried to combine two different events."
        		       << G4endl;
				return;
        	}

			// Add input event data to
        	totalLifetime += right.totalLifetime;
        	numNLost += right.numNLost;
        	numNProd += right.numNProd;
        	survivors->insert(survivors->end(),right.survivors->begin(),
							  right.survivors->end());
			delayed->insert(delayed->end(),right.delayed->begin(),
				  right.delayed->end());
			fSites->insert(fSites->end(),right.fSites->begin(),
				  right.fSites->end());

        }

        // Combine StorkEventData objects using '+' operator
        StorkEventData operator + (const StorkEventData &right)
		{
			StorkEventData temp;
			temp.AddEventData(*this);
			temp.AddEventData(right);

			return temp;
		}

        // Combine StorkEventData using the '+=' operator
		StorkEventData& operator += (const StorkEventData &right)
		{
			this->AddEventData(right);

			return *this;
		}


    public:
        // Public member data with marshalling instructions

        NeutronSources *survivors;  /*MSH: ptr_as_array
        [elementType: StorkNeutronData]
        [elementCount: { $ELE_COUNT = $THIS->survivors->size(); }]
        [elementGet: { $ELEMENT = $THIS->survivors->at($ELE_INDEX); }]
        [elementSet: { $THIS->survivors->push_back(*$ELEMENT); }]
        */

        NeutronSources *delayed;  /*MSH: ptr_as_array
        [elementType: StorkNeutronData]
        [elementCount: { $ELE_COUNT = $THIS->delayed->size(); }]
        [elementGet: { $ELEMENT = $THIS->delayed->at($ELE_INDEX); }]
        [elementSet: { $THIS->delayed->push_back(*$ELEMENT); }]
        */

        MSHSiteVector *fSites;  /*MSH: ptr_as_array
        [elementType: StorkTripleFloat]
        [elementCount: { $ELE_COUNT = $THIS->fSites->size(); }]
        [elementGet: { $ELEMENT = $THIS->fSites->at($ELE_INDEX); }]
        [elementSet: { $THIS->fSites->push_back(*$ELEMENT); }]
        */

        DblVector *fnEnergy;  /*MSH: ptr_as_array
        [elementType: double]
        [elementCount: { $ELE_COUNT = $THIS->fnEnergy->size(); }]
        [elementGet: { $ELEMENT = $THIS->fnEnergy->at($ELE_INDEX); }]
        [elementSet: { $THIS->fnEnergy->push_back($ELEMENT); }]
        */

        G4double totalLifetime;  //MSH: primitive
        G4int numNProd;  //MSH: primitive
        G4int numNLost;  //MSH: primitive
        G4int numDProd;  //MSH: primitive
        G4int eventNum;  //MSH: primitive
};
//MSH_END

#endif // STORKEVENTDATA_HH_INCLUDED
