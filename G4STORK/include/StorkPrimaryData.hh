/*
StorkPrimaryData.hh

Created by:		Liam Russell
Date:			07-08-2011
Modified:		11-03-2013

Definition for StorkPrimaryData class.

This is a marshallable container class that is used to pass source information
for an event from the master to a slave. The following information is stored

    1. The primary neutrons of an event.
    2. The random seed of the event.
    3. The index number of the event.
    4. Any material-property changes that need to be implemented on the slave
            at the beginning of the event.

*/

#ifndef STORKPRIMARYDATA_H
#define STORKPRIMARYDATA_H

#include "StorkContainers.hh"


//MSH_include_begin
#include "MarshaledStorkNeutronData.h"
#include "MarshaledStorkMatPropChange.h"
//MSH_include_end
//MSH_BEGIN
class StorkPrimaryData
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkPrimaryData()
        {
            primaries = new NeutronSources();
            propChanges = new StorkMatPropChangeVector;
        }
        virtual ~StorkPrimaryData()
        {
            if(primaries) delete primaries;
            if(propChanges) delete propChanges;

            primaries = NULL;
            propChanges = NULL;
        }

    public:
        // Public member variables

        NeutronSources *primaries;  /*MSH: ptr_as_array
        [elementType: StorkNeutronData]
        [elementCount: { $ELE_COUNT = $THIS->primaries->size(); }]
        [elementGet: { $ELEMENT = (*($THIS->primaries))[$ELE_INDEX]; }]
        [elementSet: { $THIS->primaries->push_back(*$ELEMENT); }]
        */

        G4long eventSeed; //MSH: primitive
        G4int eventNum;  //MSH: primitive
        StorkMatPropChangeVector *propChanges; /*MSH: ptr_as_array
        [elementType: StorkMatPropChange]
        [elementCount: { $ELE_COUNT = $THIS->propChanges->size(); }]
        [elementGet: { $ELEMENT = (*($THIS->propChanges))[$ELE_INDEX]; }]
        [elementSet: { $THIS->propChanges->push_back(*$ELEMENT); }]
        */
};
//MSH_END

#endif // STORKPRIMARYDATA_H
