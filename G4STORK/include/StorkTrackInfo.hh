/*
StorkTrackInfo.hh

Created by:		Liam Russell
Date:			12-03-2012
Modified:       11-03-2013

Implementation of StorkTrackInfo class.

This class is used to attach a particle weight (for combing) to a track
by deriving it from the G4VUserTrackInformation class. The track information
also is used to StorkPrimaryNeutronInfo for neutrons created by the
StorkUserBCStepLimiter process.

*/

#ifndef STORKTRACKINFO_H
#define STORKTRACKINFO_H

#include "G4VUserTrackInformation.hh"
#include "StorkPrimaryNeutronInfo.hh"


class StorkTrackInfo : public G4VUserTrackInformation
{
    public:
        // Public member functions

        // Constructor and destructor
        StorkTrackInfo(G4double aNum = 1.0)
        {
        	weight = aNum;
        	pnInfo = NULL;
		}
        virtual ~StorkTrackInfo()
        {
        	if(pnInfo) delete pnInfo;
		}

        // Get and set weight
        void SetWeight(G4double pWeight) { weight = pWeight; };
        G4double GetWeight() { return weight; };

        // Get and set primary neutron info
        void SetStorkPrimaryNeutronInfo(StorkPrimaryNeutronInfo *pnInfoPtr)
        {
        	pnInfo = pnInfoPtr;
		}
		StorkPrimaryNeutronInfo* GetStorkPrimaryNeutronInfo() const
		{
		    return pnInfo;
        }


    private:
        // Public member variables

        G4double weight;
        StorkPrimaryNeutronInfo *pnInfo;
};

#endif // STORKTRACKINFO_H
