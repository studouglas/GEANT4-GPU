#ifndef STORKPERIODICBCTRANSFORM_HH
#define STORKPERIODICBCTRANSFORM_HH

#include "StorkBCTransform.hh"

class StorkPeriodicBCTransform: public StorkBCTransform
{
    public:
        StorkPeriodicBCTransform();
        StorkPeriodicBCTransform(G4ThreeVector n1, G4ThreeVector n2);
        virtual ~StorkPeriodicBCTransform();
        void InitializeTransform(G4ThreeVector n1, G4ThreeVector n2);
    protected:
    private:
};

#endif // STORKPERIODICBCTRANSFORM_HH
