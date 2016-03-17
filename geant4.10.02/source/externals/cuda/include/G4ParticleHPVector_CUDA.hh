#ifndef G4ParticleHPVector_CUDA_h
#define G4ParticleHPVector_CUDA_h 1

#include <stdio.h>
#include <iostream>
#include <algorithm> // for std::max
#include <cfloat>
#include "G4ParticleHPDataPoint_CUDA.hh"
#include "G4ParticleHPInterpolator_CUDA.hh"
#include "G4InterpolationScheme_CUDA.hh"
#include "G4InterpolationManager_CUDA.hh"
#include "G4Types_CUDA.hh"
#include "G4Pow_CUDA.hh"

#define THREADS_PER_BLOCK 32 // must be multiple of 32

typedef struct GetXsecResultStruct {
    G4double y; // if -1, other 3 are non-null
    G4ParticleHPDataPoint pointLow;
    G4ParticleHPDataPoint pointHigh;
    G4int indexHigh;
} GetXsecResultStruct;

class G4ParticleHPVector_CUDA {

    /******************************************
    * CONSTRUCTORS / DECONSTRUCTORS
    *******************************************/
    public:
	
	G4int nEntries;
    G4ParticleHPVector_CUDA();
    G4ParticleHPVector_CUDA(G4int);
    void PerformInitialization(G4int);
    ~G4ParticleHPVector_CUDA();

    G4double GetXsec(G4double e);
	void GetXsecBuffer(G4double * queryList, G4int length);
    void Dump();
    void ThinOut(G4double precision);
    void Merge(G4InterpolationScheme aScheme, G4double aValue, G4ParticleHPVector_CUDA * active, G4ParticleHPVector_CUDA * passive);
    G4double Sample();
    G4double Get15percentBorder();
    G4double Get50percentBorder();

    void OperatorEquals (G4ParticleHPVector_CUDA* right);

    void Check(G4int i);
    G4bool IsBlocked(G4double aX);

    /******************************************
    * Getters from .hh
    *******************************************/
    G4ParticleHPDataPoint & GetPoint(G4int i);

    inline G4int GetVectorLength() const {
        return nEntries;
    }

    G4double GetX(G4int i);
    G4double GetXsec(G4double e, G4int min);
    G4double GetY(G4double x);
    G4double GetY(G4int i);
    G4double GetMeanX();

    inline G4double GetLabel() {
        return label;
    }

    inline G4double GetIntegral() {
        if (totalIntegral < -0.5) {
            Integrate();
        }
        return totalIntegral;
    }

    inline const G4InterpolationManager & GetInterpolationManager() const {
        return theManager;
    }

    inline G4InterpolationScheme GetScheme(G4int anIndex) {
        return theManager.GetScheme(anIndex);
    }


    /******************************************
    * Setters from .hh
    ******************************************/
    inline void SetVerbose(G4int ff) {
        Verbose = ff;
    }

    inline void SetPoint(G4int i, const G4ParticleHPDataPoint & it) {
        G4double x = it.GetX();
        G4double y = it.GetY();
        SetData(i,x,y);
    }

    void SetData(G4int i, G4double x, G4double y);
    void SetX(G4int i, G4double e);
    void SetEnergy(G4int i, G4double e);
    void SetY(G4int i, G4double x);
    void SetXsec(G4int i, G4double x);

    inline void SetLabel(G4double aLabel) {
        label = aLabel;
    }

    inline void SetInterpolationManager(const G4InterpolationManager & aManager) {
        theManager = aManager;
    }

    inline void SetInterpolationManager(G4InterpolationManager & aMan) {
        theManager = aMan;
    }

    inline void SetScheme(G4int aPoint, const G4InterpolationScheme & aScheme) {
        theManager.AppendScheme(aPoint, aScheme);
    }


    /******************************************
    * Computations from .hh
    ******************************************/
    void Init(std::istream & aDataFile, G4int total, G4double ux=1., G4double uy=1.);
    void Init(std::istream & aDataFile, G4double ux=1., G4double uy=1.);

    inline void InitInterpolation(std::istream & aDataFile) {
        theManager.Init(aDataFile);
    }

    void CleanUp();

    inline void Merge(G4ParticleHPVector_CUDA * active, G4ParticleHPVector_CUDA * passive) {
        printf("\nMERGE CALLED");
        CleanUp();
        G4int s_tmp = 0;
        G4int n = 0;
        G4int m_tmp = 0;

        G4ParticleHPVector_CUDA * tmp;
        G4int a = s_tmp;
        G4int p = n;
        G4int t;

        while (a < active->GetVectorLength() && p < passive->GetVectorLength()) {
            if (active->GetX(a) <= passive->GetX(p)) {
                G4double xa = active->GetX(a);
                G4double yy = active->GetY(a);
                SetData(m_tmp, xa, yy);
                theManager.AppendScheme(m_tmp, active->GetScheme(a));
                m_tmp++;
                a++;
                G4double xp = passive->GetX(p);

                if (!(xa == 0) && std::abs(std::abs(xp - xa) / xa) < 0.001) {
                    p++;
                }
            } else {
                tmp = active;
                t = a;
                active = passive;
                a = p;
                passive = tmp;
                p = t;
            }
        }
        while (a != active->GetVectorLength()) {
            SetData(m_tmp, active->GetX(a), active->GetY(a));
            theManager.AppendScheme(m_tmp, active->GetScheme(a));
            m_tmp++;
            a++;
        }
        while (p != passive->GetVectorLength()) {
            if (std::abs(GetX(m_tmp - 1) - passive->GetX(p)) / passive->GetX(p) > 0.001) {
                SetData(m_tmp, passive->GetX(p), passive->GetY(p));
                theManager.AppendScheme(m_tmp, active->GetScheme(p));
                m_tmp++;
            }
            p++;
        }
    }

    G4double SampleLin();
    G4double * Debug();
    void Integrate();
    void IntegrateAndNormalise();
    void Times(G4double factor);

    /******************************************
    * PRIVATE
    *******************************************/
    private:
    G4double GetUniformRand();
    void CopyToCpuIfDirty();
    inline int GetNumBlocks(int totalNumThreads) {
        if (totalNumThreads == 0) {
            return 0;
        }
        return (totalNumThreads / THREADS_PER_BLOCK + ((totalNumThreads % THREADS_PER_BLOCK == 0) ? 0 : 1));
    }

    G4ParticleHPInterpolator theLin;
    G4double totalIntegral;

    GetXsecResultStruct * h_res;
    GetXsecResultStruct * d_res;
    G4int *h_singleIntResult;
    G4int *d_singleIntResult;
    G4double *h_singleDoubleResult;
    G4double *d_singleDoubleResult;

    G4ParticleHPDataPoint * d_theData;
    G4ParticleHPDataPoint * h_theData;
    G4bool isDataDirtyHost;
    G4bool isIntegralDirtyHost;
    G4int nPointsHost;

    G4double * h_theIntegral;
    G4double * d_theIntegral;
    G4InterpolationManager theManager;

    G4int nPoints;
    G4double label;

    G4ParticleHPInterpolator theInt;
    G4int Verbose;
    G4int isFreed;

    G4double maxValue;

    G4double the15percentBorderCash;
    G4double the50percentBorderCash;
};

#endif
