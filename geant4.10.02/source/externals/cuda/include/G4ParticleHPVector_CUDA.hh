#ifndef G4ParticleHPVector_CUDA_h
#define G4ParticleHPVector_CUDA_h 1

#include <stdio.h>
#include <algorithm> // for std::max
#include <cfloat>
#include "G4ParticleHPDataPoint_CUDA.hh"
#include "G4ParticleHPInterpolator_CUDA.hh"
#include "G4InterpolationScheme_CUDA.hh"
#include "G4InterpolationManager_CUDA.hh"
#include "G4Types_CUDA.hh"

class G4ParticleHPVector_CUDA {
    
    /******************************************
    * CONSTRUCTORS / DECONSTRUCTORS
    *******************************************/
    public:
    G4ParticleHPVector_CUDA();
    G4ParticleHPVector_CUDA(G4int);
    ~G4ParticleHPVector_CUDA();

    //G4ParticleHPVector & operator = (const G4ParticleHPVector & right);
    G4double GetXsec(G4double e);
    void Dump();
    void ThinOut(G4double precision);
    void Merge(G4InterpolationScheme aScheme, G4double aValue, G4ParticleHPVector_CUDA * active, G4ParticleHPVector_CUDA * passive);
    G4double Sample();
    G4double Get15percentBorder();
    G4double Get50percentBorder();

    void Check(G4int i);
    G4bool IsBlocked(G4double aX);

    /******************************************
    * Getters from .hh
    *******************************************/
    inline const G4ParticleHPDataPoint & GetPoint(G4int i) const {
        return theData[i];
    }
    inline G4int GetVectorLength() const {
        return 0;
    }
    inline G4double GetEnergy(G4int i) const {
        return 0;
    }
    inline G4double GetX(G4int i) const {
        return 0;
    }
    inline G4double GetXsec(G4int i) {
        return 0;
    }
    G4double GetXsec(G4double e, G4int min) {
        return 0;
    }
    inline G4double GetY(G4double x) {
        return 0;
    }
    inline G4double GetY(G4int i) {
        return 0;
    }
    inline G4double GetY(G4int i) const {
        return 0;
    }
    G4double GetMeanX() {
        return 0;
    }
    inline G4double GetLabel() {
        return 0;
    }
    inline G4double GetIntegral() {
        return 0;
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

    }
    inline void SetPoint(G4int i, const G4ParticleHPDataPoint & it) {

    }
    inline void SetData(G4int i, G4double x, G4double y) {

    }
    inline void SetX(G4int i, G4double e) {

    }
    inline void SetEnergy(G4int i, G4double e) {

    }
    inline void SetY(G4int i, G4double x) {

    }
    inline void SetXsec(G4int i, G4double x) {

    }
    inline void SetLabel(G4double aLabel) {

    }
    inline void SetInterpolationManager(const G4InterpolationManager & aManager) {

    }
    inline void SetInterpolationManager(G4InterpolationManager & aMan) {

    }
    inline void SetScheme(G4int aPoint, const G4InterpolationScheme & aScheme) {

    }


    /******************************************
    * Computations from .hh
    ******************************************/
    void Init(std::istream & aDataFile, G4int total, G4double ux=1., G4double uy=1.) {

    }
    void Init(std::istream & aDataFile,G4double ux=1., G4double uy=1.) {

    }
    inline void InitInterpolation(std::istream & aDataFile) {

    }
    inline void CleanUp() {

    }
    inline void Merge(G4ParticleHPVector_CUDA * active, G4ParticleHPVector_CUDA * passive) {

    }
    G4double SampleLin() {
        return 0;
    }
    G4double * Debug() {
        return 0;
    }
    inline void Integrate() {

    }
    inline void IntegrateAndNormalise() {

    }
    inline void Times(G4double factor) {

    }

    /******************************************
    * PRIVATE                                 
    *******************************************/
    private:
    G4ParticleHPInterpolator theLin;
    G4double totalIntegral;
    
    G4ParticleHPDataPoint * theData;
    G4InterpolationManager theManager;
    G4double * theIntegral;
    G4int nEntries;
    G4int nPoints;
    G4double label;
    
    G4ParticleHPInterpolator theInt;
    G4int Verbose;
    G4int isFreed;
    
    G4double maxValue;
    // std::vector<G4double> theBlocked;

    G4double the15percentBorderCash;
    G4double the50percentBorderCash;
};

#endif
