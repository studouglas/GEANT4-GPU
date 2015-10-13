
/*
ZED2Constructor.hh

Created by:		Salma Mahzooni
Date:			26-07-2013
Modified:       11-03-2013

Header for C6LatticeConstructor class.

This class creates the simulation geometry for a CANDU 6 lattice cell.  The
material and geometric composition of the lattice cell were taken from the
DRAGON manual[1].

Based on the "C6World" class created by Wesley Ford 10-05-2012.

[1] G. Marleau, A. Hebert, and R. Roy, "A User Guide for DRAGON 3.06".  Ecole
Polytechnique de Montreal, 2012, pp. 148-152. IGE-174 Rev. 10.

*/

#ifndef ZED2Constructor_H
#define ZED2Constructor_H

// Include header files
#include "StorkNeutronSD.hh"
#include "StorkVWorldConstructor.hh"
#include "G4UnionSolid.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Orb.hh"
#include "G4SubtractionSolid.hh"



class ZED2Constructor : public StorkVWorldConstructor
{
	public:
        // Public memeber functions

		// Constructor and destructor
		ZED2Constructor();
		virtual ~ZED2Constructor();
		virtual G4VPhysicalVolume* ConstructNewWorld(const StorkParseInput* infile);


    protected:
        // Protected member functions

		// Material name enumerator
		enum WorldMats
		{ e_U235=0, e_Godiva, e_UHW, NUM_MATERIALS };

        // Private member functions
        virtual G4VPhysicalVolume* ConstructWorld();
        virtual void ConstructMaterials();

    protected:
        // Protected member variables

        // Logical Volumes
    G4LogicalVolume *worldLogical;
    G4LogicalVolume *reactDimTubeLogical;
    G4LogicalVolume *airTubeLogical;
	G4LogicalVolume *vesselLogical;
    G4LogicalVolume *tankLogical1;
    G4LogicalVolume *ModLogical;
	G4LogicalVolume *logicCalandria1;
    G4LogicalVolume *logicGasAnn1;
	G4LogicalVolume *logicPressure1;
    G4LogicalVolume *logicCoolant1;
    G4LogicalVolume *logicAir1;
	G4LogicalVolume *logicRodA1;
	G4LogicalVolume *logicRodB1;
	G4LogicalVolume *logicSheathA1;
	G4LogicalVolume *logicSheathB1;
    G4LogicalVolume *logicEndPlate2;
    G4LogicalVolume *logicEndPlate1;
    G4LogicalVolume *logicCalandria1Mod;
    G4LogicalVolume *logicGasAnn1Mod;
	G4LogicalVolume *logicPressure1Mod;
	G4LogicalVolume *logicCoolant1Mod;
    G4LogicalVolume *logicRodA1Cut2;
	G4LogicalVolume *logicRodB1Cut2;
	G4LogicalVolume *logicSheathA1Cut2;
	G4LogicalVolume *logicSheathB1Cut2;
    G4LogicalVolume *logicEndPlate2Cut2;
    G4LogicalVolume *logicRodA1Mod;
	G4LogicalVolume *logicRodB1Mod;
	G4LogicalVolume *logicSheathA1Mod;
	G4LogicalVolume *logicSheathB1Mod;
    G4LogicalVolume *logicEndPlate2Mod;
    G4LogicalVolume *logicEndPlate1Mod;
    G4LogicalVolume *logicRodA1Cut1;
	G4LogicalVolume *logicRodB1Cut1;
	G4LogicalVolume *logicSheathA1Cut1;
	G4LogicalVolume *logicSheathB1Cut1;
    G4LogicalVolume *logicEndPlate2Cut1;
    G4LogicalVolume *logicCalandria1RU;
    G4LogicalVolume *logicGasAnn1RU;
	G4LogicalVolume *logicPressure1RU;
    G4LogicalVolume *logicCoolant1RU;
    G4LogicalVolume *logicAir1RU;
	G4LogicalVolume *logicRodA1RU;
	G4LogicalVolume *logicRodB1RU;
	G4LogicalVolume *logicSheathA1RU;
	G4LogicalVolume *logicSheathB1RU;
    G4LogicalVolume *logicEndPlate2RU;
    G4LogicalVolume *logicEndPlate1RU;
    G4LogicalVolume *logicCalandria1ModRU;
    G4LogicalVolume *logicGasAnn1ModRU;
	G4LogicalVolume *logicPressure1ModRU;
	G4LogicalVolume *logicCoolant1ModRU;
    G4LogicalVolume *logicRodA1Cut2RU;
	G4LogicalVolume *logicRodB1Cut2RU;
	G4LogicalVolume *logicSheathA1Cut2RU;
	G4LogicalVolume *logicSheathB1Cut2RU;
    G4LogicalVolume *logicEndPlate2Cut2RU;
    G4LogicalVolume *logicRodA1ModRU;
	G4LogicalVolume *logicRodB1ModRU;
	G4LogicalVolume *logicSheathA1ModRU;
	G4LogicalVolume *logicSheathB1ModRU;
    G4LogicalVolume *logicEndPlate2ModRU;
    G4LogicalVolume *logicEndPlate1ModRU;
    G4LogicalVolume *logicRodA1Cut1RU;
	G4LogicalVolume *logicRodB1Cut1RU;
	G4LogicalVolume *logicSheathA1Cut1RU;
	G4LogicalVolume *logicSheathB1Cut1RU;
    G4LogicalVolume *logicEndPlate2Cut1RU;
    G4LogicalVolume *logicDumplineAl;
    G4LogicalVolume *logicDumplineHW;
    G4LogicalVolume *logicDumplineAlC;
    G4LogicalVolume *logicDumplineHWC;



        // Visualization attributes

        G4VisAttributes * vesselVisAtt;
        G4VisAttributes * tank1VisATT;
        G4VisAttributes * ModVisAtt;
        G4VisAttributes * fuelA1VisATT;
        G4VisAttributes * fuelB1VisATT;
        G4VisAttributes * sheathA1VisATT;
        G4VisAttributes * sheathB1VisATT;
        G4VisAttributes * Air1VisAtt;
        G4VisAttributes * Coolant1VisAtt;
        G4VisAttributes * Pressure1VisAtt;
        G4VisAttributes * GasAnn1VisAtt;
        G4VisAttributes * Calandria1VisAtt;
        G4VisAttributes * EndPlate2VisATT;
        G4VisAttributes * airTubeVisAtt;
        G4VisAttributes * DumplineAlVisAtt;
        G4VisAttributes * DumplineHWVisAtt;
        G4VisAttributes * reactDimTubeVisAtt;




        // Multifunctional Detector
        //G4MultiFunctionalDetector *FluxScorer;
        // Stored variables from infile
        G4String materialID;
        G4double matTemp;
        G4double matDensity[NUM_MATERIALS];
        G4double reactorRadius;
        G4double fuelTemp;
        G4double fuelDensity;
        G4double AirDensity;
        G4double moderatorTemp;
        G4double moderatorDensity;
        G4double coolantDensity;
        G4double coolantTemp;
};


#endif // ZED2Constructor_H
