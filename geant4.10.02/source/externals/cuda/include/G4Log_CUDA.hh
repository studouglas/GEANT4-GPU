//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// $Id:$
//
//
// --------------------------------------------------------------------
//
// Class Description:
//
//
// The basic idea is to exploit Pade polynomials.
// A lot of ideas were inspired by the cephes math library
// (by Stephen L. Moshier moshier@na-net.ornl.gov) as well as actual code. 
// The Cephes library can be found here:  http://www.netlib.org/cephes/
// Code and algorithms for G4Exp have been extracted and adapted for Geant4
// from the original implementation in the VDT mathematical library
// (https://svnweb.cern.ch/trac/vdt), version 0.3.7.

// Original implementation created on: Jun 23, 2012
//      Author: Danilo Piparo, Thomas Hauth, Vincenzo Innocente
//
// --------------------------------------------------------------------
/* 
 * VDT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
// --------------------------------------------------------------------
#ifndef G4Log_h
#define G4Log_h 1

#ifdef WIN32

  #define G4Log std::log

#else

#include <limits>
#include <stdint.h>
 #include <cuda.h>
 #include <cuda_runtime.h>
// #include "G4Types.hh"

// local namespace for the constants/functions which are necessary only here
//
namespace G4LogConsts
{
  const double LOG_UPPER_LIMIT = 1e307;
  const double LOG_LOWER_LIMIT = 0;

  const double SQRTH = 0.70710678118654752440;
  const float MAXNUMF = 3.4028234663852885981170418348451692544e38f;

  //----------------------------------------------------------------------------
  // Used to switch between different type of interpretations of the data
  // (64 bits)
  //
  union ieee754
  {
	ieee754 () {};
	ieee754 (double thed) {d=thed;};
	ieee754 (uint64_t thell) {ll=thell;};
	ieee754 (float thef) {f[0]=thef;};
	ieee754 (uint32_t thei) {i[0]=thei;};
    double d;
    float f[2];
    uint32_t i[2];
    uint64_t ll;
    uint16_t s[4];
  };

  __host__ __device__ inline double get_log_px(const double x)
  {
        const double PX1log = 1.01875663804580931796E-4;
        const double PX2log = 4.97494994976747001425E-1;
        const double PX3log = 4.70579119878881725854E0;
        const double PX4log = 1.44989225341610930846E1;
        const double PX5log = 1.79368678507819816313E1;
        const double PX6log = 7.70838733755885391666E0;

        double px = PX1log;
        px *= x;
        px += PX2log;
        px *= x;
        px += PX3log;
        px *= x;
        px += PX4log;
        px *= x;
        px += PX5log;
        px *= x;
        px += PX6log;
        return px;
  }

  __host__ __device__ inline double get_log_qx(const double x)
  {
        const double QX1log = 1.12873587189167450590E1;
        const double QX2log = 4.52279145837532221105E1;
        const double QX3log = 8.29875266912776603211E1;
        const double QX4log = 7.11544750618563894466E1;
        const double QX5log = 2.31251620126765340583E1;

        double qx = x;
        qx += QX1log;
        qx *=x;
        qx += QX2log;
        qx *=x;
        qx += QX3log;
        qx *=x;
        qx += QX4log;
        qx *=x;
        qx += QX5log;
        return qx;
  }

  //----------------------------------------------------------------------------
  // Converts a double to an unsigned long long
  //
  __host__ __device__ inline uint64_t dp2uint64(double x)
  {
    ieee754 tmp;
    tmp.d=x;
    return tmp.ll;
  }

  //----------------------------------------------------------------------------
  // Converts an unsigned long long to a double
  //
  __host__ __device__ inline double uint642dp(uint64_t ll)
  {
    ieee754 tmp;
    tmp.ll=ll;
    return tmp.d;
  }

  //----------------------------------------------------------------------------
  // Converts an int to a float
  //
  __host__ __device__ inline float uint322sp(int x)
  {
    ieee754 tmp;
    tmp.i[0]=x;
    return tmp.f[0];
  }

  //----------------------------------------------------------------------------
  // Converts a float to an int
  //
  __host__ __device__ inline uint32_t sp2uint32(float x)
  {
    ieee754 tmp;
    tmp.f[0]=x;
    return tmp.i[0];
  }

  //----------------------------------------------------------------------------
  /// Like frexp but vectorising and the exponent is a double.
  __host__ __device__ inline double getMantExponent(const double x, double & fe)
  {
    uint64_t n = dp2uint64(x);

    // Shift to the right up to the beginning of the exponent.
    // Then with a mask, cut off the sign bit
    uint64_t le = (n >> 52);

    // chop the head of the number: an int contains more than 11 bits (32)
    int32_t e = le; // This is important since sums on uint64_t do not vectorise
    fe = e-1023 ;

    // This puts to 11 zeroes the exponent
    n &=0x800FFFFFFFFFFFFFULL;
    // build a mask which is 0.5, i.e. an exponent equal to 1022
    // which means *2, see the above +1.
    const uint64_t p05 = 0x3FE0000000000000ULL; //dp2uint64(0.5);
    n |= p05;

    return uint642dp(n);
  }

  //----------------------------------------------------------------------------
  /// Like frexp but vectorising and the exponent is a float.
  __host__ __device__ inline float getMantExponentf(const float x, float & fe)
  {
    uint32_t n = sp2uint32(x);
    int32_t e = (n >> 23)-127;
    fe = e;

    // fractional part
    const uint32_t p05f = 0x3f000000; // //sp2uint32(0.5);
    n &= 0x807fffff;// ~0x7f800000;
    n |= p05f;

    return uint322sp(n);
  }
}

// Log double precision --------------------------------------------------------

__host__ __device__ inline double G4Log(double x) {
        const double original_x = x;

        /* separate mantissa from exponent */
        double fe;
        x = G4LogConsts::getMantExponent(x,fe);

        // blending
        x > G4LogConsts::SQRTH? fe+=1. : x+=x ;
        x -= 1.0;

        /* rational form */
        double px =  G4LogConsts::get_log_px(x);

        //for the final formula
        const double x2 = x*x;
        px *= x;
        px *= x2;

        const double qx = G4LogConsts::get_log_qx(x);

        double res = px / qx ;

        res -= fe * 2.121944400546905827679e-4;
        res -= 0.5 * x2  ;

        res = x + res;
        res += fe * 0.693359375;

        if (original_x > G4LogConsts::LOG_UPPER_LIMIT)
                res = std::numeric_limits<double>::infinity();
        if (original_x < G4LogConsts::LOG_LOWER_LIMIT) // THIS IS NAN!
                res =  - std::numeric_limits<double>::quiet_NaN();

        return res;
}

// Log single precision --------------------------------------------------------

namespace G4LogConsts
{
  const float LOGF_UPPER_LIMIT = MAXNUMF;
  const float LOGF_LOWER_LIMIT = 0;

  const float PX1logf = 7.0376836292E-2f;
  const float PX2logf = -1.1514610310E-1f;
  const float PX3logf = 1.1676998740E-1f;
  const float PX4logf = -1.2420140846E-1f;
  const float PX5logf = 1.4249322787E-1f;
  const float PX6logf = -1.6668057665E-1f;
  const float PX7logf = 2.0000714765E-1f;
  const float PX8logf = -2.4999993993E-1f;
  const float PX9logf = 3.3333331174E-1f;

  __host__ __device__ inline float get_log_poly(const float x)
  {
        float y = x*PX1logf;
        y += PX2logf;
        y *= x;
        y += PX3logf;
        y *= x;
        y += PX4logf;
        y *= x;
        y += PX5logf;
        y *= x;
        y += PX6logf;
        y *= x;
        y += PX7logf;
        y *= x;
        y += PX8logf;
        y *= x;
        y += PX9logf;
        return y;
  }

  const float SQRTHF = 0.707106781186547524f;
}

// Log single precision --------------------------------------------------------

__host__ __device__ inline float G4Logf( float x )
{
        const float original_x = x;

        float fe;
        x = G4LogConsts::getMantExponentf( x, fe);

        x > G4LogConsts::SQRTHF? fe+=1.f : x+=x ;
        x -= 1.0f;

        const float x2 = x*x;

        float res = G4LogConsts::get_log_poly(x);
        res *= x2*x;

        res += -2.12194440e-4f * fe;
        res +=  -0.5f * x2;

        res= x + res;

        res += 0.693359375f * fe;

        if (original_x > G4LogConsts::LOGF_UPPER_LIMIT)
                res = std::numeric_limits<float>::infinity();
        if (original_x < G4LogConsts::LOGF_LOWER_LIMIT)
                res = -std::numeric_limits<float>::quiet_NaN();

        return res;
}

//------------------------------------------------------------------------------

__host__ __device__ void logv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
__host__ __device__ void G4Logv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
__host__ __device__ void logfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);
__host__ __device__ void G4Logfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);

#endif /* WIN32 */

#endif /* LOG_H_ */
