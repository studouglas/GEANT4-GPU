#ifndef G4NeutronHPVector_h
#define G4NeutronHPVector_h 1

//#include "G4NeutronHPDataPoint.hh"
//#include "G4PhysicsVector.hh"
//#include "G4NeutronHPInterpolator.hh"
//#include "Randomize.hh"
//#include "G4ios.hh"
//#include <fstream>
//#include "G4InterpolationManager.hh"
//#include "G4NeutronHPInterpolator.hh"
//#include "G4NeutronHPHash.hh"
#include <cmath>
//#include <vector> -- Replaced by Thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#if defined WIN32-VC
   #include <float.h>
#endif

class G4NeutronHPVector
{
//  friend G4NeutronVector & operator + (G4NeutronHPVector & left,
//									   G4NeutronHPVector & right);

  public :
//  __device__ __host__ G4NeutronHPVector();
//  __device__ __host__ G4NeutronHPVector(G4int n);
//  __device__ __host__ ~G4NeutronHPVector();
  //__device__ __host__ G4NeutronHPVector & operator = (const);
  
   
  inline __device__ __host__ void SetVerbose(G4int ff)
  {
	Verbose == ff;
  }
  
  inline __device__ __host__ void Times(G4double factor)
  {
	G4int i;
	for(i = 0; i<nEntries; i++)
	{
		theData[i].SetY(theData[i].GetY()*factor);
	}
	if(theIntegral != 0)
	{
		theIntegral[i] *= factor;
	}
  }
  
  inline __device__ __host__ void SetPoint(G4int i, const G4NeutronHPDataPoint & it)
  {
    G4double x = it.GetX();
    G4double y = it.GetY();
    SetData(i, x, y);
  }
  
  inline __device__ __host__ void SetData(G4int i, G4double x, G4double y) 
  { 
	Check(i);
    if(y>maxValue) maxValue=y;
    theData[i].SetData(x, y);
  }

  inline __device__ __host__ void SetX(G4int i, G4double e)
  {
    Check(i);
    theData[i].SetX(e);
  }
  
  inline __device__ __host__ void SetEnergy(G4int i, G4double e)
  {
    Check(i);
    theData[i].SetX(e);
  }
 
  inline __device__ __host__ void SetY(G4int i, G4double x)
  {
    Check(i);
    if(x>maxValue) maxValue=x;
    theData[i].SetY(x);
  }
 
  inline __device__ __host__ void SetXsec(G4int i, G4double x)
  {
    Check(i);
    if(x>maxValue) maxValue=x;
    theData[i].SetY(x);
  }
  
  inline __device__ __host__ G4double GetEnergy(G4int i) const { return theData[i].GetX(); }
  inline __device__ __host__ G4double GetXsec(G4int i) { return theData[i].GetY(); }
  inline __device__ __host__ G4double GetX(G4int i) const
  { 
	if (i<0) i=0;
    if(i>=GetVectorLength()) i=GetVectorLength()-1;
    return theData[i].GetX();
  }
 
  inline __device__ __host__ const G4NeutronHPDataPoint & GetPoint(G4int i) const { return theData[i]; }
 
  __device__ __host__ void Hash() 
  {
    G4int i;
    G4double x, y;
    for(i=0 ; i<nEntries; i++)
    {
      if(0 == (i+1)%10)
      {
        x = GetX(i);
	y = GetY(i);
	theHash.SetData(i, x, y);
      }
    }
  }
 
  __device__ __host__ void ReHash()
  {
    theHash.Clear();
    Hash();
  }
 
  __device__ __host__ G4double GetXsec(G4double e);
 
  __device__ __host__ G4double GetXsec(G4double e, G4int min)
  {
    G4int i;
    for(i=min ; i<nEntries; i++)
    {
      if(theData[i].GetX()>e) break;
    }
    G4int low = i-1;
    G4int high = i;
    if(i==0)
    {
      low = 0;
      high = 1;
    }
    else if(i==nEntries)
    {
      low = nEntries-2;
      high = nEntries-1;
    }
    G4double y;
    if(e<theData[nEntries-1].GetX()) 
    {
      // Protect against doubled-up x values
      if( (theData[high].GetX()-theData[low].GetX())/theData[high].GetX() < 0.000001)
      {
        y = theData[low].GetY();
      }
      else
      {
        y = theInt.Interpolate(theManager.GetScheme(high), e, 
                               theData[low].GetX(), theData[high].GetX(),
		  	       theData[low].GetY(), theData[high].GetY());
      }
    }
    else
    {
      y=theData[nEntries-1].GetY();
    }
    return y;
  }
 
  inline __device__ __host__ G4double GetY(G4double x)  {return GetXsec(x);}
  inline __device__ __host__ G4int GetVectorLength() const {return nEntries;}
 
  inline __device__ __host__ G4double GetY(G4int i)
  { 
    if (i<0) i=0;
    if(i>=GetVectorLength()) i=GetVectorLength()-1;
    return theData[i].GetY(); 
  }
 
  inline __device__ __host__ G4double GetY(G4int i) const
  {
    if (i<0) i=0;
    if(i>=GetVectorLength()) i=GetVectorLength()-1;
    return theData[i].GetY(); 
  }
  
  __device__ __host__ void Dump();
 
  inline __device__ __host__ void InitInterpolation(std::istream & aDataFile)
  {
    theManager.Init(aDataFile);
  }
 
  __device__ __host__ void Init(std::istream & aDataFile, G4int total, G4double ux=1., G4double uy=1.)
  {
    G4double x,y;
    for (G4int i=0;i<total;i++)
    {
      aDataFile >> x >> y;
      x*=ux;
      y*=uy;
      SetData(i,x,y);
      if(0 == nEntries%10)
      {
        theHash.SetData(nEntries-1, x, y);
      }
    }
  }
 
  __device__ __host__ void Init(std::istream & aDataFile,G4double ux=1., G4double uy=1.)
  {
    G4int total;
    aDataFile >> total;
    if(theData!=0) delete [] theData;
    theData = new G4NeutronHPDataPoint[total]; 
    nPoints=total;
    nEntries=0;    
    theManager.Init(aDataFile);
    Init(aDataFile, total, ux, uy);
  }
  
  __device__ __host__ void ThinOut(G4double precision);
 
  inline __device__ __host__ void SetLabel(G4double aLabel)
  {
    label = aLabel;
  }
 
  inline __device__ __host__ G4double GetLabel()
  {
    return label;
  }
  
  inline __device__ __host__ void CleanUp()
  {
    nEntries=0;   
    theManager.CleanUp();
    maxValue = -DBL_MAX;
    theHash.Clear();
//080811 TK DB 
    delete[] theIntegral;
    theIntegral = NULL;
  }

  // merges the vectors active and passive into *this
  inline __device__ __host__ void Merge(G4NeutronHPVector * active, G4NeutronHPVector * passive)
  {
    CleanUp();
    G4int s_tmp = 0, n=0, m_tmp=0;
    G4NeutronHPVector * tmp;
    G4int a = s_tmp, p = n, t;
    while (a<active->GetVectorLength()&&p<passive->GetVectorLength())
    {
      if(active->GetEnergy(a) <= passive->GetEnergy(p))
      {
        G4double xa = active->GetEnergy(a);
        G4double yy = active->GetXsec(a);
        SetData(m_tmp, xa, yy);
        theManager.AppendScheme(m_tmp, active->GetScheme(a));
        m_tmp++;
        a++;
        G4double xp = passive->GetEnergy(p);

//080409 TKDB 
        //if( std::abs(std::abs(xp-xa)/xa)<0.001 ) p++;
        if ( !( xa == 0 ) && std::abs(std::abs(xp-xa)/xa)<0.001 ) p++;
      } else {
        tmp = active; 
        t=a;
        active = passive; 
        a=p;
        passive = tmp; 
        p=t;
      }
    }
    while (a!=active->GetVectorLength())
    {
      SetData(m_tmp, active->GetEnergy(a), active->GetXsec(a));
      theManager.AppendScheme(m_tmp++, active->GetScheme(a));
      a++;
    }
    while (p!=passive->GetVectorLength())
    {
      if(std::abs(GetEnergy(m_tmp-1)-passive->GetEnergy(p))/passive->GetEnergy(p)>0.001)
      //if(std::abs(GetEnergy(m)-passive->GetEnergy(p))/passive->GetEnergy(p)>0.001)
      {
        SetData(m_tmp, passive->GetEnergy(p), passive->GetXsec(p));
        theManager.AppendScheme(m_tmp++, active->GetScheme(p));
      }
      p++;
    }
  }    
  
  __device__ __host__ void Merge(G4InterpolationScheme aScheme, G4double aValue, 
             G4NeutronHPVector * active, G4NeutronHPVector * passive);
  
  __device__ __host__ G4double SampleLin() // Samples X according to distribution Y, linear int
  {
    G4double result;
    if(theIntegral==0) IntegrateAndNormalise();
    if(GetVectorLength()==1)
    {
      result = theData[0].GetX();
    }
    else
    {
      G4int i;
      G4double rand = G4UniformRand();
      
      // this was replaced 
//      for(i=1;i<GetVectorLength();i++)
//      {
//	if(rand<theIntegral[i]/theIntegral[GetVectorLength()-1]) break;
//      }

// by this (begin)
      for(i=GetVectorLength()-1; i>=0 ;i--)
      {
	if(rand>theIntegral[i]/theIntegral[GetVectorLength()-1]) break;
      }
      if(i!=GetVectorLength()-1) i++;
// until this (end)
      
      G4double x1, x2, y1, y2;
      y1 = theData[i-1].GetX();
      x1 = theIntegral[i-1];
      y2 = theData[i].GetX();
      x2 = theIntegral[i];
      if(std::abs((y2-y1)/y2)<0.0000001) // not really necessary, since the case is excluded by construction
      {
	y1 = theData[i-2].GetX();
	x1 = theIntegral[i-2];
      }
      result = theLin.Lin(rand, x1, x2, y1, y2);
    }
    return result;
  }
  
  __device__ __host__ G4double Sample(); // Samples X according to distribution Y
  
  __device__ __host__ G4double * Debug()
  {
    return theIntegral;
  }

  inline __device__ __host__ void IntegrateAndNormalise()
  {
    G4int i;
    if(theIntegral!=0) return;
    theIntegral = new G4double[nEntries];
    if(nEntries == 1)
    {
      theIntegral[0] = 1;
      return;
    }
    theIntegral[0] = 0;
    G4double sum = 0;
    G4double x1 = 0;
    G4double x0 = 0;
    for(i=1;i<GetVectorLength();i++)
    {
      x1 = theData[i].GetX();
      x0 = theData[i-1].GetX();
      if (std::abs(x1-x0) > std::abs(x1*0.0000001) )
      {
	//********************************************************************
	//EMendoza -> the interpolation scheme is not always lin-lin
	/*
        sum+= 0.5*(theData[i].GetY()+theData[i-1].GetY())*
                  (x1-x0);
	*/
	//********************************************************************
        G4InterpolationScheme aScheme = theManager.GetScheme(i);
        G4double y0 = theData[i-1].GetY();
        G4double y1 = theData[i].GetY();
	G4double integ=theInt.GetBinIntegral(aScheme,x0,x1,y0,y1);
#if defined WIN32-VC
	if(!_finite(integ)){integ=0;}
#elif defined __IBMCPP__
	if(isinf(integ)||isnan(integ)){integ=0;}
#else
	if(std::isinf(integ)||std::isnan(integ)){integ=0;}
#endif
	sum+=integ;
	//********************************************************************
      }
      theIntegral[i] = sum;
    }
    G4double total = theIntegral[GetVectorLength()-1];
    for(i=1;i<GetVectorLength();i++)
    {
      theIntegral[i]/=total;
    }
  }

  inline __device__ __host__ void Integrate() 
  {
    G4int i;
    if(nEntries == 1)
    {
      totalIntegral = 0;
      return;
    }
    G4double sum = 0;
    for(i=1;i<GetVectorLength();i++)
    {
      if(std::abs((theData[i].GetX()-theData[i-1].GetX())/theData[i].GetX())>0.0000001)
      {
        G4double x1 = theData[i-1].GetX();
        G4double x2 = theData[i].GetX();
        G4double y1 = theData[i-1].GetY();
        G4double y2 = theData[i].GetY();
        G4InterpolationScheme aScheme = theManager.GetScheme(i);
        if(aScheme==LINLIN||aScheme==CLINLIN||aScheme==ULINLIN)
        {
          sum+= 0.5*(y2+y1)*(x2-x1);
        }
        else if(aScheme==LINLOG||aScheme==CLINLOG||aScheme==ULINLOG)
        {
          G4double a = y1;
          G4double b = (y2-y1)/(std::log(x2)-std::log(x1));
          sum+= (a-b)*(x2-x1) + b*(x2*std::log(x2)-x1*std::log(x1));
        }
        else if(aScheme==LOGLIN||aScheme==CLOGLIN||aScheme==ULOGLIN)
        {
          G4double a = std::log(y1);
          G4double b = (std::log(y2)-std::log(y1))/(x2-x1);
          sum += (std::exp(a)/b)*(std::exp(b*x2)-std::exp(b*x1));
        }
        else if(aScheme==HISTO||aScheme==CHISTO||aScheme==UHISTO)
        {
          sum+= y1*(x2-x1);
        }
        else if(aScheme==LOGLOG||aScheme==CLOGLOG||aScheme==ULOGLOG)
        {
          G4double a = std::log(y1);
          G4double b = (std::log(y2)-std::log(y1))/(std::log(x2)-std::log(x1));
          sum += (std::exp(a)/(b+1))*(std::pow(x2,b+1)-std::pow(x1,b+1));
        }
        else
        {
          throw G4HadronicException(__FILE__, __LINE__, "Unknown interpolation scheme in G4NeutronHPVector::Integrate");
        }
          
      }
    }
    totalIntegral = sum;
  }
  
  inline __device__ __host__ G4double GetIntegral() // linear interpolation; use with care
  { 
    if(totalIntegral<-0.5) Integrate();
    return totalIntegral; 
  }
  
  inline __device__ __host__ void SetInterpolationManager(const G4InterpolationManager & aManager)
  {
    theManager = aManager;
  }
  
  inline __device__ __host__ const G4InterpolationManager & GetInterpolationManager() const
  {
    return theManager;
  }
  
  inline __device__ __host__ void SetInterpolationManager(G4InterpolationManager & aMan)
  {
    theManager = aMan;
  }
  
  inline __device__ __host__ void SetScheme(G4int aPoint, const G4InterpolationScheme & aScheme)
  {
    theManager.AppendScheme(aPoint, aScheme);
  }
  
  inline __device__ __host__ G4InterpolationScheme GetScheme(G4int anIndex)
  {
    return theManager.GetScheme(anIndex);
  }
  
  __device__ __host__ G4double GetMeanX()
  {
    G4double result;
    G4double running = 0;
    G4double weighted = 0;
    for(G4int i=1; i<nEntries; i++)
    {
      running += theInt.GetBinIntegral(theManager.GetScheme(i-1),
                           theData[i-1].GetX(), theData[i].GetX(),
                           theData[i-1].GetY(), theData[i].GetY());
      weighted += theInt.GetWeightedBinIntegral(theManager.GetScheme(i-1),
                           theData[i-1].GetX(), theData[i].GetX(),
                           theData[i-1].GetY(), theData[i].GetY());
    }  
    result = weighted / running;  
    return result;
  }
  
  //__device__ __host__ std::vector<G4double> GetBlocked() {return theBlocked;}
  //__device__ __host__ std::vector<G4double> GetBuffered() {return theBuffered;}
  
  //__device__ __host__ G4double Get15percentBorder();
  //__device__ __host__ G4double Get50percentBorder();
  __device__ __host__ double Get15percentBorder();
  __device__ __host__ double Get50percentBorder()
  
  
  
  private:
  //__device__ __host__ void Check(G4int i);
  __device__ __host__ void Check(int i);
  
  
  //__device__ __host__ G4bool IsBlocked(G4double aX);
  __devuce__ __host__ bool IsBlocked(double aX);
  
  //private:
  
  //G4NeutronHPInterpolator theLin;
  
  private:
  
  //G4double totalIntegral;
  double totalIntegral;
  
  //G4NeutronHPDataPoint * theData; // the data
  //G4InterpolationManager theManager; // knows how to interpolate the data.
  //G4double * theIntegral;
  //G4int nEntries;
  //G4int nPoints;
  //G4double label;
  double * theIntegral;
  int nEntries;
  int nPoints;
  double label;
  
  //G4NeutronHPInterpolator theInt;
  //G4int Verbose;
  // debug only
  //G4int isFreed;
  int Verbose;
  int isFreed;
  
  //G4NeutronHPHash theHash;
  //G4double maxValue;
  double maxValue;
  
  //std::vector<G4double> theBlocked;
  //std::vector<G4double> theBuffered;
  //G4double the15percentBorderCash;
  //G4double the50percentBorderCash;
  double the15percentBorderCash;
  double the50percentBorderCash;
  
};

#endif