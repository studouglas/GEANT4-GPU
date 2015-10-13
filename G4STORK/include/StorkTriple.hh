/*
StorkTriple.hh

Created by:		Liam Russell
Date:			14-06-2011
Modified:		11-03-2013

Template class designed to store three objects (can be different types).

*/


#ifndef STORKTRIPLE_HH_INCLUDED
#define STORKTRIPLE_HH_INCLUDED


template <class T1, class T2, class T3>
class StorkTriple
{
    public:

        // Data types
        typedef T1 first_type;
        typedef T2 second_type;
        typedef T3 third_type;

        T1 first;
        T2 second;
        T3 third;

        // Default constructor and destructor
        StorkTriple() : first(T1()), second(T2()), third(T3()) {}
        ~StorkTriple() {}

        // Constructor
        StorkTriple(const T1& x, const T2& y, const T3& z)
        : first(x), second(y), third(z) {}

        // Set data members
        void Set(const T1& x, const T2& y, const T3& z)
        {
            first = x;
            second = y;
            third = z;
        }
};

#endif // STORKTRIPLE_HH_INCLUDED
