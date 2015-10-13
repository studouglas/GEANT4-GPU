/*
StorkQuad.hh

Created by:		Liam Russell
Date:			07-08-2011
Modified:		11-03-2013

Template class that can be used to store four items of different types.

*/


#ifndef STORKQUAD_HH_INCLUDED
#define STORKQUAD_HH_INCLUDED


template <class T1, class T2, class T3, class T4>
class StorkQuad
{
    public:

        // Data types
        typedef T1 first_type;
        typedef T2 second_type;
        typedef T3 third_type;
        typedef T4 fourth_type;

        T1 first;
        T2 second;
        T3 third;
        T4 fourth;

        // Default Constructor and destructor
        StorkQuad()
        :first(T1()), second(T2()), third(T3()), fourth(T4()) {}
        ~StorkQuad() {}

        // Constructor
        StorkQuad(const T1& x, const T2& y, const T3& z, const T4& w)
        : first(x), second(y), third(z), fourth(w) {}

        // Copy constructor
        template <class U, class V, class R, class S>
        StorkQuad(const StorkQuad<U,V,R,S> &p)
        : first(p.first), second(p.second), third(p.third), fourth(p.fourth) {}
};

#endif // STORKQUAD_HH_INCLUDED
