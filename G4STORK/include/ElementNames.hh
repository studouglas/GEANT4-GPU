#ifndef ElementNames_HH
#define ElementNames_HH

#include <string>
#include <iostream>
using namespace std;

class ElementNames
{
    public:
        ElementNames();
        virtual ~ElementNames();
        static void ClearStore();
        static void SetElementNames();
        static string GetName(int Z)
        {
            return elementName[Z];
        }
        static bool CheckName(string name);
        static bool CheckName(string name, int Z);
        static string *elementName;
    protected:
    private:

};

#endif // ElementNames_HH
