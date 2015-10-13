General Information
===================

This is version 1.0 of Marshalgen, last updated in November, 2005.
Marshalgen stands for MARSHALing code GENerator. It is a package that
allows one to generate marshaling code for C++ structs or classes
based on annotations of the existing source code.



Installation
============
The installation can be done by typing the command
  make

The executable, ./marshalgen, is then available in the current directory.
Usage:  ./marshalgen  FILE.h
  where FILE.h is an include file with Marshalgen annotation.
  FILE.h can be in any directory.

This creates MarshaledFILE.h in the same directory as FILE.h, and it
does:  #include <FILE.h>



Manual
============

See the file 'MANUAL.txt'


Contact
============
 For any bug report, suggestion, or question, feel free to contact us.
Viet Ha Nguyen (vietha@ccs.neu.edu)
Gene Cooperman (gene@ccs.neu.edu)  
