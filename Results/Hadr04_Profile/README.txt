		    About how this data was gathered
------------------------------------------------------------------------
| These .CSV files were generated using MSV's performance analyzer.    |
| They break down a simple test of Hadr_04 run on a CPU.               |
| Hadr_04 is setup and the macro file run01.mac is called.             |
------------------------------------------------------------------------

This folder contains a set of CSV's which can be opened my excel or open office

-----------------------------------NOTE------------------------------------
***To parse the files correctly ensure that you are seperating by commas***
The notable files:
-The ModuleSummary file breaks down Hadr04 into it's modules and notes the number of calls to each module and the percentage of time spent in each module.
-The FunctionSummary file breaks down Hadr04 into each function and the number of calls made to and the percentage of time spent in the function,
it as well as the encapusalting module.
-The Call Tree file shows how each function is called, the level indicates how many calls deep is is. i.e. Main calls Cats Calls Catnip which terminates.
          												   |Level 0   |Level 1   |Level 2
