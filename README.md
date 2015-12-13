Porting G4-STORK particle simulation software to GPU
==========

GEANT-4 is a widely-used software toolkit used to simulate particle
interactions. Several members of McMaster's Engineering Physics department 
have created G4-STORK, a program that uses the GEANT-4 toolkit to simulate 
particle interactions in McMaster's nuclear reactor. Due to the computation 
time needed to run these simulations, they cannot simulate particle 
interactions that take place over the course of minutes (or even seconds), or 
that have a large numbers of particles.

The goal of this project is to port the calculations done by GEANT-4 to a GPU 
architecture using CUDA. This will significantly increase the performance of 
the simulations, allowing researchers to use more accurate models of the 
system.

## Table of Contents
**[Prerequisites](#prerequisites)**<br>
**[Installation](#installation)**<br>
**[Testing Installation](#testing-installation)**<br>
**[Compiling After Changes](#compiling-after-changes)**<br>
**[Troubleshooting](#troubleshooting)**<br>
**[FAQ](#FAQ)**<br>

Prerequisites
==========
The following operating systems are supported:
- Mac OS X (tested on 10.11 with Xcode 6)
- Fedora 20

*Note*: Xcode 7 includes a newer version of the clang compiler that is not yet 
supported.

To run computations on the GPU with CUDA, a relatively new **NVIDIA GPU** is 
required.

Installation
==========
**Install cmake**<br>
1. Download cmake 2.8.4 from https://cmake.org/files/v2.8/<br>
2. Follow the instructions in the readme included with the download

**Install topc-2.5.2**<br>
1. `cd /path/to/GEANT4-GPU/topc-2.5.2`<br>
2. `./configure` (installs to `usr/local/include` and `usr/local/src`)<br>
3. `make install`

**Install marshalgen-1.0**<br>
1. (Fedora only) `yum install bison flex`<br>
2. `cd /path/to/GEANT4-GPU/marshalgen-1.0`<br>
3. `make`<br>
4. At this point, make sure that the `usr/local/marshalgen-1.0` folder exists
and contains all the same files as the *marshalgen-1.0* folder in GEANT4-GPU. If
not, simply copy those files into `usr/local/marshalgen-1.0`.<br>
5. Note: there is a syntax error in marshalgen-1.0/Phase1/mgen.pl line 683 (one
extra `\)` before the `;\n";`. The version in the repo is fixed, but if you 
downloaded marshalgen from another source, you need to fix this)

**Install GEANT-4**<br>
1. (Fedora only) `yum install expat-devel`<br>
2. `mkdir /path/to/GEANT4-GPU/geant4.10.00.p02-build /path/to/GEANT4-GPU/
geant4.10.00.p02-install`<br>
3. `cd /path/to/GEANT4-GPU/geant4.10.00.p02-build`<br>
4. `cmake -DGEANT4_INSTALL_DATA=ON -DCMAKE_INSTALL_PREFIX=/path/to/GEANT4-GPU/geant4.10.00.p02-install /path/to/GEANT4-GPU/geant4.10.00.p02`<br>
5. `make -jN` where `N` is the number of processors on your computer<br>
6. `make install`

**Install G4-STORK**<br>
1. `mkdir /path/to/GEANT4-GPU/G4STORK/Build`<br>
2. `cd /path/to/GEANT4-GPU/G4STORK/Build`<br>
3. `source /path/to/GEANT4-GPU/geant4.10.00.p02-install/bin/geant4.sh`<br>
4. `rm -rf CMakeCache.txt CMakeFiles/`<br>
5. `cmake -DTOPC_USE=1 -DGeant4_DIR=/path/to/GEANT4-GPU/geant4.10.01.p02-install/lib/Geant4.10.00.p02/Geant4Config.cmake ../` (note: it may be *lib64*
 instead of *lib* on Linux)<br>
6. `make -f ../MarshalMakefile` (note: if this fails, make sure
`usr/local/marshalgen-1.0` contains the `marshalgen` binary)<br>
7. `make -jN` where `N` is the number of processors on your computer
8. Open `/path/to/GEANT4-GPU/Build/addFilesG4STORK` and modify the top few 
variables with the correct paths for your install.

**Installing Geant4 on McMaster's Server (no root privileges)**<br>
1. SSH into one of McMaster's servers (i.e. `ssh yourMacId@gpu1.mcmaster.ca`), account is on a shared drive across all department servers so once you install once you can access it from any one.<br>
2. Set up your .gitconfig file, ssh keys, and clone the repo in your home folder (path is `/u50/yourMacId/`)<br>
3. You'll need to install cmake, to do this download the latest version onto your regular desktop<br>
4. Copy the tarred file to McMaster's server via SSH: `scp cmake-3.4.0.tar yourMacId@gpu1.mcmaster.ca:/u50/yourMacId/`<br>
5. Return to your SSH terminal and untar the file (this may take a while): `tar -xvf cmake-3.4.0.tar`<br>
6. Build and install cmake: `cd cmake-3.4.0;./bootstrap;make;make install`<br>
8. Add cmake's bin folder to your path. Open `/u50/yourMacId/.bash_profile` and add the following line right before `export PATH`: `PATH=$PATH:$HOME/cmake-3.4.0/bin`<br>
9. Follow the instructions above to "Install GEANT-4" (starting from 2), and add the following flag to the cmake command in step 4: `-DGEANT4_USE_SYSTEM_EXPAT=OFF`<br>

**Setting Environment Variables**<br>
It is recommended to add a line to your bash_profile that loads the Geant4
environment variables, like so:
```
source /path/to/geant4.10.00.p02-install/bin/geant4.sh
```

Testing Installation
==========
**Testing GEANT4**<br>
There are several basic examples included with GEANT4 which can be run to test 
the install. To run the example `B1`:<br>
1. `cd /path/to/GEANT4-GPU/geant4.10.00.p02/examples/basic`<br>
2. `mkdir B1-build`<br>
3. `cd B1-build`<br>
4. `cmake -DGeant4_DIR=/path/to/GEANT4-GPU/geant4.10.00.p02-install/lib/Geant4-10.0.2 ../B1`<br>
5. `make -jN` where `N` is the number of cores on your machine<br>
6. `./exampleB1`

**Testing G4-STORK**<br>
To test that everything installed properly, run the following command from 
`/path/to/GEANT4-GPU/G4STORK/Build`: 
```
./g4stork ../InputFiles/C6LatticeInput.txt
```
If it runs with no errors, then you should be all set up!


Compiling After Changes
==========
Every time you change the source code of G4STORK or GEANT4, you need to 
recompile. From `/path/to/GEANT4-GPU/geant4.10.00.p02-build` rerun the cmake command from *Install GEANT-4: Step 4*, run `make`, and then run `make install`. Note that you only need to rerun cmake if you modified any CMake files, but you must always run the two make commands.

Troubleshooting
==========
Potential problems include:
- Spaces in pathname to GEANT-GPU
- Unsupported OS
- Newer version of Clang (included with Xcode 7), download Xcode 6 and uninstall
 Xcode 7 if this is the case

FAQ
==========
**What is GEANT-4**<br>
Many physics researchers use GEANT-4 to learn about how particles interact 
with a specific environment. It is a toolkit (i.e. library) that uses the 
Monte Carlo model, meaning each particle's properties are calculated 
independently according to certain probabilities. It runs all those 
calculations, and provides output.

**What is G4-STORK**<br>
McMaster's Engineering Physics department created G4-STORK -- a project that 
leverages GEANT-4 to study the McMaster nuclear reactor. G4-STORK includes the 
necessary data structures and algorithms specific to the reactor, and also 
adds CPU parallelization via the MPI model.

**Why will running the simulations on a GPU improve the performance**<br>
GPU's contain a large amount of cores that can perform calculations much more 
quickly than a CPU if the problem is well-suited to parallelization. GEANT-4 
runs relatively simple calculations on millions of particles, and each 
particle is completely independent of the others. This is exactly that sort of 
well-suited problem, and stands to see large performance gains.
