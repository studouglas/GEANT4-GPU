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
3. `make`

**Install GEANT-4**<br>
1. (Fedora only) `yum install expat-devel`<br>
2. `cd /path/to/GEANT4-GPU/geant4.10.01.p02-build`<br>
3. `cmake -DGEANT4_INSTALL_DATA=ON -DCMAKE_INSTALL_PREFIX=/path/to/GEANT4-GPU/
geant4.10.01.p02-install /path/to/GEANT4-GPU/geant4.10.01.p02`<br>
4. `make -jN` where `N` is the number of processors on your computer<br>
5. `make install`

**Install G4-STORK**<br>
1. `cd /path/to/GEANT4-GPU/G4STORK/Build`<br>
2. `source /path/to/GEANT4-GPU/geant4.10.01.p02-install/bin/geant4.sh`<br>
3. `rm -rf CMakeCache.txt CMakeFiles/`<br>
4. `cmake -DTOPC_USE=1 -DGeant4_DIR=/path/to/GEANT4-GPU/geant4.10.01.p02-
install/lib/Geant4.10.00.p02/Geant4Config.cmake ../`<br>
5. `make -jN` where `N` is the number of processors on your computer

To test that everything installed properly, (TODO: figure out a test).


Troubleshooting
==========


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
