Porting G4-STORK particle simulation software to GPU
==========

Geant4 is a widely-used software toolkit used to simulate particle
interactions with matter. Several members of McMaster's Engineering Physics department 
have created G4-STORK, a program that uses the Geant4 toolkit to simulate 
particle interactions in McMaster's nuclear reactor. Due to the computation 
time needed to run these simulations, they cannot simulate particle 
interactions that take place over the course of minutes (or even seconds), or 
that have a large numbers of particles.

The goal of this project is to port some of the calculations done by Geant4 to a GPU 
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
The following software is required:
- gcc (must be version 4.8 - 4.9)

The following hardware is required:
- NVIDIA GPU with CUDA compute capability of at least 2.0

Installation
==========
**Install cmake**<br>
1. Download cmake 2.8.4 from https://cmake.org/files/v2.8/<br>
2. Follow the instructions in the Readme included with the download

**Install Geant4**<br>
If installing on McMaster's servers, add `. /opt/rh/devtoolset-3/enable` to your bash_profile to use the newer version of gcc.

1. `mkdir /path/to/Geant4-GPU/geant4.10.02-build`<br>
2. `mkdir /path/to/Geant4-GPU/geant4.10.02-install`<br>
3. `cd /path/to/Geant4-GPU/geant4.10.02-build`<br>
4. `cmake -DGEANT4_ENABLE_CUDA=ON -DGEANT4_USE_SYSTEM_EXPAT=OFF -DCMAKE_INSTALL_PREFIX=/path/to/Geant4-GPU/geant4.10.02-install ../geant4.10.02`<br>
IF installing on McMaster's server, you must add flag `-DCUDA_HOST_COMPILER=/usr/bin/g++`<br>
5. `make install -jN` where `N` is the number of processors on your computer<br>

**Installing Geant4 on McMaster's Server (no root privileges)**<br>
1. Clone the repo<br>
3. Download the latest version of CMake onto your local desktop<br>
4. Copy the tarred file to McMaster's server via SSH: `scp cmake-3.4.0.tar yourMacId@gpu1.mcmaster.ca:/u50/yourMacId/`<br>
5. Return to your SSH terminal and untar the file (this may take a while): `tar -xvf cmake-3.4.0.tar`<br>
6. Build and install cmake: `cd cmake-3.4.0;./bootstrap;make;make install`<br>
8. Add cmake's bin folder to your path. Open `/u50/yourMacId/.bash_profile` and add the following line right before `export PATH`: `PATH=$PATH:$HOME/cmake-3.4.0/bin`<br>
9. Follow the instructions above to "Install Geant4"<br>

**Setting Environment Variables**<br>
It is recommended to add a line to your bash_profile that loads the Geant4
environment variables when you login, like so:
```
source /path/to/Geant4.10.02-install/bin/Geant4.sh
```

Testing Installation
==========
**Testing Geant4**<br>


Compiling After Changes
==========
Every time you change the source code of Geant4, you need to recompile by running `make install` from `/path/to/Geant4-GPU/geant4.10.02-build`

Troubleshooting
==========
Potential problems include:
- Spaces in pathname to GEANT-GPU
- Unsupported OS

FAQ
==========
**What is Geant4**<br>
Many physics researchers use Geant4 to learn about how particles interact 
with a specific environment. It is a toolkit (i.e. library) that uses the 
Monte Carlo model, meaning each particle's properties are calculated 
independently according to certain probabilities. It runs all those 
calculations, and provides output.

**Why will running the simulations on a GPU improve the performance**<br>
GPU's contain a large amount of cores that can perform calculations much more 
quickly than a CPU if the problem is well-suited to parallelization. Geant4 
runs relatively simple calculations on millions of particles, and each 
particle is completely independent of the others. This is exactly that sort of 
well-suited problem, and stands to see large performance gains.
