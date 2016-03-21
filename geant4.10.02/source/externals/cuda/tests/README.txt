Running tests for G4ParticleHPVector using CUDA.

Assumptions
The testing files assume that there is a working copy of Geant4 on the user's system, setup as per the installation instructions of Geant4-GPU. In particular, the 'geant4.10.02-install' directory must be in the same directory as the 'geant4.10.02' folder.

Generating Test Results
1. Build Geant4 with CUDA disabled (see project README for more information).
2. Run 'make clean' to remove any old files.
3. Run 'make' to build the executable files for generating and analyzing the test results.
4. Execute './GenerateTestResults 0', where 0 indicates that Geant4 was compiled with CUDA disabled.
5. Rebuild Geant4 with CUDA enabled, and repeat steps 2-4.

At this point, there should be 4 text files in the tests directory. These contain the unit test results and runtimes, respectively, for each of the CPU and GPU runs.

Analyzing Test Results
1. After generating the test results as outlined above, simply run ./AnalyzeTestResults. The result of each individual test will be printed to the terminal.
2. A csv file will be created to easily compare runtimes of each function with CUDA enabled and disabled.

