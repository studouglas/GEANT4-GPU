#if ${ENABLE_CUDA}
	#include "${PROJECT_SOURCE_DIR}/source/externals/cuda/include/G4ParticleHPVector_CUDA.hh"
	#define GEANT4_ENABLE_CUDA 1
#else
	#define GEANT4_ENABLE_CUDA 0
#endif