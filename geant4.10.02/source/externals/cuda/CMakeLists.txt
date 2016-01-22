cmake_minimum_required (VERSION 2.8)

find_package (CUDA REQUIRED)

include_directories(/usr/local/cuda/include)
include_directories(include)
include_directories(src)

# sets the macro in the source file
add_definitions(-DGEANT4_ENABLE_CUDA=1)

set(G4cuda_HEADERS
	include/G4ParticleHPVector_CUDA.h
	include/G4ParticleHPInterpolator_CUDA.hh
	include/G4InterpolationScheme_CUDA.hh
	include/G4ParticleHPDataPoint_CUDA.hh
)

set(G4cuda_SOURCES
	src/G4ParticleHPVector_CUDA.cu
)

include(Geant4MacroLibraryTargets)
GEANT4_LIBRARY_TARGET_CUDA(NAME G4cuda SOURCES ${G4cuda_SOURCES} ${G4cuda_HEADERS})

set_property(GLOBAL APPEND PROPERTY GEANT4_BUILDTREE_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(DIRECTORY include/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
	COMPONENT Development
	PATTERN "*.h"
	PATTERN "*.hh")
