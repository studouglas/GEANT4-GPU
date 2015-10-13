#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(G4_STORK)


#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default
# You can set WITH_GEANT4_UIVIS to OFF via the command line or ccmake/cmake-gui
# to build a batch mode only executable
#
option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()


#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
#
include(${Geant4_USE_FILE})
add_definitions(-fpermissive)


#----------------------------------------------------------------------------
# Custom commands for generating marshalled classes
#


#----------------------------------------------------------------------------
# Add options
#

# Timing of individual components
option(G4TIME_USE "Output object execution times" OFF)
if(G4TIME_USE)
  add_definitions(-DG4TIME -DG4TIMESD -DG4TIMERA -DG4TIMEEA)
endif()

# Explicit totals for each neutron loss type (fission, capture, etc)
option(STORK_EXPLICIT_LOSS "Show explicit loss type counters" OFF)
if(STORK_EXPLICIT_LOSS)
  add_definitions(-DSTORK_EXPLICIT_LOSS)
endif()

# Visualization of geometry, hits, etc
option(G4VISUALIZE_USE "Provide visualization of simulation" OFF)
if(G4VISUALIZE_USE)
    add_definitions(-DG4VISUALIZE_USE)
endif()

# Step-by-step tracking
option(G4VERBOSE_TRACKING_USE "Show step-by-step tracking output" OFF)
if(G4VERBOSE_TRACKING_USE)
    add_definitions(-G4VERBOSE_TRACKING_USE)
endif()

# Parallel processing using TOP-C
option(TOPC_USE "Use TOP-C parallel architecture" OFF)
option(TOPC_USE_SEQ "Use TOP-C in sequential mode" OFF)
if(TOPC_USE)
  if(TOPC_USE_SEQ)
    set(PARMODE "--seq")
  else()
    set(PARMODE "--mpi")
  endif()

  # Get TOP-C libraries
  execute_process(COMMAND "topc-config" ${PARMODE} "--libs"
          OUTPUT_VARIABLE TOPC_LIBS)

  # Remove trailing whitespace
  string(STRIP ${TOPC_LIBS} TOPC_LIBS)

  # Get TOP-C compiler flags
  execute_process(COMMAND "topc-config" ${PARMODE} "--cflags"
          OUTPUT_VARIABLE TOPC_CFLAGS)

  add_definitions(${TOPC_CFLAGS} -DG4USE_TOPC)
endif()


#----------------------------------------------------------------------------
# Locate sources and headers for this project
#
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${Geant4_INCLUDE_DIR})
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc
           ${PROJECT_SOURCE_DIR}/src/*.icc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh
           ${PROJECT_SOURCE_DIR}/include/*.h)


#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
add_executable(g4stork g4stork.cc ${sources} ${headers})
target_link_libraries(g4stork ${Geant4_LIBRARIES} ${TOPC_LIBS})


#----------------------------------------------------------------------------
# Copy all scripts to the build directory, i.e. the directory in which we
# build air_shower. This is so that we can run the executable directly because it
# relies on these scripts being in the current working directory.
#


#----------------------------------------------------------------------------
# Add program to the project targets
# (this avoids the need of typing the program name after make)
#
add_custom_target(G4_STORK DEPENDS g4stork)


#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
install(TARGETS g4stork DESTINATION bin)
