CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ARTOS)

set(BUILD_VERSION 1)
set(API_VERSION 1)


#### Build FFLD static library ####
ADD_SUBDIRECTORY(ffld)
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/ffld")
#### Build static library for image repository access ####
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/ImageNet")
ADD_SUBDIRECTORY(ImageNet)
# Add compiler flags to build static libraries as place-independent code (pic),
# so that it can be linked to our shared ARTOS library.
if(UNIX)
    SET_TARGET_PROPERTIES(ffld PROPERTIES COMPILE_FLAGS "-fpic")
    SET_TARGET_PROPERTIES(imagenet PROPERTIES COMPILE_FLAGS "-fpic")
endif(UNIX)


#### Build ARTOS shared library ####

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

# List files and set properties
SET(HEADERS DPMDetection.h FeatureExtractor.h HOGFeatureExtractor.h StationaryBackground.h ModelLearner.h ImageNetModelLearner.h ModelEvaluator.h
Random.h harmony_search.h sysutils.h strutils.h portable_endian.h libartos_def.h libartos.h)
SET(SOURCES DPMDetection.cc FeatureExtractor.cc StationaryBackground.cc ModelLearner.cc ImageNetModelLearner.cc ModelEvaluator.cc
Random.cc harmony_search.cc sysutils.cc strutils.cc)
ADD_LIBRARY(artos SHARED ${SOURCES} libartos.cc)
SET_TARGET_PROPERTIES(artos PROPERTIES VERSION ${BUILD_VERSION} SOVERSION ${API_VERSION})

# Link sub-libraries
IF(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC)
    TARGET_LINK_LIBRARIES(artos -Wl,-whole-archive ffld imagenet -Wl,-no-whole-archive)
ELSEIF("${CMAKE_SYSTEM}" MATCHES IRIX)
    TARGET_LINK_LIBRARIES(artos -Wl,-all ffld imagenet -Wl,-notall)
ELSE()
    TARGET_LINK_LIBRARIES(artos ffld imagenet)
ENDIF()


#### Find and link libraries used by FFLD ####

# There are no CMake FFTW package, so find it ourselves
FIND_PATH(FFTW3_INCLUDE_DIR fftw3.h)
FIND_LIBRARY(FFTW3_LIBRARIES fftw3f)
IF(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARIES)
  MESSAGE(FATAL_ERROR "Could not find fftw3.")
ENDIF()
INCLUDE_DIRECTORIES(${FFTW3_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(artos ${FFTW3_LIBRARIES})

FIND_PACKAGE(JPEG REQUIRED)
IF(JPEG_FOUND)
  INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(artos ${JPEG_LIBRARIES})
ENDIF()

FIND_PACKAGE(LibXml2 REQUIRED)
IF(LIBXML2_FOUND)
  INCLUDE_DIRECTORIES(${LIBXML2_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(artos ${LIBXML2_LIBRARIES})
  ADD_DEFINITIONS(${LIBXML2_DEFINITIONS})
ENDIF()

# Not required, but strongly recommended on multi-core systems
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF()
