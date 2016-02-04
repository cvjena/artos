CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2)

PROJECT(ARTOS)

set(BUILD_VERSION "2.0")
set(API_VERSION 2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/")


#### Set C++ standard ####
IF(CMAKE_VERSION VERSION_GREATER 3.1)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
ELSEIF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()


#### Build options ####
OPTION(ARTOS_CACHE_POSITIVES "Keep positive samples in RAM to save time." ON)
OPTION(ARTOS_BUILD_TOOLS "Build C++ files in the tools directory (not required by any part of ARTOS)." ON)

IF(NOT ARTOS_CACHE_POSITIVES)
  ADD_DEFINITIONS(-DNO_CACHE_POSITIVES)
ENDIF()

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()


#### Build static library for image repository access ####
IF(NOT ARTOS_IMAGE_REPOSITORY_SRC)
  SET(ARTOS_IMAGE_REPOSITORY_SRC "ImageNet" CACHE STRING "Subdirectory with source files for image repository access.")
ENDIF()
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/${ARTOS_IMAGE_REPOSITORY_SRC}")
ADD_SUBDIRECTORY(${ARTOS_IMAGE_REPOSITORY_SRC})
# Add compiler flags to build static libraries as place-independent code (pic),
# so that it can be linked to our shared ARTOS library.
if(UNIX)
    SET_TARGET_PROPERTIES(imagenet PROPERTIES COMPILE_FLAGS "-fpic")
endif(UNIX)


#### Build ARTOS shared library ####

# List files and set properties
SET(SOURCES defs.cc DPMDetection.cc FeatureExtractor.cc FeaturePyramid.cc HOGFeatureExtractor.cc JPEGImage.cc
ModelLearnerBase.cc ModelLearner.cc ImageNetModelLearner.cc Mixture.cc Model.cc ModelEvaluator.cc
Object.cc Patchwork.cc Random.cc Rectangle.cc Scene.cc StationaryBackground.cc
blf.cc harmony_search.cc sysutils.cc strutils.cc timingtools.cc)
ADD_LIBRARY(artos SHARED ${SOURCES} libartos.cc)
SET_TARGET_PROPERTIES(artos PROPERTIES VERSION ${BUILD_VERSION} SOVERSION ${API_VERSION})

# Link sub-libraries
IF(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC)
    TARGET_LINK_LIBRARIES(artos LINK_PRIVATE -Wl,-whole-archive imagenet -Wl,-no-whole-archive)
ELSEIF("${CMAKE_SYSTEM}" MATCHES IRIX)
    TARGET_LINK_LIBRARIES(artos LINK_PRIVATE -Wl,-all imagenet -Wl,-notall)
ELSE()
    TARGET_LINK_LIBRARIES(artos LINK_PRIVATE imagenet)
ENDIF()


#### Find and link external libraries ####

# There is no CMake Eigen package, so find it ourselves
FILE(GLOB EIGEN_ARCHIVE "eigen*")
FIND_PATH(EIGEN_INCLUDE_DIR Eigen ${EIGEN_ARCHIVE} .)
IF(NOT EIGEN_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "Could not find Eigen.")
ENDIF()
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# There is no CMake FFTW package, so find it ourselves
FIND_PATH(FFTW3_INCLUDE_DIR fftw3.h)
FIND_LIBRARY(FFTW3_LIBRARIES fftw3f)
IF(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARIES)
  MESSAGE(FATAL_ERROR "Could not find fftw3.")
ENDIF()
INCLUDE_DIRECTORIES(${FFTW3_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(artos LINK_PUBLIC ${FFTW3_LIBRARIES})

FIND_PACKAGE(JPEG REQUIRED)
IF(JPEG_FOUND)
  INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(artos LINK_PUBLIC ${JPEG_LIBRARIES})
ENDIF()

FIND_PACKAGE(LibXml2 REQUIRED)
IF(LIBXML2_FOUND)
  INCLUDE_DIRECTORIES(${LIBXML2_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(artos LINK_PUBLIC ${LIBXML2_LIBRARIES})
  ADD_DEFINITIONS(${LIBXML2_DEFINITIONS})
ENDIF()



#### Detect and enable some useful compiler features ####

# OpenMP (not required, but strongly recommended on multi-core systems)
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF()

# SSE
# enabling SSE caused ugly access violations on Win32 and I don't know why at the moment
#include(CheckSSEFeatures)



#### Build tools ####

IF(ARTOS_BUILD_TOOLS AND (EXISTS "${CMAKE_SOURCE_DIR}/../tools/CMakeLists.txt"))
  ADD_SUBDIRECTORY("${CMAKE_SOURCE_DIR}/../tools" "tools")
ENDIF()
