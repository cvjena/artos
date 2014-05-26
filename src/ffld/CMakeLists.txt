#                          Implementation of the paper
#                "Exact Acceleration of Linear Object Detectors"
#              12th European Conference on Computer Vision, 2012.
#
#      Copyright (c) 2012 Idiap Research Institute, <http://www.idiap.ch/>
#              Written by Charles Dubout <charles.dubout@idiap.ch>

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(FFLD)

# The build options
OPTION(FFLD_HOGPYRAMID_FELZENSZWALB_FEATURES "Use Felzenszwalb's original features (slower and not as accurate, provided for compatibility only)." OFF)
OPTION(FFLD_HOGPYRAMID_DOUBLE "Use doubles instead of floats (slower, uses twice more memory, and the increase in precision is not necessarily useful)." OFF)

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

SET(CMAKE_CXX_FLAGS_DEBUG "-pg")

# Also list the headers so that they are displayed along the .cpp files in the IDE
SET(HEADERS Rectangle.h JPEGImage.h Object.h Scene.h HOGPyramid.h Patchwork.h Model.h Mixture.h Intersector.h timingtools.h)
SET(SOURCES Rectangle.cpp JPEGImage.cpp Object.cpp Scene.cpp HOGPyramid.cpp
Patchwork.cpp Model.cpp Mixture.cpp timingtools.cpp)
ADD_EXECUTABLE(detect_ffld ${HEADERS} ${SOURCES} ffld.cpp)

# Define the options
IF(FFLD_HOGPYRAMID_FELZENSZWALB_FEATURES)
  ADD_DEFINITIONS(-DFFLD_HOGPYRAMID_FELZENSZWALB_FEATURES)
ENDIF()

IF(FFLD_HOGPYRAMID_DOUBLE)
  ADD_DEFINITIONS(-DFFLD_HOGPYRAMID_DOUBLE)
ENDIF()

# There are no CMake Eigen package, so find it ourselves
FILE(GLOB EIGEN_ARCHIVE "eigen*")
FIND_PATH(EIGEN_INCLUDE_DIR Eigen ${EIGEN_ARCHIVE} .)
IF(NOT EIGEN_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "Could not find Eigen.")
ENDIF()
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# There are no CMake FFTW package, so find it ourselves
FIND_PATH(FFTW3_INCLUDE_DIR fftw3.h)
IF(FFLD_HOGPYRAMID_DOUBLE)
  FIND_LIBRARY(FFTW3_LIBRARIES fftw3)
ELSE()
  FIND_LIBRARY(FFTW3_LIBRARIES fftw3f)
ENDIF()
IF(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARIES)
  MESSAGE(FATAL_ERROR "Could not find fftw3.")
ENDIF()
INCLUDE_DIRECTORIES(${FFTW3_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(detect_ffld ${FFTW3_LIBRARIES})

FIND_PACKAGE(JPEG REQUIRED)
IF(JPEG_FOUND)
  INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(detect_ffld ${JPEG_LIBRARIES})
ENDIF()

FIND_PACKAGE(LibXml2 REQUIRED)
IF(LIBXML2_FOUND)
  INCLUDE_DIRECTORIES(${LIBXML2_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(detect_ffld ${LIBXML2_LIBRARIES})
  ADD_DEFINITIONS(${LIBXML2_DEFINITIONS})
ENDIF()

# Not required, but strongly recommended on multi-core systems
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF()

ADD_LIBRARY(ffld ${SOURCES})
