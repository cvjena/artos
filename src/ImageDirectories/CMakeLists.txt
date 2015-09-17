CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ImageDirectories)

INCLUDE_DIRECTORIES("${ARTOS_SOURCE_DIR}")

SET(SOURCES ImageRepository.cc SynsetImage.cc SynsetIterators.cc)
ADD_LIBRARY(imagenet ${SOURCES})
