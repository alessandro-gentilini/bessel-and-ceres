CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(bessel)

FIND_PACKAGE(Ceres REQUIRED)
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})

# helloworld
ADD_EXECUTABLE(bessel bessel.cc)
TARGET_LINK_LIBRARIES(bessel ${CERES_LIBRARIES})
SET(CMAKE_CXX_FLAGS "-std=c++11")
