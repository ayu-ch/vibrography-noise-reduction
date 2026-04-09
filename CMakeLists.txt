cmake_minimum_required(VERSION 3.16)
project(WirelessTransmission)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# ArUco Homography Stabilizer
add_executable(aruco_homography aruco_homography.cpp)
target_link_libraries(aruco_homography 
    ${OpenCV_LIBS}
)

# RANSAC Stabilizer with ArUco Displacement Tracking
add_executable(ransac_stabilizer ransac_stabilizer.cpp)
target_link_libraries(ransac_stabilizer 
    ${OpenCV_LIBS}
)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
