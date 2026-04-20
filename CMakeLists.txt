cmake_minimum_required(VERSION 3.16)
project(WirelessTransmission)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(CUDA QUIET)

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

# RL Stabilizer — Phase Correlation + ONNX Policy (optional, needs onnxruntime)
find_library(ONNXRUNTIME_LIB onnxruntime HINTS /usr/local/lib $ENV{HOME}/local/lib)
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h HINTS /usr/local/include $ENV{HOME}/local/include)
if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
    add_executable(rl_stabilizer rl_stabilizer.cpp)
    target_include_directories(rl_stabilizer PRIVATE ${ONNXRUNTIME_INCLUDE})
    target_link_libraries(rl_stabilizer ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})
    message(STATUS "ONNX Runtime found — building rl_stabilizer")
else()
    message(STATUS "ONNX Runtime not found — skipping rl_stabilizer (install onnxruntime to enable)")
endif()

# DIC Analysis — cuFFT Phase Correlation (needs CUDA toolkit)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_executable(dic_analysis dic_analysis.cu)
    target_link_libraries(dic_analysis ${OpenCV_LIBS} CUDA::cufft CUDA::cudart)
    set_target_properties(dic_analysis PROPERTIES CUDA_ARCHITECTURES "70")
    message(STATUS "CUDA toolkit found — building dic_analysis")
else()
    message(STATUS "CUDA toolkit not found — skipping dic_analysis")
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
