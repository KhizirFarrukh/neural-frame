cmake_minimum_required(VERSION 3.26)

project(neuralframe_native LANGUAGES CXX)

option(NEURALFRAME_BUILD_TESTS "Build the GoogleTest suite for the native module" OFF)
option(NEURALFRAME_ENABLE_CUDA "Build optional CUDA kernels when a CUDA toolkit is available" ON)

add_subdirectory(native)
