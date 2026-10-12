cmake_minimum_required(VERSION 3.26)

project(neuralframe_native LANGUAGES CXX)

option(NEURALFRAME_BUILD_TESTS "Build the GoogleTest suite for the native module" OFF)

# GPU-side colour conversion, resizing and tile merging are implemented with PyTorch tensor ops
# (neuralframe/ops/torch_ops.py), so no CUDA toolkit is needed to build this module.
if(NEURALFRAME_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(native)
