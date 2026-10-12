set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NEURALFRAME_BUILD_PYTHON "Build the neuralframe._native Python extension" ON)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library (pure C++, no Python) — shared by the extension and the C++ tests
# ---------------------------------------------------------------------------
add_library(nf_core STATIC
  src/version.cpp
  src/threadpool.cpp
  src/color.cpp
  src/resample.cpp
  src/ringbuffer.cpp
)
target_include_directories(nf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nf_core PUBLIC Threads::Threads)

# Floating-point contraction (FMA) is disabled: kernels must reproduce the NumPy reference bit for bit.
add_library(nf_flags INTERFACE)
if(MSVC)
  target_compile_options(nf_flags INTERFACE /W4 /permissive- /O2 /fp:precise /Zc:__cplusplus /utf-8)
  target_compile_definitions(nf_flags INTERFACE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(nf_flags INTERFACE -Wall -Wextra -O3 -ffp-contract=off)
endif()
target_link_libraries(nf_core PRIVATE nf_flags)

# ---------------------------------------------------------------------------
# Python extension: neuralframe._native
# ---------------------------------------------------------------------------
if(NEURALFRAME_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native MODULE src/bindings.cpp)
  target_link_libraries(_native PRIVATE nf_core nf_flags)
  if(DEFINED SKBUILD_PROJECT_VERSION)
    target_compile_definitions(_native PRIVATE NEURALFRAME_VERSION_INFO="${SKBUILD_PROJECT_VERSION}")
  endif()
  install(TARGETS _native LIBRARY DESTINATION neuralframe RUNTIME DESTINATION neuralframe)
endif()

# ---------------------------------------------------------------------------
# GoogleTest suite (cmake -S . -B build/native-tests -DNEURALFRAME_BUILD_TESTS=ON -DNEURALFRAME_BUILD_PYTHON=OFF)
# ---------------------------------------------------------------------------
if(NEURALFRAME_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(nf_tests
    tests/test_color.cpp
    tests/test_resample.cpp
    tests/test_ringbuffer.cpp
    tests/test_threadpool.cpp
  )
  target_link_libraries(nf_tests PRIVATE nf_core nf_flags GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(nf_tests)
endif()
