set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library (pure C++, no Python) — shared by the extension and the C++ tests
# ---------------------------------------------------------------------------
add_library(nf_core STATIC
  src/version.cpp
)
target_include_directories(nf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nf_core PUBLIC Threads::Threads)

add_library(_native_flags INTERFACE)
if(MSVC)
  target_compile_options(nf_core PRIVATE /W4 /permissive- /O2 /fp:precise /Zc:__cplusplus /utf-8)
  target_compile_options(_native_flags INTERFACE /Zc:__cplusplus /utf-8)
else()
  target_compile_options(nf_core PRIVATE -Wall -Wextra -O3)
endif()

# ---------------------------------------------------------------------------
# Python extension: neuralframe._native
# ---------------------------------------------------------------------------
pybind11_add_module(_native MODULE src/bindings.cpp)
target_link_libraries(_native PRIVATE nf_core _native_flags)
target_compile_definitions(_native PRIVATE NEURALFRAME_VERSION_INFO="${SKBUILD_PROJECT_VERSION}")

install(TARGETS _native LIBRARY DESTINATION neuralframe RUNTIME DESTINATION neuralframe)
