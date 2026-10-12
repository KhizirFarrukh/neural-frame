#include <pybind11/pybind11.h>

#include "neuralframe/version.hpp"

namespace py = pybind11;

#ifndef NEURALFRAME_VERSION_INFO
#define NEURALFRAME_VERSION_INFO "dev"
#endif

PYBIND11_MODULE(_native, m) {
  m.doc() = "NeuralFrame native hot-path operations";
  m.attr("__version__") = NEURALFRAME_VERSION_INFO;
  m.def("build_info", &nf::build_info, "Compiler and build flavour of the native module");
}
