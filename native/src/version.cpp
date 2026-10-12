#include "neuralframe/version.hpp"

#include <sstream>
#include <thread>

namespace nf {

std::string build_info() {
  std::ostringstream os;
#if defined(_MSC_VER)
  os << "MSVC " << _MSC_VER;
#elif defined(__clang__)
  os << "clang " << __clang_major__ << "." << __clang_minor__;
#elif defined(__GNUC__)
  os << "gcc " << __GNUC__ << "." << __GNUC_MINOR__;
#else
  os << "unknown compiler";
#endif
  os << ", C++" << (__cplusplus / 100 % 100) << ", hw threads " << std::thread::hardware_concurrency();
  return os.str();
}

}  // namespace nf
