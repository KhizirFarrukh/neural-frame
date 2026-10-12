#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace nf {

// Persistent worker pool shared by all native ops.
//
// parallel_for() may be called concurrently from several Python threads (e.g. the preprocess and
// postprocess stages): each call enqueues its own chunks and waits on its own counter, and the calling
// thread helps drain the queue instead of blocking idle. Work items only touch disjoint output rows, so
// results never depend on scheduling (outputs are bit-identical for any thread count).
class ThreadPool {
 public:
  static ThreadPool& instance();

  void parallel_for(std::size_t n, const std::function<void(std::size_t, std::size_t)>& fn);
  void set_threads(unsigned n);
  unsigned threads() const;

 private:
  explicit ThreadPool(unsigned n);
  void start(unsigned n);
  void stop();
  void worker_loop();
  bool run_one();

  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool stopping_ = false;
  unsigned threads_ = 1;
};

inline void parallel_rows(std::size_t rows, const std::function<void(std::size_t, std::size_t)>& fn) {
  ThreadPool::instance().parallel_for(rows, fn);
}

}  // namespace nf
