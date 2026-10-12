#include "neuralframe/threadpool.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>

namespace nf {

namespace {

unsigned default_threads() {
  if (const char* env = std::getenv("NEURALFRAME_NATIVE_THREADS")) {
    const int n = std::atoi(env);
    if (n > 0) return static_cast<unsigned>(n);
  }
  const unsigned hw = std::thread::hardware_concurrency();
  return hw > 0 ? hw : 1;
}

struct Completion {
  std::mutex m;
  std::condition_variable cv;
  std::size_t remaining = 0;
  std::exception_ptr error;

  void done(std::exception_ptr err) {
    std::lock_guard<std::mutex> lock(m);
    if (err && !error) error = err;
    if (--remaining == 0) cv.notify_all();
  }
};

}  // namespace

ThreadPool& ThreadPool::instance() {
  // Intentionally leaked: joining threads from a static destructor inside a DLL can deadlock on the
  // Windows loader lock at interpreter shutdown. The OS reclaims the threads at process exit.
  static ThreadPool* pool = new ThreadPool(default_threads());
  return *pool;
}

ThreadPool::ThreadPool(unsigned n) { start(n); }

void ThreadPool::start(unsigned n) {
  threads_ = std::max(1u, n);
  stopping_ = false;
  for (unsigned i = 1; i < threads_; ++i) workers_.emplace_back([this] { worker_loop(); });
}

void ThreadPool::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
  workers_.clear();
}

void ThreadPool::set_threads(unsigned n) {
  n = std::max(1u, n);
  if (n == threads_) return;
  stop();
  start(n);
}

unsigned ThreadPool::threads() const { return threads_; }

void ThreadPool::worker_loop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (stopping_ && queue_.empty()) return;
      task = std::move(queue_.front());
      queue_.pop_front();
    }
    task();
  }
}

bool ThreadPool::run_one() {
  std::function<void()> task;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    task = std::move(queue_.front());
    queue_.pop_front();
  }
  task();
  return true;
}

void ThreadPool::parallel_for(std::size_t n, const std::function<void(std::size_t, std::size_t)>& fn) {
  if (n == 0) return;
  const std::size_t chunks = std::min<std::size_t>(n, threads_);
  if (chunks <= 1) {
    fn(0, n);
    return;
  }
  auto completion = std::make_shared<Completion>();
  completion->remaining = chunks - 1;
  const std::size_t base = n / chunks;
  const std::size_t extra = n % chunks;
  std::size_t begin = 0;
  std::size_t first_end = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t c = 0; c < chunks; ++c) {
      const std::size_t len = base + (c < extra ? 1 : 0);
      const std::size_t end = begin + len;
      if (c == 0) {
        first_end = end;
      } else {
        queue_.emplace_back([completion, &fn, begin, end] {
          std::exception_ptr err;
          try {
            fn(begin, end);
          } catch (...) {
            err = std::current_exception();
          }
          completion->done(err);
        });
      }
      begin = end;
    }
  }
  cv_.notify_all();

  std::exception_ptr own_error;
  try {
    fn(0, first_end);
  } catch (...) {
    own_error = std::current_exception();
  }
  // Help with queued work (ours or another caller's) until our chunks are finished.
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(completion->m);
      if (completion->remaining == 0) break;
    }
    if (!run_one()) {
      std::unique_lock<std::mutex> lock(completion->m);
      completion->cv.wait(lock, [&] { return completion->remaining == 0; });
      break;
    }
  }
  if (own_error) std::rethrow_exception(own_error);
  if (completion->error) std::rethrow_exception(completion->error);
}

}  // namespace nf
