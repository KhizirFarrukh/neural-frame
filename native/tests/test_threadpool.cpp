#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "neuralframe/threadpool.hpp"

TEST(ThreadPool, CoversEveryIndexExactlyOnce) {
  std::vector<std::atomic<int>> hits(10007);
  nf::ThreadPool::instance().parallel_for(hits.size(), [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) hits[i]++;
  });
  for (auto& h : hits) EXPECT_EQ(h.load(), 1);
}

TEST(ThreadPool, ConcurrentCallersDoNotDeadlock) {
  std::atomic<long long> total{0};
  std::vector<std::thread> callers;
  for (int t = 0; t < 4; ++t) {
    callers.emplace_back([&] {
      for (int rep = 0; rep < 50; ++rep) {
        nf::ThreadPool::instance().parallel_for(1000, [&](std::size_t b, std::size_t e) {
          total += static_cast<long long>(e - b);
        });
      }
    });
  }
  for (auto& c : callers) c.join();
  EXPECT_EQ(total.load(), 4LL * 50 * 1000);
}

TEST(ThreadPool, ExceptionsPropagateToCaller) {
  EXPECT_THROW(nf::ThreadPool::instance().parallel_for(64,
                                                       [](std::size_t, std::size_t e) {
                                                         if (e == 64) throw std::runtime_error("boom");
                                                       }),
               std::runtime_error);
}

TEST(ThreadPool, ResizeKeepsWorking) {
  auto& pool = nf::ThreadPool::instance();
  const unsigned before = pool.threads();
  pool.set_threads(2);
  std::atomic<int> n{0};
  pool.parallel_for(100, [&](std::size_t b, std::size_t e) { n += static_cast<int>(e - b); });
  EXPECT_EQ(n.load(), 100);
  pool.set_threads(before);
}
