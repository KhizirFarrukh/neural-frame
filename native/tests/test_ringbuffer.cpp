#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "neuralframe/ringbuffer.hpp"

TEST(RingBuffer, FifoOrderAcrossThreads) {
  nf::RingBuffer rb(3, 64);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(rb.data()) % 4096, 0u);
  constexpr int kItems = 500;
  std::vector<std::int64_t> seen;
  std::thread consumer([&] {
    for (;;) {
      std::int64_t tag = 0;
      const int slot = rb.acquire_read(-1, &tag);
      if (slot < 0) break;
      std::int64_t payload = 0;
      std::memcpy(&payload, rb.slot_ptr(slot), sizeof(payload));
      EXPECT_EQ(payload, tag);
      seen.push_back(tag);
      rb.release_read(slot);
    }
  });
  for (std::int64_t i = 0; i < kItems; ++i) {
    const int slot = rb.acquire_write(-1);
    ASSERT_GE(slot, 0);
    std::memcpy(rb.slot_ptr(slot), &i, sizeof(i));
    rb.commit_write(slot, i);
  }
  rb.close();
  consumer.join();
  ASSERT_EQ(seen.size(), static_cast<std::size_t>(kItems));
  for (int i = 0; i < kItems; ++i) EXPECT_EQ(seen[i], i);
}

TEST(RingBuffer, FullRingBlocksProducerUntilTimeout) {
  nf::RingBuffer rb(2, 16);
  rb.commit_write(rb.acquire_write(0), 1);
  rb.commit_write(rb.acquire_write(0), 2);
  EXPECT_EQ(rb.acquire_write(20), -1);  // backpressure
  std::int64_t tag = 0;
  const int slot = rb.acquire_read(0, &tag);
  EXPECT_EQ(tag, 1);
  rb.release_read(slot);
  EXPECT_GE(rb.acquire_write(0), 0);
}

TEST(RingBuffer, CloseWakesWaiters) {
  nf::RingBuffer rb(2, 16);
  std::atomic<int> result{0};
  std::thread waiter([&] { result = rb.acquire_read(-1, nullptr); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  rb.close();
  waiter.join();
  EXPECT_EQ(result.load(), -1);
}
