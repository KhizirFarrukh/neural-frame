#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace nf {

// Fixed-capacity FIFO of equally sized frame slots backed by one page-aligned allocation.
//
// Producers acquire a free slot, fill it and commit; consumers acquire the oldest committed slot and
// release it when done. Slots are reused forever, so steady-state decoding performs no allocations, and a
// full ring blocks the producer (backpressure). The whole block can be page-locked once for fast,
// asynchronous host-to-device copies (cudaHostRegister on data()/bytes() from Python).
class RingBuffer {
 public:
  RingBuffer(std::size_t slots, std::size_t slot_bytes);
  ~RingBuffer();
  RingBuffer(const RingBuffer&) = delete;
  RingBuffer& operator=(const RingBuffer&) = delete;

  // Return a slot index, or -1 on timeout / when closed. timeout_ms < 0 waits forever.
  int acquire_write(int timeout_ms);
  void commit_write(int slot, std::int64_t tag);
  int acquire_read(int timeout_ms, std::int64_t* tag);
  void release_read(int slot);
  void close();

  std::uint8_t* slot_ptr(int slot) const;
  std::uint8_t* data() const { return base_; }
  std::size_t slot_bytes() const { return slot_bytes_; }
  std::size_t slots() const { return states_.size(); }
  std::size_t bytes() const { return total_; }
  std::size_t ready() const;
  bool closed() const;

 private:
  enum class State : std::uint8_t { Free, Writing, Ready, Reading };

  std::uint8_t* base_ = nullptr;
  std::size_t slot_bytes_ = 0;
  std::size_t total_ = 0;
  std::vector<State> states_;
  std::vector<std::int64_t> tags_;
  std::size_t write_pos_ = 0;  // next slot to hand to a producer
  std::size_t read_pos_ = 0;   // next slot a consumer will receive
  std::size_t ready_ = 0;
  bool closed_ = false;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace nf
