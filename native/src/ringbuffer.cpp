#include "neuralframe/ringbuffer.hpp"

#include <chrono>
#include <cstdlib>
#include <new>
#include <stdexcept>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace nf {

namespace {

constexpr std::size_t kAlign = 4096;  // page alignment: required for efficient cudaHostRegister

std::uint8_t* aligned_alloc_bytes(std::size_t bytes) {
#if defined(_WIN32)
  void* p = _aligned_malloc(bytes, kAlign);
#else
  void* p = nullptr;
  if (posix_memalign(&p, kAlign, bytes) != 0) p = nullptr;
#endif
  if (p == nullptr) throw std::bad_alloc();
  return static_cast<std::uint8_t*>(p);
}

void aligned_free_bytes(std::uint8_t* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

template <typename Pred>
bool wait_for(std::condition_variable& cv, std::unique_lock<std::mutex>& lock, int timeout_ms, Pred pred) {
  if (timeout_ms < 0) {
    cv.wait(lock, pred);
    return true;
  }
  return cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), pred);
}

}  // namespace

RingBuffer::RingBuffer(std::size_t slots, std::size_t slot_bytes)
    : slot_bytes_((slot_bytes + kAlign - 1) / kAlign * kAlign),
      states_(slots, State::Free),
      tags_(slots, 0) {
  if (slots == 0 || slot_bytes == 0) throw std::invalid_argument("RingBuffer needs at least one non-empty slot");
  total_ = slot_bytes_ * slots;
  base_ = aligned_alloc_bytes(total_);
}

RingBuffer::~RingBuffer() { aligned_free_bytes(base_); }

int RingBuffer::acquire_write(int timeout_ms) {
  std::unique_lock<std::mutex> lock(mutex_);
  const bool ok = wait_for(cv_, lock, timeout_ms, [&] { return closed_ || states_[write_pos_] == State::Free; });
  if (!ok || closed_) return -1;
  const std::size_t slot = write_pos_;
  states_[slot] = State::Writing;
  write_pos_ = (write_pos_ + 1) % states_.size();
  return static_cast<int>(slot);
}

void RingBuffer::commit_write(int slot, std::int64_t tag) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot < 0 || static_cast<std::size_t>(slot) >= states_.size() || states_[slot] != State::Writing) {
      throw std::logic_error("commit_write on a slot that is not being written");
    }
    states_[slot] = State::Ready;
    tags_[slot] = tag;
    ++ready_;
  }
  cv_.notify_all();
}

int RingBuffer::acquire_read(int timeout_ms, std::int64_t* tag) {
  std::unique_lock<std::mutex> lock(mutex_);
  const bool ok = wait_for(cv_, lock, timeout_ms,
                           [&] { return states_[read_pos_] == State::Ready || (closed_ && ready_ == 0); });
  const std::size_t slot = read_pos_;
  if (!ok || states_[slot] != State::Ready) return -1;
  states_[slot] = State::Reading;
  --ready_;
  read_pos_ = (read_pos_ + 1) % states_.size();
  if (tag != nullptr) *tag = tags_[slot];
  return static_cast<int>(slot);
}

void RingBuffer::release_read(int slot) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot < 0 || static_cast<std::size_t>(slot) >= states_.size() || states_[slot] != State::Reading) {
      throw std::logic_error("release_read on a slot that is not being read");
    }
    states_[slot] = State::Free;
  }
  cv_.notify_all();
}

void RingBuffer::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
  }
  cv_.notify_all();
}

std::uint8_t* RingBuffer::slot_ptr(int slot) const {
  if (slot < 0 || static_cast<std::size_t>(slot) >= states_.size()) throw std::out_of_range("bad slot index");
  return base_ + static_cast<std::size_t>(slot) * slot_bytes_;
}

std::size_t RingBuffer::ready() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ready_;
}

bool RingBuffer::closed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return closed_;
}

}  // namespace nf
