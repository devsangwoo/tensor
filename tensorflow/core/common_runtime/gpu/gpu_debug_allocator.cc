<<<<<<< HEAD
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"

#include <cstddef>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"
=======
#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#define MASK_WORDS 2
#define MASK_BYTES (MASK_WORDS * sizeof(int64))

<<<<<<< HEAD
namespace tensorflow {
namespace {

int64* NewMask(int64 word) {
=======
namespace {

static int64* NewMask(int64 word) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  int64* m = new int64[MASK_WORDS];
  for (int i = 0; i < MASK_WORDS; ++i) {
    m[i] = word;
  }
  return m;
}

<<<<<<< HEAD
int64* before_mask = NewMask(0xabababababababab);
int64* after_mask = NewMask(0xcdcdcdcdcdcdcdcd);

bool CheckMask(se::StreamExecutor* exec, void* ptr, int64* mask) {
  se::DeviceMemory<int64> gpu_ptr{se::DeviceMemoryBase{ptr, MASK_BYTES}};
  int64 tmp[MASK_WORDS];

  Status result = exec->SynchronousMemcpyD2H(gpu_ptr, MASK_BYTES, tmp);
  if (!result.ok()) {
    LOG(FATAL) << "Could not copy debug mask, " << result;
=======
static int64* before_mask = NewMask(0xabababababababab);
static int64* after_mask = NewMask(0xcdcdcdcdcdcdcdcd);

bool CheckMask(perftools::gputools::StreamExecutor* exec, void* ptr,
               int64* mask) {
  gpu::DeviceMemory<int64> gpu_ptr{gpu::DeviceMemoryBase{ptr, MASK_BYTES}};
  int64 tmp[MASK_WORDS];

  if (!exec->SynchronousMemcpy(&tmp, gpu_ptr, MASK_BYTES)) {
    LOG(FATAL) << "Could not copy debug mask";
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  bool ok = true;
  for (int i = 0; i < MASK_WORDS; ++i) {
    ok &= (mask[i] == tmp[i]);
    if (!ok) {
      LOG(ERROR) << "i=" << i
                 << " mask=" << reinterpret_cast<const void*>(mask[i])
                 << " field=" << reinterpret_cast<const void*>(tmp[i]);
    }
  }

  return ok;
}

<<<<<<< HEAD
void InitMask(se::StreamExecutor* exec, void* ptr, int64* mask) {
  se::DeviceMemory<int64> gpu_ptr{se::DeviceMemoryBase{ptr, MASK_BYTES}};
  Status result = exec->SynchronousMemcpyH2D(mask, MASK_BYTES, &gpu_ptr);
  if (!result.ok()) {
    LOG(FATAL) << "Could not copy debug mask, " << result;
=======
void InitMask(perftools::gputools::StreamExecutor* exec, void* ptr,
              int64* mask) {
  gpu::DeviceMemory<int64> gpu_ptr{gpu::DeviceMemoryBase{ptr, MASK_BYTES}};
  if (!exec->SynchronousMemcpy(&gpu_ptr, mask, MASK_BYTES)) {
    LOG(FATAL) << "Could not copy debug mask";
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// GPUDebugAllocator
// -----------------------------------------------------------------------------
<<<<<<< HEAD
GPUDebugAllocator::GPUDebugAllocator(Allocator* allocator,
                                     PlatformGpuId platform_gpu_id)
    : base_allocator_(allocator) {
  stream_exec_ =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
=======
GPUDebugAllocator::GPUDebugAllocator(VisitableAllocator* allocator,
                                     int device_id)
    : base_allocator_(allocator) {
  stream_exec_ = GPUMachineManager()->ExecutorForDevice(device_id).ValueOrDie();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

GPUDebugAllocator::~GPUDebugAllocator() { delete base_allocator_; }

void* GPUDebugAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  num_bytes += (2 * MASK_BYTES);
<<<<<<< HEAD
  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);
  if (allocated_ptr == nullptr) return allocated_ptr;
=======

  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Return the pointer after the header
  void* rv = static_cast<char*>(allocated_ptr) + MASK_BYTES;

  // Write the header at allocated_ptr
  InitMask(stream_exec_, allocated_ptr, before_mask);

  // Write the footer at the end.
  size_t req_size = base_allocator_->RequestedSize(allocated_ptr);
  InitMask(stream_exec_,
           static_cast<char*>(allocated_ptr) + req_size - MASK_BYTES,
           after_mask);
  return rv;
}
void GPUDebugAllocator::DeallocateRaw(void* ptr) {
<<<<<<< HEAD
  if (ptr != nullptr) {
    CHECK(CheckHeader(ptr)) << "before_mask has been overwritten";
    CHECK(CheckFooter(ptr)) << "after_mask has been overwritten";

    // Backtrack to the beginning of the header.
    ptr = static_cast<void*>(static_cast<char*>(ptr) - MASK_BYTES);
  }
=======
  CHECK(CheckHeader(ptr)) << "before_mask has been overwritten";
  CHECK(CheckFooter(ptr)) << "after_mask has been overwritten";

  // Backtrack to the beginning of the header.
  ptr = static_cast<void*>(static_cast<char*>(ptr) - MASK_BYTES);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  // Deallocate the memory
  base_allocator_->DeallocateRaw(ptr);
}

<<<<<<< HEAD
bool GPUDebugAllocator::TracksAllocationSizes() const { return true; }

size_t GPUDebugAllocator::RequestedSize(const void* ptr) const {
  auto req_size = base_allocator_->RequestedSize(static_cast<const char*>(ptr) -
                                                 MASK_BYTES);
  return req_size - 2 * MASK_BYTES;
}

size_t GPUDebugAllocator::AllocatedSize(const void* ptr) const {
  return base_allocator_->AllocatedSize(static_cast<const char*>(ptr) -
                                        MASK_BYTES);
}

int64 GPUDebugAllocator::AllocationId(const void* ptr) const {
  return base_allocator_->AllocationId(static_cast<const char*>(ptr) -
                                       MASK_BYTES);
}

absl::optional<AllocatorStats> GPUDebugAllocator::GetStats() {
  return base_allocator_->GetStats();
}

void GPUDebugAllocator::ClearStats() { base_allocator_->ClearStats(); }
=======
void GPUDebugAllocator::AddAllocVisitor(Visitor visitor) {
  return base_allocator_->AddAllocVisitor(visitor);
}

void GPUDebugAllocator::AddFreeVisitor(Visitor visitor) {
  return base_allocator_->AddFreeVisitor(visitor);
}

bool GPUDebugAllocator::TracksAllocationSizes() { return true; }

size_t GPUDebugAllocator::RequestedSize(void* ptr) {
  auto req_size =
      base_allocator_->RequestedSize(static_cast<char*>(ptr) - MASK_BYTES);
  return req_size - 2 * MASK_BYTES;
}

size_t GPUDebugAllocator::AllocatedSize(void* ptr) {
  return base_allocator_->AllocatedSize(static_cast<char*>(ptr) - MASK_BYTES);
}
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

bool GPUDebugAllocator::CheckHeader(void* ptr) {
  return CheckMask(stream_exec_, static_cast<char*>(ptr) - MASK_BYTES,
                   before_mask);
}

bool GPUDebugAllocator::CheckFooter(void* ptr) {
  char* original_ptr = static_cast<char*>(ptr) - MASK_BYTES;
  size_t req_size = base_allocator_->RequestedSize(original_ptr);
  return CheckMask(stream_exec_, original_ptr + req_size - MASK_BYTES,
                   after_mask);
}

// -----------------------------------------------------------------------------
// GPUNanResetAllocator
// -----------------------------------------------------------------------------
<<<<<<< HEAD
GPUNanResetAllocator::GPUNanResetAllocator(Allocator* allocator,
                                           PlatformGpuId platform_gpu_id)
    : base_allocator_(allocator) {
  stream_exec_ =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
=======
GPUNanResetAllocator::GPUNanResetAllocator(VisitableAllocator* allocator,
                                           int device_id)
    : base_allocator_(allocator) {
  stream_exec_ = GPUMachineManager()->ExecutorForDevice(device_id).ValueOrDie();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

GPUNanResetAllocator::~GPUNanResetAllocator() { delete base_allocator_; }

void* GPUNanResetAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);
<<<<<<< HEAD
  if (allocated_ptr == nullptr) return allocated_ptr;

  // Initialize the buffer to Nans
  size_t req_size = base_allocator_->RequestedSize(allocated_ptr);
  std::vector<float> nans((req_size + sizeof(float) - 1) / sizeof(float),
                          std::nanf(""));
  se::DeviceMemory<float> nan_ptr{
      se::DeviceMemoryBase{static_cast<float*>(allocated_ptr), req_size}};

  Status result =
      stream_exec_->SynchronousMemcpyH2D(&nans[0], req_size, &nan_ptr);
  if (!result.ok()) {
    LOG(ERROR) << "Could not initialize to NaNs, " << result;
=======

  // Initialize the buffer to Nans
  size_t req_size = base_allocator_->RequestedSize(allocated_ptr);
  std::vector<float> nans(req_size / sizeof(float), std::nanf(""));
  gpu::DeviceMemory<float> nan_ptr{
      gpu::DeviceMemoryBase{static_cast<float*>(allocated_ptr), req_size}};

  if (!stream_exec_->SynchronousMemcpy(&nan_ptr, &nans[0], req_size)) {
    LOG(ERROR) << "Could not initialize to NaNs";
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  return allocated_ptr;
}
void GPUNanResetAllocator::DeallocateRaw(void* ptr) {
<<<<<<< HEAD
  if (ptr != nullptr) {
    // Reset the buffer to Nans
    size_t req_size = base_allocator_->RequestedSize(ptr);
    std::vector<float> nans((req_size + sizeof(float) - 1) / sizeof(float),
                            std::nanf(""));
    se::DeviceMemory<float> nan_ptr{
        se::DeviceMemoryBase{static_cast<float*>(ptr), req_size}};
    Status result =
        stream_exec_->SynchronousMemcpyH2D(&nans[0], req_size, &nan_ptr);
    if (!result.ok()) {
      LOG(ERROR) << "Could not initialize to NaNs, " << result;
    }
=======
  // Reset the buffer to Nans
  size_t req_size = base_allocator_->RequestedSize(ptr);
  std::vector<float> nans(req_size / sizeof(float), std::nanf(""));
  gpu::DeviceMemory<float> nan_ptr{
      gpu::DeviceMemoryBase{static_cast<float*>(ptr), req_size}};
  if (!stream_exec_->SynchronousMemcpy(&nan_ptr, &nans[0], req_size)) {
    LOG(ERROR) << "Could not initialize to NaNs";
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  // Deallocate the memory
  base_allocator_->DeallocateRaw(ptr);
}

<<<<<<< HEAD
size_t GPUNanResetAllocator::RequestedSize(const void* ptr) const {
  return base_allocator_->RequestedSize(ptr);
}

size_t GPUNanResetAllocator::AllocatedSize(const void* ptr) const {
  return base_allocator_->AllocatedSize(ptr);
}

absl::optional<AllocatorStats> GPUNanResetAllocator::GetStats() {
  return base_allocator_->GetStats();
}

void GPUNanResetAllocator::ClearStats() { base_allocator_->ClearStats(); }
=======
void GPUNanResetAllocator::AddAllocVisitor(Visitor visitor) {
  return base_allocator_->AddAllocVisitor(visitor);
}

void GPUNanResetAllocator::AddFreeVisitor(Visitor visitor) {
  return base_allocator_->AddFreeVisitor(visitor);
}

size_t GPUNanResetAllocator::RequestedSize(void* ptr) {
  return base_allocator_->RequestedSize(ptr);
}

size_t GPUNanResetAllocator::AllocatedSize(void* ptr) {
  return base_allocator_->AllocatedSize(ptr);
}
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace tensorflow
