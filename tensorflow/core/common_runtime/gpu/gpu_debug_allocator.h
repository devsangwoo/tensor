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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEBUG_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEBUG_ALLOCATOR_H_
=======
#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEBUG_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEBUG_ALLOCATOR_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#include <memory>
#include <string>
#include <unordered_map>
<<<<<<< HEAD

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
=======
#include <vector>

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/common_runtime/gpu/visitable_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

// An allocator that wraps a GPU allocator and adds debugging
// functionality that verifies that users do not write outside their
// allocated memory.
<<<<<<< HEAD
class GPUDebugAllocator : public Allocator {
 public:
  explicit GPUDebugAllocator(Allocator* allocator,
                             PlatformGpuId platform_gpu_id);
=======
class GPUDebugAllocator : public VisitableAllocator {
 public:
  explicit GPUDebugAllocator(VisitableAllocator* allocator, int device_id);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  ~GPUDebugAllocator() override;
  string Name() override { return "gpu_debug"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
<<<<<<< HEAD
  bool TracksAllocationSizes() const override;
  size_t RequestedSize(const void* ptr) const override;
  size_t AllocatedSize(const void* ptr) const override;
  int64 AllocationId(const void* ptr) const override;
  absl::optional<AllocatorStats> GetStats() override;
  void ClearStats() override;
=======
  void AddAllocVisitor(Visitor visitor) override;
  void AddFreeVisitor(Visitor visitor) override;
  bool TracksAllocationSizes() override;
  size_t RequestedSize(void* ptr) override;
  size_t AllocatedSize(void* ptr) override;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // For testing.
  bool CheckHeader(void* ptr);
  bool CheckFooter(void* ptr);

 private:
<<<<<<< HEAD
  Allocator* base_allocator_ = nullptr;  // owned

  se::StreamExecutor* stream_exec_;  // Not owned.
=======
  VisitableAllocator* base_allocator_ = nullptr;  // owned

  perftools::gputools::StreamExecutor* stream_exec_;  // Not owned.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  TF_DISALLOW_COPY_AND_ASSIGN(GPUDebugAllocator);
};

// An allocator that wraps a GPU allocator and resets the memory on
// allocation and free to 'NaN', helping to identify cases where the
// user forgets to initialize the memory.
<<<<<<< HEAD
class GPUNanResetAllocator : public Allocator {
 public:
  explicit GPUNanResetAllocator(Allocator* allocator,
                                PlatformGpuId platform_gpu_id);
=======
class GPUNanResetAllocator : public VisitableAllocator {
 public:
  explicit GPUNanResetAllocator(VisitableAllocator* allocator, int device_id);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  ~GPUNanResetAllocator() override;
  string Name() override { return "gpu_nan_reset"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
<<<<<<< HEAD
  size_t RequestedSize(const void* ptr) const override;
  size_t AllocatedSize(const void* ptr) const override;
  absl::optional<AllocatorStats> GetStats() override;
  void ClearStats() override;

 private:
  Allocator* base_allocator_ = nullptr;  // owned

  se::StreamExecutor* stream_exec_;  // Not owned.
=======
  void AddAllocVisitor(Visitor visitor) override;
  void AddFreeVisitor(Visitor visitor) override;
  size_t RequestedSize(void* ptr) override;
  size_t AllocatedSize(void* ptr) override;

 private:
  VisitableAllocator* base_allocator_ = nullptr;  // owned

  perftools::gputools::StreamExecutor* stream_exec_;  // Not owned.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  TF_DISALLOW_COPY_AND_ASSIGN(GPUNanResetAllocator);
};

}  // namespace tensorflow

<<<<<<< HEAD
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEBUG_ALLOCATOR_H_
=======
#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEBUG_ALLOCATOR_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
