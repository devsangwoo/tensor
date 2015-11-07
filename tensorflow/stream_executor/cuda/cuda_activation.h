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

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// This file contains APIs that assume a StreamExecutor is backed by CUDA.
// It reaches into the CUDA implementation to activate an underlying CUDA
// context.
//
<<<<<<< HEAD
// Having this file separate from cuda/cuda_gpu_executor.h means that dependent
=======
// Having this file separate from cuda_gpu_executor.h means that dependent
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// code does not also have to depend on cuda.h.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_ACTIVATION_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_ACTIVATION_H_

<<<<<<< HEAD
#include "tensorflow/stream_executor/gpu/gpu_activation.h"

namespace stream_executor {
=======
#include "tensorflow/stream_executor/cuda/multi_op_activation.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

class StreamExecutor;

namespace cuda {

<<<<<<< HEAD
using ScopedActivateExecutorContext = gpu::ScopedActivateExecutorContext;

}  // namespace cuda
}  // namespace stream_executor
=======
class CUDAExecutor;
class ScopedActivateContext;

// Activates a CUDA context within an enclosing scope.
class ScopedActivateExecutorContext {
 public:
  // Form that takes a CUDA executor implementation.
  explicit ScopedActivateExecutorContext(
      CUDAExecutor* cuda_exec, MultiOpActivation moa = MultiOpActivation::kNo);

  // Form that takes a pImpl executor and extracts a CUDA implementation --
  // fatal failure if it is not CUDA inside.
  explicit ScopedActivateExecutorContext(
      StreamExecutor* stream_exec,
      MultiOpActivation moa = MultiOpActivation::kNo);

  ~ScopedActivateExecutorContext();

 private:
  // The CUDA executor implementation whose context is activated.
  CUDAExecutor* cuda_exec_;

  // The cuda.h-using datatype that we wrap.
  ScopedActivateContext* driver_scoped_activate_context_;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedActivateExecutorContext);
};

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_ACTIVATION_H_
