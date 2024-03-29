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

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER2(BinaryOp, CPU, "Mod", functor::safe_mod, int32, int64);
REGISTER2(BinaryOp, CPU, "Mod", functor::fmod, float, double);
REGISTER2(BinaryOp, CPU, "TruncateMod", functor::safe_mod, int32, int64);
REGISTER2(BinaryOp, CPU, "TruncateMod", functor::fmod, float, double);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Mod")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::safe_mod<int32>>);
REGISTER_KERNEL_BUILDER(Name("TruncateMod")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::safe_mod<int32>>);
#endif
=======
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER2(BinaryOp, CPU, "Mod", functor::mod, int32, int64);
REGISTER2(BinaryOp, CPU, "Mod", functor::fmod, float, double);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
